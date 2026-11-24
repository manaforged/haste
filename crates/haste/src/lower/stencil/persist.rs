use std::sync::{Arc, Mutex};

use anyhow::Result;
use haste_codegen::Function;

use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::substrate::tally::{Tally, tally};
use crate::substrate::{InlineKey, InlineShape, SubstrateBuilder, layer_item_key};
use crate::sync::lock;

use super::super::Slots;
use super::{Bodies, Form, InlineTemplate, TemplateKey};

const FORMS: [Form; 4] = [Form::Lean, Form::Loop, Form::Compact, Form::Dense];
const BODY_HEAD: usize = 9;

pub(super) struct Persist {
    mode: u64,
    verify: bool,
    fresh: Mutex<Vec<Fresh>>,
}

struct Fresh {
    sysroot: bool,
    key: InlineKey,
    shape: Option<InlineShape>,
    body: Option<Vec<u8>>,
}

pub(super) struct Disk<'b> {
    persist: &'b Persist,
    sysroot: bool,
    key: InlineKey,
    found: Option<InlineShape>,
    has_body: bool,
}

impl Disk<'_> {
    pub(super) fn hit(&self) -> Option<InlineShape> {
        if self.found.is_some() {
            tally(Tally::MetaHit);
        }
        self.found.filter(|_| !self.persist.verify)
    }

    /// The recorded template of this key: `Some(None)` when the callee has none, or `None` when it must be built.
    pub(super) fn template(
        &self,
        cx: &Context<'_>,
        (reached, slots): (&Reached, &dyn Slots),
    ) -> Option<Option<Arc<InlineTemplate>>> {
        let Some(shape) = self.hit()? else {
            return Some(None);
        };
        if !self.has_body {
            return None;
        }
        let bytes = cx
            .store()
            .substrate()
            .and_then(|substrate| substrate.inline_body(&self.key));
        let template = bytes
            .and_then(|bytes| decode(cx, (reached, slots), self.sysroot, bytes))
            .filter(|template| template.shape() == shape);
        super::tally(if template.is_some() {
            super::Tally::Disk
        } else {
            super::Tally::Stale
        });
        template.map(|template| Some(Arc::new(template)))
    }

    pub(super) fn settle(
        self,
        cx: &Context<'_>,
        (reached, slots): (&Reached, &dyn Slots),
        built: Option<&InlineTemplate>,
        (stored, served): (bool, bool),
    ) {
        let shape = built.map(InlineTemplate::shape);
        let fresh_shape = match self.found {
            Some(found) if (stored || served) && found != shape => {
                tally(Tally::MetaMismatch);
                assert!(
                    !self.persist.verify,
                    "inline metadata of {:?} is {found:?} in the substrate and {shape:?} when built",
                    self.key
                );
                return;
            }
            None if stored => {
                tally(Tally::MetaStored);
                Some(shape)
            }
            _ => None,
        };
        let wanted = stored && (!self.has_body || self.persist.verify);
        let body = built
            .filter(|_| wanted)
            .and_then(|template| encode(cx, (reached, slots), self.sysroot, template));
        if self.has_body
            && let Some(body) = &body
        {
            let same = cx
                .store()
                .substrate()
                .and_then(|substrate| substrate.inline_body(&self.key))
                == Some(body.as_slice());
            assert!(
                same,
                "inline template body of {:?} differs between the substrate and this build",
                self.key
            );
        }
        let body = body.filter(|_| !self.has_body);
        if fresh_shape.is_some() || body.is_some() {
            lock(&self.persist.fresh).push(Fresh {
                sysroot: self.sysroot,
                key: self.key,
                shape: fresh_shape,
                body,
            });
        }
    }
}

fn encode(
    cx: &Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    sysroot: bool,
    template: &InlineTemplate,
) -> Option<Vec<u8>> {
    let plan = slots.encode_plan(cx, reached, sysroot, &template.reloc_plan)?;
    let body = template.body.encode()?;
    let mut out = Vec::with_capacity(BODY_HEAD + plan.len() + body.len());
    out.extend_from_slice(&[u8::from(template.returns), template.height]);
    out.extend_from_slice(&template.size.to_le_bytes());
    out.push(template.form as u8);
    out.extend_from_slice(&u32::try_from(plan.len()).ok()?.to_le_bytes());
    out.extend_from_slice(&plan);
    out.extend_from_slice(&body);
    Some(out)
}

fn decode(
    cx: &Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    sysroot: bool,
    bytes: &[u8],
) -> Option<InlineTemplate> {
    let (&[returns, height, size_lo, size_hi, form, len @ ..], rest) =
        bytes.split_first_chunk::<BODY_HEAD>()?;
    let (plan, body) = rest.split_at_checked(u32::from_le_bytes(len) as usize)?;
    let form = *FORMS
        .get(usize::from(form))
        .filter(|found| **found as u8 == form)?;
    let reloc_plan = slots.decode_plan(cx, reached, sysroot, plan)?;
    let body = Function::decode(body)?;
    let returns = match returns {
        0 => false,
        1 => true,
        _ => return None,
    };
    Some(InlineTemplate {
        body: Arc::new(body),
        returns,
        reloc_plan: Arc::new(reloc_plan),
        height,
        size: u16::from_le_bytes([size_lo, size_hi]),
        form,
    })
}

impl Bodies {
    /// Starts reading and recording inline template outcomes and bodies in the substrate under item hashes XORed with `mode`; with `verify`, every template is still built and a differing recorded outcome panics.
    pub(crate) fn persist(&self, mode: u64, verify: bool) {
        self.persist.get_or_init(|| Persist {
            mode,
            verify,
            fresh: Mutex::default(),
        });
    }

    /// Drops every cached body, footprint plan and cycle entry and keeps the inline template outcomes recorded for the substrate.
    pub(crate) fn reset(&mut self) {
        let persist = std::mem::take(&mut self.persist);
        *self = Self {
            persist,
            ..Self::default()
        };
    }

    pub(super) fn disk(&self, cx: &Context<'_>, key: &TemplateKey) -> Option<Disk<'_>> {
        let persist = self.persist.get()?;
        if !key.footprint.is_portable() {
            return None;
        }
        let (sysroot, item) = layer_item_key(cx.outline, key.item)?;
        let key = InlineKey {
            item: item ^ persist.mode,
            footprint: key.footprint.0,
            limit: u32::try_from(key.limit).ok()?,
            low: key.low,
            sole: key.sole,
        };
        let substrate = cx.store().substrate();
        let found = substrate.and_then(|substrate| substrate.inline_meta(&key));
        let has_body = substrate.is_some_and(|substrate| substrate.inline_body(&key).is_some());
        tally(Tally::MetaLookup);
        if found.is_none() {
            tally(Tally::MetaMiss);
        }
        Some(Disk {
            persist,
            sysroot,
            key,
            found,
            has_body,
        })
    }

    /// Whether this build recorded inline template outcomes or bodies that the substrate does not hold yet.
    pub(crate) fn has_fresh_meta(&self) -> bool {
        self.persist
            .get()
            .is_some_and(|persist| !lock(&persist.fresh).is_empty())
    }

    /// Appends the inline template outcomes and bodies this build recorded for sysroot items into `toolchain` and for dependency items into `deps`.
    pub(crate) fn export_meta(
        &self,
        mut toolchain: Option<&mut SubstrateBuilder>,
        mut deps: Option<&mut SubstrateBuilder>,
    ) -> Result<()> {
        let Some(persist) = self.persist.get() else {
            return Ok(());
        };
        for fresh in lock(&persist.fresh).iter() {
            let target = if fresh.sysroot {
                toolchain.as_deref_mut()
            } else {
                deps.as_deref_mut()
            };
            let Some(builder) = target else { continue };
            if let Some(shape) = fresh.shape {
                builder.add_inline_meta(&fresh.key, shape)?;
            }
            if let Some(body) = &fresh.body {
                builder.add_inline_body(&fresh.key, body)?;
            }
        }
        Ok(())
    }
}
