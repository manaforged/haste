use anyhow::{Result, bail};
use rustc_lexer::TokenKind;

use crate::body::adt::{AdtDef, Shape, VariantDef, adt};
use crate::body::sig::GenericKind;
use crate::graph::cfg::Cfg;
use crate::names::Primitive;

use super::attr::split_commas;
use super::item::{Attr, Item, ItemKind};
use super::source::{Source, Tokens};
use super::tree::Env;

/// The dependencies, in order of preference, whose `$crate` names core's traits in a derived impl: core, the shim that standard crates depend on in its place, and std.
const CORE_HOLDERS: [&str; 3] = ["core", "rustc_std_workspace_core", "std"];

/// A trait whose impl `#[derive]` writes by the compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derive {
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Default,
}

/// Each builtin derive with the name it is written by and the path of its trait under the core crate.
const DERIVES: [(&str, Derive, &str); 9] = [
    ("Clone", Derive::Clone, "clone::Clone"),
    ("Copy", Derive::Copy, "marker::Copy"),
    ("PartialEq", Derive::PartialEq, "cmp::PartialEq"),
    ("Eq", Derive::Eq, "cmp::Eq"),
    ("PartialOrd", Derive::PartialOrd, "cmp::PartialOrd"),
    ("Ord", Derive::Ord, "cmp::Ord"),
    ("Hash", Derive::Hash, "hash::Hash"),
    ("Debug", Derive::Debug, "fmt::Debug"),
    ("Default", Derive::Default, "default::Default"),
];

/// The derives that the attributes `attrs` in `src` ask for, through every `cfg_attr` whose predicate `holds`: the builtin ones, and the paths from first to last token of the others.
pub fn derives(src: &Source, attrs: &[Attr], holds: &dyn Fn(&Cfg) -> bool, into: &mut (Vec<Derive>, Vec<(u32, u32)>)) {
    for attr in attrs {
        match attr {
            Attr::Derive(open) => {
                for (from, to) in split_commas(src, open + 1, src.pair(*open)) {
                    let Some(last) = (from..to).rev().find(|at| matches!(src.kind(*at), TokenKind::Ident | TokenKind::RawIdent)) else { continue };
                    match DERIVES.iter().find(|(name, ..)| *name == src.text(last)) {
                        Some(&(_, derive, _)) => into.0.push(derive),
                        None => into.1.push((from, last)),
                    }
                }
            }
            Attr::CfgAttr(cfg, inner) if holds(cfg) => derives(src, inner, holds, into),
            _ => {}
        }
    }
}

/// The impls of the builtin derives that `item` of `src` asks for in crate `krate` with dependencies `deps` under `env`, and the paths of the other derives it asks for.
pub fn derived(src: &Source, item: &Item, env: &Env, (krate, deps): (usize, &[(String, usize)])) -> (Option<Result<Source>>, Vec<(u32, u32)>) {
    let mut found = (Vec::new(), Vec::new());
    derives(src, &item.attrs, &|cfg| env.allows(cfg), &mut found);
    let (wanted, foreign) = found;
    if wanted.is_empty() {
        return (None, foreign);
    }
    let named = |name: &&str| deps.iter().find(|(dep, _)| dep == name).map(|(_, at)| *at);
    let holder = CORE_HOLDERS.iter().find_map(named).unwrap_or(krate) as u32;
    (Some(expand(src, item, (&wanted, holder), &|attrs| env.enabled(attrs))), foreign)
}

/// The impls of the builtin derives `wanted` for the struct, enum or union `item` of `src`, naming core's traits through `$crate` of crate `holder`, over the variants and fields that `enabled` keeps.
pub fn expand(src: &Source, item: &Item, (wanted, holder): (&[Derive], u32), enabled: &dyn Fn(&[Attr]) -> bool) -> Result<Source> {
    let Some(name) = item.name else { bail!("a derive on an item with no name") };
    let mut def = adt(src, name, item.end, item.kind == ItemKind::Enum)?;
    def.configure(src, enabled);
    let simple = wanted.contains(&Derive::Copy) && !def.generics.params.iter().any(|param| param.kind == GenericKind::Type);
    let union = item.kind == ItemKind::Union;
    let mut writer = Writer { src, name: src.text(name).to_owned(), is_enum: item.kind == ItemKind::Enum, repr: repr(src, &item.attrs), def, holder, out: Tokens::default() };
    for &derive in wanted {
        writer.header(derive);
        let body = writer.method(derive, simple || union)?;
        writer.text("{");
        writer.text(&body);
        writer.text("}");
    }
    writer.out.finish(src.path.clone())
}

fn repr(src: &Source, attrs: &[Attr]) -> String {
    let words = attrs.iter().filter_map(|attr| match *attr {
        Attr::Other(at) if src.text(at) == "repr" && src.kind(at + 1) == TokenKind::OpenParen => Some(split_commas(src, at + 2, src.pair(at + 1))),
        _ => None,
    });
    let int = words.flatten().map(|(from, _)| src.text(from)).find(|word| Primitive::named(word).is_some());
    int.unwrap_or("isize").to_owned()
}

struct Writer<'a> {
    src: &'a Source,
    def: AdtDef,
    name: String,
    is_enum: bool,
    repr: String,
    holder: u32,
    out: Tokens,
}

impl Writer<'_> {
    fn text(&mut self, text: &str) {
        for (index, piece) in text.split("$crate").enumerate() {
            if index > 0 {
                self.out.dollar_crate(self.holder);
            }
            self.out.lexed(piece);
        }
    }

    fn copy(&mut self, (from, to): (u32, u32)) {
        (from..to).for_each(|at| self.out.copy(self.src, at));
        self.out.space();
    }

    fn header(&mut self, derive: Derive) {
        let path = DERIVES.iter().find(|(_, each, _)| *each == derive).map_or("", |(.., path)| path);
        let params = self.def.generics.params.clone();
        self.text("impl <");
        params.iter().for_each(|param| {
            self.copy(param.span);
            self.text(",");
        });
        self.text(&format!("> $crate::{path} for {} <", self.name));
        params.iter().for_each(|param| {
            self.copy((param.name, param.name + 1));
            self.text(",");
        });
        self.text("> where");
        for param in params.iter().filter(|param| param.kind == GenericKind::Type) {
            self.copy((param.name, param.name + 1));
            self.text(&format!(": $crate::{path},"));
        }
        for span in self.def.wheres.clone() {
            self.copy(span);
            if self.src.kind(span.1 - 1) != TokenKind::Comma {
                self.text(",");
            }
        }
    }

    fn method(&self, derive: Derive, simple: bool) -> Result<String> {
        Ok(match derive {
            Derive::Copy | Derive::Eq => String::new(),
            Derive::Clone => format!("fn clone(&self) -> Self {{ {} }}", if simple { "*self".to_owned() } else { self.clone_body() }),
            Derive::PartialEq => format!("fn eq(&self, other: &Self) -> bool {{ {} }}", self.eq_body()),
            Derive::PartialOrd => {
                let body = self.order_body(("PartialOrd", "partial_cmp"), "$crate::option::Option::Some($crate::cmp::Ordering::Equal)")?;
                format!("fn partial_cmp(&self, other: &Self) -> $crate::option::Option<$crate::cmp::Ordering> {{ {body} }}")
            }
            Derive::Ord => format!("fn cmp(&self, other: &Self) -> $crate::cmp::Ordering {{ {} }}", self.order_body(("Ord", "cmp"), "$crate::cmp::Ordering::Equal")?),
            Derive::Hash => format!("fn hash<__H: $crate::hash::Hasher>(&self, state: &mut __H) {{ {} }}", self.hash_body()?),
            Derive::Debug => format!("fn fmt(&self, f: &mut $crate::fmt::Formatter<'_>) -> $crate::fmt::Result {{ {} }}", self.debug_body()),
            Derive::Default => format!("fn default() -> Self {{ {} }}", self.default_body()?),
        })
    }

    fn path(&self, variant: &VariantDef) -> String {
        match self.is_enum {
            true => format!("{}::{}", self.name, self.src.text(variant.name)),
            false => self.name.clone(),
        }
    }

    fn pattern(&self, variant: &VariantDef, prefix: &str) -> String {
        let binds = (0..variant.fields.len()).map(|at| format!("{prefix}_{at}"));
        self.build(variant, binds.collect())
    }

    fn build(&self, variant: &VariantDef, parts: Vec<String>) -> String {
        let path = self.path(variant);
        match variant.shape {
            Shape::Unit => path,
            Shape::Tuple => format!("{path}({})", parts.join(", ")),
            Shape::Named => {
                let named = variant.fields.iter().zip(parts).map(|(field, part)| format!("{}: {part}", field.name.map_or("", |at| self.src.text(at))));
                format!("{path} {{ {} }}", named.collect::<Vec<_>>().join(", "))
            }
        }
    }

    fn arms(&self, arm: impl Fn(&VariantDef) -> String) -> String {
        match self.def.variants.is_empty() {
            true => "match *self {}".to_owned(),
            false => format!("match self {{ {} }}", self.def.variants.iter().map(arm).collect::<Vec<_>>().join(" ")),
        }
    }

    fn pairs(&self, arm: impl Fn(&VariantDef) -> String, otherwise: &str) -> String {
        if self.def.variants.is_empty() {
            return "match *self {}".to_owned();
        }
        let arms: Vec<String> = self.def.variants.iter().map(|variant| format!("({}, {}) => {},", self.pattern(variant, "__self"), self.pattern(variant, "__arg1"), arm(variant))).collect();
        let rest = if self.def.variants.len() > 1 { format!("_ => {otherwise},") } else { String::new() };
        format!("match (self, other) {{ {} {rest} }}", arms.join(" "))
    }

    fn clone_body(&self) -> String {
        self.arms(|variant| {
            let parts = (0..variant.fields.len()).map(|at| format!("$crate::clone::Clone::clone(__self_{at})"));
            format!("{} => {},", self.pattern(variant, "__self"), self.build(variant, parts.collect()))
        })
    }

    fn eq_body(&self) -> String {
        self.pairs(|variant| (0..variant.fields.len()).fold("true".to_owned(), |chain, at| format!("{chain} && *__self_{at} == *__arg1_{at}")), "false")
    }

    fn order_body(&self, (name, method): (&str, &str), equal: &str) -> Result<String> {
        let call = |at: usize| format!("$crate::cmp::{name}::{method}(__self_{at}, __arg1_{at})");
        let chain = |variant: &VariantDef| match variant.fields.len() {
            0 => equal.to_owned(),
            count => (0..count - 1).rev().fold(call(count - 1), |next, at| format!("match {} {{ {equal} => {next}, cmp => cmp }}", call(at))),
        };
        let otherwise = match self.def.variants.len() > 1 {
            true => format!("$crate::cmp::{name}::{method}(&{}, &{})", self.discriminant("self")?, self.discriminant("other")?),
            false => String::new(),
        };
        Ok(self.pairs(chain, &otherwise))
    }

    fn hash_body(&self) -> Result<String> {
        let tag = match self.def.variants.len() > 1 {
            true => format!("$crate::hash::Hash::hash(&{}, state);", self.discriminant("self")?),
            false => String::new(),
        };
        let fields = |variant: &VariantDef| {
            let each: String = (0..variant.fields.len()).map(|at| format!("$crate::hash::Hash::hash(__self_{at}, state);")).collect();
            format!("{} => {{ {each} }}", self.pattern(variant, "__self"))
        };
        Ok(format!("{tag} {}", self.arms(fields)))
    }

    fn debug_body(&self) -> String {
        self.arms(|variant| {
            let label = if self.is_enum { self.src.text(variant.name) } else { &self.name };
            let fields = variant.fields.iter().enumerate();
            let body = match variant.shape {
                Shape::Unit => format!("$crate::fmt::Formatter::write_str(f, \"{label}\")"),
                Shape::Tuple => format!("$crate::fmt::Formatter::debug_tuple(f, \"{label}\"){}.finish()", fields.map(|(at, _)| format!(".field(__self_{at})")).collect::<String>()),
                Shape::Named => {
                    let each: String = fields.map(|(at, field)| format!(".field(\"{}\", __self_{at})", field.name.map_or("", |name| self.src.text(name)))).collect();
                    format!("$crate::fmt::Formatter::debug_struct(f, \"{label}\"){each}.finish()")
                }
            };
            format!("{} => {body},", self.pattern(variant, "__self"))
        })
    }

    fn default_body(&self) -> Result<String> {
        let chosen = match self.is_enum {
            true => self.def.variants.iter().find(|variant| self.marked_default(variant)),
            false => self.def.variants.first(),
        };
        let Some(variant) = chosen else { bail!("derive(Default) on enum `{}` with no #[default] variant", self.name) };
        let parts = (0..variant.fields.len()).map(|_| "$crate::default::Default::default()".to_owned());
        Ok(self.build(variant, parts.collect()))
    }

    fn marked_default(&self, variant: &VariantDef) -> bool {
        variant.attrs.is_some_and(|start| (start..variant.name).any(|at| self.src.text(at) == "default" && self.src.kind(at.saturating_sub(1)) == TokenKind::OpenBracket))
    }

    fn discriminant(&self, of: &str) -> Result<String> {
        let explicit = self.def.variants.iter().any(|variant| variant.discriminant.is_some());
        if explicit && self.def.variants.iter().any(|variant| !variant.fields.is_empty()) {
            bail!("derive on enum `{}` with explicit discriminants and fields is not supported yet", self.name);
        }
        if explicit {
            return Ok(format!("(*{of} as {})", self.repr));
        }
        let arms = self.def.variants.iter().enumerate().map(|(at, variant)| {
            let shape = match variant.shape {
                Shape::Unit => self.path(variant),
                Shape::Tuple => format!("{}(..)", self.path(variant)),
                Shape::Named => format!("{} {{ .. }}", self.path(variant)),
            };
            format!("{shape} => {at}{},", self.repr)
        });
        Ok(format!("match {of} {{ {} }}", arms.collect::<String>()))
    }
}
