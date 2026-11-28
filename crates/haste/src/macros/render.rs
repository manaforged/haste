use crate::outline::attr;
use crate::outline::item::Attr;
use crate::outline::source::Source;

use super::expand::push_layer_span;
use super::split_commas;

fn push_raw_span(parts: &mut Vec<String>, src: &Source, from: u32, to: u32) {
    if from >= to {
        return;
    }
    if from < src.offset && to > src.offset {
        push_layer_span(parts, src, from, src.offset);
        push_layer_span(parts, src, src.offset, to);
    } else {
        push_layer_span(parts, src, from, to);
    }
}

pub(super) fn render_tokens(
    scope: Option<(&[(String, usize)], usize)>,
    src: &Source,
    from: u32,
    to: u32,
) -> String {
    let mut parts = Vec::new();
    let Some((deps, krate)) = scope else {
        push_raw_span(&mut parts, src, from, to);
        return parts.join(" ");
    };
    let mut cursor = from;
    for at in from..to {
        let Some(path) = src
            .dollar(at)
            .filter(|_| at >= cursor && at + 1 < to)
            .and_then(|named| crate_path(deps, krate, named as usize))
        else {
            continue;
        };
        push_raw_span(&mut parts, src, cursor, at);
        parts.push(path);
        cursor = at + 2;
    }
    push_raw_span(&mut parts, src, cursor, to);
    parts.join(" ")
}

fn crate_path(deps: &[(String, usize)], krate: usize, named: usize) -> Option<String> {
    if named == krate {
        return Some("crate".to_owned());
    }
    deps.iter()
        .find(|(_, dep)| *dep == named)
        .map(|(name, _)| format!("::{name}"))
}

pub(super) fn item_text(
    src: &Source,
    item: &crate::outline::item::Item,
    scope: Option<(&[(String, usize)], usize)>,
    keep: &dyn Fn(u32) -> bool,
    enabled: Option<&dyn Fn(&[Attr]) -> bool>,
) -> String {
    let mut parts = Vec::new();
    let mut at = item.lead;
    while at < item.start && src.text(at) == "#" {
        let close = src.pair(at + 1);
        if keep(at + 2) {
            parts.push(render_tokens(scope, src, at, close + 1));
        }
        at = close + 1;
    }
    match item
        .body
        .zip(enabled)
        .and_then(|(open, enabled)| Some((open, configured(src, scope, open, enabled)?)))
    {
        Some((open, members)) => parts.extend([
            render_tokens(scope, src, at, open + 1),
            members,
            render_tokens(scope, src, src.pair(open), item.end),
        ]),
        None => parts.push(render_tokens(scope, src, at, item.end)),
    }
    parts.join(" ")
}

fn configured(
    src: &Source,
    scope: Option<(&[(String, usize)], usize)>,
    open: u32,
    enabled: &dyn Fn(&[Attr]) -> bool,
) -> Option<String> {
    let (mut kept, mut dropped) = (Vec::new(), false);
    for (from, to) in split_commas(src, open + 1, src.pair(open))
        .into_iter()
        .filter(|(from, to)| from < to)
    {
        let mut attrs = Vec::new();
        attr::attrs(src, from, false, &mut attrs).ok()?;
        if enabled(&attrs) {
            kept.push(if scope.is_some() {
                render_tokens(scope, src, from, to)
            } else {
                src.spelled((from, to))
            });
        } else {
            dropped = true
        }
    }
    dropped.then(|| kept.join(" , "))
}
