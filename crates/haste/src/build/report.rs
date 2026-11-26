use std::time::{Duration, Instant};

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

#[cfg(feature = "diff-harness")]
use crate::cli::print_failures;
use crate::cli::{Failures, TestItem, TestRun, error_kind, print_test_report};
use crate::names::ItemRef;
use crate::outline;
use crate::sema;

use super::batch::Batch;
#[cfg(feature = "diff-harness")]
use super::batch::BatchKind;
use super::test_args::parse_test_args;

#[cfg(feature = "diff-harness")]
include!("../../../../tests/case_inputs.rs");

fn format_diagnostic(diag: &str) -> String {
    if crate::diag::is_rendered(diag) {
        diag.to_owned()
    } else {
        format!("error: {diag}")
    }
}

struct ParsedSpan<'a> {
    file: &'a str,
    line: usize,
    col_start: usize,
    col_end: usize,
    is_primary: bool,
    source_line: Option<&'a str>,
    label: Option<&'a str>,
}

struct ParsedDiag<'a> {
    code: Option<&'a str>,
    message: &'a str,
    spans: Vec<ParsedSpan<'a>>,
    children: Vec<(&'static str, &'a str)>,
}

fn parse_loc(loc: &str) -> Option<(&str, usize, usize)> {
    let (rest, col_s) = loc.rsplit_once(':')?;
    let (file, line_s) = rest.rsplit_once(':')?;
    Some((file, line_s.parse().ok()?, col_s.parse().ok()?))
}

fn parse_underline(body: &str) -> Option<(usize, usize, bool, Option<&str>)> {
    let indent = body.bytes().take_while(|&b| b == b' ').count();
    let after = &body[indent..];
    let mark = match after.as_bytes().first().copied() {
        Some(m @ (b'^' | b'-')) => m,
        _ => return None,
    };
    let width = after.bytes().take_while(|&b| b == mark).count();
    let tail = &after[width..];
    if !tail.is_empty() && !tail.starts_with(' ') {
        return None;
    }
    let label = tail.strip_prefix(' ').filter(|s| !s.is_empty());
    let col_start = indent + 1;
    Some((col_start, col_start + width, mark == b'^', label))
}

struct RenderedState<'a> {
    spans: Vec<ParsedSpan<'a>>,
    children: Vec<(&'static str, &'a str)>,
    cur_file: Option<&'a str>,
    primary_loc: Option<(&'a str, usize, usize)>,
    last_source: Option<(usize, &'a str)>,
}

impl<'a> RenderedState<'a> {
    fn feed(&mut self, line: &'a str) {
        let trimmed = line.trim_start();
        if let Some(loc) = trimmed.strip_prefix("--> ") {
            if let Some((file, num, col)) = parse_loc(loc) {
                self.cur_file = Some(file);
                self.primary_loc = Some((file, num, col));
            }
        } else if let Some(loc) = trimmed.strip_prefix("::: ") {
            if let Some((file, _, _)) = parse_loc(loc) {
                self.cur_file = Some(file);
            }
        } else if let Some((prefix, body)) = line.split_once(" | ") {
            self.feed_gutter(prefix.trim(), body);
        } else if let Some(note) = trimmed.strip_prefix("= note: ") {
            self.children.push(("note", note));
        } else if let Some(help) = trimmed.strip_prefix("= help: ") {
            self.children.push(("help", help));
        }
    }

    fn feed_gutter(&mut self, ptrim: &str, body: &'a str) {
        if let Ok(line_num) = ptrim.parse::<usize>() {
            self.last_source = Some((line_num, body));
            return;
        }
        if ptrim.is_empty()
            && let (Some(file), Some((line_num, src)), Some((cs, ce, is_primary, label))) =
                (self.cur_file, self.last_source, parse_underline(body))
        {
            self.spans.push(ParsedSpan {
                file,
                line: line_num,
                col_start: cs,
                col_end: ce,
                is_primary,
                source_line: Some(src),
                label,
            });
        }
    }
}

fn parse_rendered_diagnostic(diag: &str) -> ParsedDiag<'_> {
    let mut lines = diag.lines();
    let first = lines.next().unwrap_or(diag);
    let (code, message) = first
        .strip_prefix("error[")
        .and_then(|rest| rest.split_once("]: "))
        .map_or((None, first), |(c, m)| (Some(c), m));
    let mut state = RenderedState {
        spans: Vec::new(),
        children: Vec::new(),
        cur_file: None,
        primary_loc: None,
        last_source: None,
    };
    for line in lines {
        state.feed(line);
    }
    if state.spans.is_empty()
        && let Some((file, line, col)) = state.primary_loc
    {
        state.spans.push(ParsedSpan {
            file,
            line,
            col_start: col,
            col_end: col + 1,
            is_primary: true,
            source_line: None,
            label: None,
        });
    }
    state.spans.sort_by_key(|s| !s.is_primary);
    ParsedDiag {
        code,
        message,
        spans: state.spans,
        children: state.children,
    }
}

fn parse_diagnostic(diag: &str) -> ParsedDiag<'_> {
    if crate::diag::is_rendered(diag) {
        return parse_rendered_diagnostic(diag);
    }
    if let Some((head, rest)) = diag.split_once(' ') {
        let clean = head.strip_suffix(':').unwrap_or(head);
        let parsed = parse_loc(clean)
            .map(|(f, l, c)| (f, l, c, c + 1))
            .or_else(|| {
                clean.rsplit_once(':').and_then(|(f, ls)| {
                    let l = ls.parse::<usize>().ok().filter(|&n| n > 0)?;
                    (!f.is_empty()).then_some((f, l, 0, 0))
                })
            });
        if let Some((file, line, col_start, col_end)) = parsed {
            return ParsedDiag {
                code: None,
                message: rest,
                spans: vec![ParsedSpan {
                    file,
                    line,
                    col_start,
                    col_end,
                    is_primary: true,
                    source_line: None,
                    label: None,
                }],
                children: Vec::new(),
            };
        }
    }
    ParsedDiag {
        code: None,
        message: diag,
        spans: Vec::new(),
        children: Vec::new(),
    }
}

fn format_short_diagnostic(diag: &str) -> String {
    let parsed = parse_diagnostic(diag);
    let head = match parsed.code {
        Some(code) => format!("error[{code}]: {}", parsed.message),
        None => format!("error: {}", parsed.message),
    };
    let Some(span) = parsed
        .spans
        .iter()
        .find(|s| s.is_primary)
        .or_else(|| parsed.spans.first())
    else {
        return head;
    };
    if span.col_start > 0 {
        format!("{}:{}:{}: {head}", span.file, span.line, span.col_start)
    } else {
        format!("{}:{}: {head}", span.file, span.line)
    }
}

pub(super) fn push_json_str(out: &mut String, s: &str) {
    use std::fmt::Write as _;
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => drop(write!(out, "\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_json_opt_str(out: &mut String, s: Option<&str>) {
    match s {
        Some(v) => push_json_str(out, v),
        None => out.push_str("null"),
    }
}

fn span_byte_range(text: &str, line: usize, col_start: usize, col_end: usize) -> (usize, usize) {
    if line == 0 {
        return (0, 0);
    }
    let mut line_start = 0usize;
    let mut line_text = None;
    for (idx, slice) in text.split_inclusive('\n').enumerate() {
        if idx + 1 == line {
            line_text = Some(slice.trim_end_matches(['\r', '\n']));
            break;
        }
        line_start += slice.len();
    }
    let Some(line_text) = line_text else {
        return (0, 0);
    };
    let col_to_byte = |target_col: usize| -> usize {
        if target_col <= 1 {
            return line_start;
        }
        let mut col = 1usize;
        for (byte_off, ch) in line_text.char_indices() {
            if col >= target_col {
                return line_start + byte_off;
            }
            col += if ch == '\t' { 4 } else { 1 };
        }
        line_start + line_text.len()
    };
    let sb = col_to_byte(col_start);
    let eb = col_to_byte(col_end).max(sb);
    (sb, eb)
}

fn file_span_byte_range(
    file: &str,
    line: usize,
    col_start: usize,
    col_end: usize,
) -> (usize, usize) {
    std::fs::read_to_string(file).map_or((0, 0), |text| {
        span_byte_range(&text, line, col_start, col_end)
    })
}

fn push_json_span(out: &mut String, span: &ParsedSpan<'_>) {
    use std::fmt::Write as _;
    let cs = span.col_start.max(1);
    let ce = span.col_end.max(cs);
    let (byte_start, byte_end) = file_span_byte_range(span.file, span.line, cs, ce);
    out.push_str("{\"file_name\":");
    push_json_str(out, span.file);
    let _ = write!(
        out,
        ",\"byte_start\":{byte_start},\"byte_end\":{byte_end},\"line_start\":{},\"line_end\":{},\"column_start\":{cs},\"column_end\":{ce},\"is_primary\":{},\"text\":[",
        span.line, span.line, span.is_primary
    );
    if let Some(src) = span.source_line {
        out.push_str("{\"text\":");
        push_json_str(out, src);
        let _ = write!(out, ",\"highlight_start\":{cs},\"highlight_end\":{ce}}}");
    }
    out.push_str("],\"label\":");
    push_json_opt_str(out, span.label);
    out.push_str(
        ",\"suggested_replacement\":null,\"suggestion_applicability\":null,\"expansion\":null}",
    );
}

fn push_json_child(out: &mut String, level: &str, message: &str) {
    out.push_str("{\"$message_type\":\"diagnostic\",\"level\":");
    push_json_str(out, level);
    out.push_str(",\"code\":null,\"message\":");
    push_json_str(out, message);
    out.push_str(",\"spans\":[],\"children\":[],\"rendered\":null}");
}

fn format_json_diagnostic(outline: Option<&outline::Outline>, diag: &str) -> String {
    let parsed = parse_diagnostic(diag);
    let rendered = format!("{}\n", format_diagnostic(diag));
    let mut out = String::with_capacity(diag.len() + 512);
    out.push_str("{\"reason\":\"compiler-message\"");
    super::json_diag::push_json_package_header(
        &mut out,
        outline,
        parsed.spans.first().map(|s| s.file),
    );
    out.push_str(",\"message\":{\"$message_type\":\"diagnostic\",\"level\":\"error\",\"code\":");
    match parsed.code {
        Some(code) => {
            out.push_str("{\"code\":");
            push_json_str(&mut out, code);
            out.push_str(",\"explanation\":null}");
        }
        None => out.push_str("null"),
    }
    out.push_str(",\"message\":");
    push_json_str(&mut out, parsed.message);
    out.push_str(",\"spans\":[");
    for (idx, span) in parsed.spans.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        push_json_span(&mut out, span);
    }
    out.push_str("],\"children\":[");
    for (idx, &(level, msg)) in parsed.children.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        push_json_child(&mut out, level, msg);
    }
    out.push_str("],\"rendered\":");
    push_json_str(&mut out, &rendered);
    out.push_str("}}");
    out
}

fn print_human_diagnostic(diag: &str) {
    let rendered = format_diagnostic(diag);
    if crate::diag::is_rendered(diag) {
        eprintln!("{rendered}\n");
    } else {
        eprintln!("{rendered}");
    }
}

pub(super) fn format_diagnostics(diagnostics: &[String]) -> Option<String> {
    (!diagnostics.is_empty()).then(|| {
        diagnostics
            .iter()
            .map(|diag| format_diagnostic(diag))
            .collect::<Vec<_>>()
            .join("\n\n")
    })
}

pub(super) fn split_error_diagnostics(text: &str) -> Vec<String> {
    if !crate::diag::is_rendered(text) {
        return vec![text.to_owned()];
    }
    let mut out = Vec::new();
    let (mut start, mut offset) = (0usize, 0usize);
    for line in text.split_inclusive('\n') {
        if offset > start && crate::diag::is_rendered(line) {
            out.push(text[start..offset].trim_end_matches('\n').to_owned());
            start = offset;
        }
        offset += line.len();
    }
    let tail = text[start..].trim_end_matches('\n');
    if !tail.is_empty() {
        out.push(tail.to_owned());
    }
    out
}

pub(super) fn print_formatted_diagnostics(
    outline: Option<&outline::Outline>,
    diagnostics: &[String],
) {
    let fmt = crate::selection().message_format;
    for diag in diagnostics {
        match fmt {
            crate::MessageFormat::Human => print_human_diagnostic(diag),
            crate::MessageFormat::Short => eprintln!("{}", format_short_diagnostic(diag)),
            crate::MessageFormat::Json => println!("{}", format_json_diagnostic(outline, diag)),
            crate::MessageFormat::JsonRenderDiagnostics => {
                println!("{}", format_json_diagnostic(outline, diag));
                print_human_diagnostic(diag);
            }
        }
    }
}

impl Batch {
    pub(super) fn format_entry(
        outline: Option<&outline::Outline>,
        item: ItemRef,
        name: Option<&str>,
        error: &str,
    ) -> String {
        if crate::diag::is_rendered(error) {
            return error.to_owned();
        }
        let loc = outline.map(|o| item_loc(o, item)).unwrap_or_default();
        match name {
            Some(name) if !error.starts_with(name) => format!("{loc}{name}: {error}"),
            _ => format!("{loc}{error}"),
        }
    }

    fn entry_error(&self, at: usize) -> Option<&str> {
        self.reached.list[at]
            .1
            .as_ref()
            .err()
            .or_else(|| self.broken.get(&at))
            .map(String::as_str)
    }

    #[cfg(feature = "diff-harness")]
    fn sort(&self, outline: Option<&outline::Outline>) -> (Failures, Vec<(usize, &str)>) {
        let mut failures = Failures::new();
        let mut runnable = Vec::new();
        for (&root, name) in self.roots.iter().zip(&self.labels) {
            let closure = self.reached.closure(&[root]);
            let found = closure
                .iter()
                .rev()
                .find_map(|&at| self.entry_error(at).map(|err| (at, err)));
            match found {
                Some((at, error)) => {
                    let headline = error.lines().next().unwrap_or(error);
                    let loc = outline
                        .map(|o| item_loc(o, self.reached.list[at].0.0))
                        .unwrap_or_default();
                    let msg = if headline.starts_with(name.as_str()) {
                        format!("{loc}{headline}")
                    } else {
                        format!("{loc}{name}: {headline}")
                    };
                    failures.entry(error_kind(error)).or_default().push(msg);
                }
                None => runnable.push((root, name.as_str())),
            }
        }
        for &(item, ref error) in &self.poly_failures {
            let headline = error.lines().next().unwrap_or(error);
            let loc = outline.map(|o| item_loc(o, item)).unwrap_or_default();
            failures
                .entry(error_kind(error))
                .or_default()
                .push(format!("{loc}{headline}"));
        }
        (failures, runnable)
    }

    pub(super) fn collect_diagnostics(&self, outline: &outline::Outline) -> (Vec<String>, usize) {
        let mut out = Vec::new();
        let mut seen = HashSet::default();
        let mut failed_items = 0usize;
        let mut record = |item: ItemRef, error: &str| {
            failed_items += 1;
            let msg = Self::format_entry(Some(outline), item, None, error);
            if seen.insert(msg.clone()) {
                out.push(msg);
            }
        };
        for at in self.reached.closure(&self.roots) {
            if let Some(error) = self.entry_error(at) {
                record(self.reached.list[at].0.0, error);
            }
        }
        for &(item, ref error) in &self.poly_failures {
            record(item, error);
        }
        for msg in &self.member_errors {
            failed_items += 1;
            if seen.insert(msg.clone()) {
                out.push(msg.clone());
            }
        }
        (out, failed_items)
    }

    pub(super) fn print_diagnostics(&self, outline: &outline::Outline) -> usize {
        let (diagnostics, failed_items) = self.collect_diagnostics(outline);
        print_formatted_diagnostics(Some(outline), &diagnostics);
        failed_items
    }

    pub(super) fn formatted_diagnostics(&self, outline: &outline::Outline) -> Option<String> {
        let (diagnostics, _) = self.collect_diagnostics(outline);
        format_diagnostics(&diagnostics)
    }

    pub(super) fn failures(&self, cx: &sema::Context<'_>) -> Failures {
        let mut failures = Failures::new();
        for at in self.reached.closure(&self.roots) {
            if let Some(error) = self.entry_error(at) {
                let (item, substs) = self.reached.list[at].0;
                let (src, entry) = cx.item(item);
                let substs: Vec<String> = cx
                    .types
                    .substs(substs)
                    .iter()
                    .map(|&ty| cx.type_text(ty, &|ty| ty))
                    .collect();
                let headline = error.lines().next().unwrap_or(error);
                failures.entry(error_kind(error)).or_default().push(format!(
                    "{}:{} {headline} [{}]",
                    src.path.display(),
                    src.line(entry.start),
                    substs.join(", ")
                ));
            }
        }
        failures
    }

    pub(super) fn check(&self, outline: &outline::Outline, built: Duration) -> i32 {
        let failed = self.print_diagnostics(outline);
        let sel = crate::selection();
        if !sel.quiet && sel.message_format != crate::MessageFormat::Json {
            eprintln!(
                "{} functions reached, {failed} failed, built in {:.1} ms",
                self.reached.list.len(),
                built.as_secs_f64() * 1e3
            );
        }
        i32::from(failed > 0)
    }

    pub(super) fn report(
        &self,
        outline: &outline::Outline,
        started: Instant,
        _built: Duration,
        filters: &[String],
    ) -> Result<i32> {
        #[cfg(feature = "diff-harness")]
        if self.kind != BatchKind::Tests {
            return self.report_cases(outline, started, _built);
        }
        self.report_tests(outline, started, filters)
    }

    #[cfg(feature = "diff-harness")]
    fn report_cases(
        &self,
        outline: &outline::Outline,
        started: Instant,
        built: Duration,
    ) -> Result<i32> {
        let (mut failures, runnable) = self.sort(Some(outline));
        let statuses = self.image.run_cases(&runnable, &CASE_INPUTS)?;
        for ((_, name), status) in runnable
            .iter()
            .zip(statuses)
            .filter(|(_, status)| *status != 0)
        {
            failures
                .entry(format!("crashed with status {status}"))
                .or_default()
                .push((*name).to_owned());
        }
        print_failures(&failures);
        let ms = |at: Duration| at.as_secs_f64() * 1e3;
        eprintln!(
            "cases {} found, {} ran, {} functions reached, built in {:.1} ms, total {:.1} ms",
            self.labels.len(),
            runnable.len(),
            self.reached.list.len(),
            ms(built),
            ms(started.elapsed())
        );
        Ok(if failures.is_empty() { 0 } else { 101 })
    }

    fn report_tests(
        &self,
        outline: &outline::Outline,
        started: Instant,
        filters: &[String],
    ) -> Result<i32> {
        if !self.member_errors.is_empty() {
            print_formatted_diagnostics(Some(outline), &self.member_errors);
            crate::diag::print_build_finished(false);
            return Ok(101);
        }
        let sel = crate::selection();
        let args = parse_test_args(filters);
        let ignored_only = sel.ignored_only || args.ignored_only;
        let include_ignored = sel.include_ignored || args.include_ignored;
        if args.list {
            for ((_, name), (meta, _)) in self.roots.iter().zip(&self.labels).zip(&self.tests) {
                if !args.matches_name(name) || (ignored_only && !meta.ignored) {
                    continue;
                }
                println!("{name}: test");
            }
            return Ok(0);
        }
        let mut errs: Vec<String> = Vec::new();
        let mut raw: Vec<(
            &str,
            &TestItem,
            Option<Result<(usize, Option<usize>), usize>>,
        )> = Vec::new();
        for ((&root, name), (meta, start)) in self.roots.iter().zip(&self.labels).zip(&self.tests) {
            if !args.matches_name(name) {
                continue;
            }
            if ignored_only && !meta.ignored {
                continue;
            }
            let skip = meta.ignored && !ignored_only && !include_ignored;
            let check_roots: Vec<usize> =
                (*start).into_iter().chain(std::iter::once(root)).collect();
            let outcome = if skip {
                None
            } else if let Some((at, err)) = self
                .reached
                .closure(&check_roots)
                .iter()
                .rev()
                .find_map(|&at| self.entry_error(at).map(|e| (at, e)))
            {
                let idx = errs.len();
                errs.push(Self::format_entry(
                    Some(outline),
                    self.reached.list[at].0.0,
                    None,
                    err,
                ));
                Some(Err(idx))
            } else {
                Some(Ok((root, *start)))
            };
            raw.push((name.as_str(), meta, outcome));
        }
        let mut entries: Vec<(&str, &TestItem, Option<Result<TestRun, &str>>)> =
            Vec::with_capacity(raw.len());
        for (name, meta, outcome) in raw {
            let env = outline.crates[meta.item.krate as usize].env.vars.as_slice();
            let manifest_dir = env
                .iter()
                .find(|(k, _)| k == "CARGO_MANIFEST_DIR")
                .map(|(_, v)| std::path::Path::new(v.as_str()));
            let resolved = match outcome {
                None => None,
                Some(Err(idx)) => Some(Err(errs[idx].as_str())),
                Some(Ok((root, start))) => {
                    let nocapture = args.nocapture && meta.expected.is_none();
                    let run = self.image.run_test_captured(
                        root,
                        start,
                        name,
                        env,
                        manifest_dir,
                        nocapture,
                    )?;
                    Some(Ok(run))
                }
            };
            entries.push((name, meta, resolved));
        }
        let total_tests = if args.filters.is_empty() && args.skips.is_empty() {
            self.tests.len()
        } else {
            crate::cli::outline_test_count(outline).max(self.tests.len())
        };
        let filtered_out = total_tests.saturating_sub(entries.len());
        Ok(print_test_report(
            &entries,
            started.elapsed(),
            args.show_output,
            filtered_out,
        ))
    }
}

fn item_loc(outline: &outline::Outline, item: ItemRef) -> String {
    let outer = item.body_owner();
    let file = &outline.crates[outer.krate as usize].files[outer.file as usize];
    let Some(entry) = file.items.items.get(outer.item as usize) else {
        return format!("{} ", file.source.path.display());
    };
    let line = file.source.line(entry.start);
    format!("{}:{line} ", file.source.path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_byte_range_matches_multiline_and_tabs() {
        let text = "fn main() {\n    let x: i32 = \"hi\";\n\tbad;\n}\n";
        let hi = text.find("\"hi\"").expect("literal");
        assert_eq!(span_byte_range(text, 2, 18, 22), (hi, hi + 4));
        let bad = text.find("bad").expect("bad");
        assert_eq!(span_byte_range(text, 3, 5, 8), (bad, bad + 3));
        assert_eq!(span_byte_range(text, 0, 1, 2), (0, 0));
    }

    #[test]
    fn json_diagnostic_matches_cargo_metadata_for_integration_test_and_lib() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let api_test = manifest_dir.join("tests/compiler_api.rs");
        let diag = format!("error[type]: mismatch\n --> {}:1:1", api_test.display());
        let json = format_json_diagnostic(None, &diag);
        let expected_id = format!(
            "\"package_id\":\"path+file://{}#0.0.0\"",
            manifest_dir.display()
        );
        assert!(
            json.contains(&expected_id),
            "unexpected package_id in:\n{json}"
        );
        assert!(
            json.contains("\"target\":{\"kind\":[\"test\"],\"crate_types\":[\"bin\"],\"name\":\"compiler_api\","),
            "unexpected target in:\n{json}"
        );
        assert!(
            json.contains("\"edition\":\"2024\""),
            "expected 2024 edition in:\n{json}"
        );
        assert!(
            json.contains("\"doc\":false,\"doctest\":false,\"test\":true}"),
            "unexpected target flags in:\n{json}"
        );

        let codegen_dir = manifest_dir.parent().expect("crates dir").join("codegen");
        let codegen_lib = codegen_dir.join("src/lib.rs");
        let cg_diag = format!("error[type]: mismatch\n --> {}:1:1", codegen_lib.display());
        let cg_json = format_json_diagnostic(None, &cg_diag);
        let cg_id = format!(
            "\"package_id\":\"path+file://{}#haste-codegen@0.0.0\"",
            codegen_dir.display()
        );
        assert!(
            cg_json.contains(&cg_id),
            "unexpected codegen id in:\n{cg_json}"
        );
        assert!(
            cg_json.contains("\"target\":{\"kind\":[\"lib\"],\"crate_types\":[\"lib\"],\"name\":\"haste_codegen\","),
            "unexpected codegen target in:\n{cg_json}"
        );
        assert!(
            cg_json.contains("\"edition\":\"2024\""),
            "expected 2024 edition in:\n{cg_json}"
        );

        let plain = format_json_diagnostic(None, "package `missing` is not in Cargo.lock");
        assert!(plain.contains("\"package_id\":\"path+file://"));
        assert!(plain.contains("\"spans\":[]"));
    }
}
