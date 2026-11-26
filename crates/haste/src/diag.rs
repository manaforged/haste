//! Structured compiler diagnostics with source spans, secondary labels, and snippet rendering.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::body::{Body, ExprId, PatId};
use crate::outline::source::Source;

/// Classification of a compiler diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DiagnosticKind {
    /// Borrow checker, move, mutability, or lifetime violation.
    Borrow,
    /// Unsafe operation outside an `unsafe` block or function.
    Unsafe,
    /// Type mismatch or invalid type operation.
    Type,
    /// Unresolved path, field, method, or binding.
    Name,
    /// Unsatisfied trait or auto-trait bound.
    Trait,
    /// Non-exhaustive pattern match.
    Exhaustive,
    /// Visibility or privacy violation.
    Privacy,
    /// Syntax or body parsing failure.
    Syntax,
    /// IR lowering or codegen failure.
    Lower,
}

impl DiagnosticKind {
    /// Short machine-readable tag rendered in `error[<tag>]: ...`.
    pub fn code(self) -> &'static str {
        match self {
            Self::Borrow => "borrow",
            Self::Unsafe => "unsafe",
            Self::Type => "type",
            Self::Name => "name",
            Self::Trait => "trait",
            Self::Exhaustive => "exhaustive",
            Self::Privacy => "privacy",
            Self::Syntax => "syntax",
            Self::Lower => "lower",
        }
    }
}

/// A resolved source location covering a half-open column range on a source line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    file: PathBuf,
    line: usize,
    col: usize,
    end_col: usize,
    source_line: String,
    from_macro: bool,
}

impl Span {
    /// File path containing this span.
    pub fn file(&self) -> &Path {
        &self.file
    }

    /// One-based line number of the span start.
    pub fn line(&self) -> usize {
        self.line
    }

    /// One-based character column of the span start.
    pub fn column(&self) -> usize {
        self.col
    }

    /// Full text of the source line containing this span (without trailing newline).
    pub fn source_line(&self) -> &str {
        &self.source_line
    }

    /// Whether this span originated from a macro expansion.
    pub fn from_macro(&self) -> bool {
        self.from_macro
    }

    /// Resolves a byte range `(start_byte..end_byte)` in `text` into a displayable `Span`.
    #[cfg(test)]
    pub(crate) fn from_bytes(
        file: PathBuf,
        text: &str,
        start_byte: u32,
        end_byte: u32,
        from_macro: bool,
    ) -> Self {
        Self::from_bytes_indexed(file, text, None, start_byte, end_byte, from_macro)
    }

    fn from_bytes_indexed(
        file: PathBuf,
        text: &str,
        newlines: Option<&[u32]>,
        start_byte: u32,
        end_byte: u32,
        from_macro: bool,
    ) -> Self {
        let len = text.len();
        let sb = clamp_char_boundary(text, (start_byte as usize).min(len));
        let eb = clamp_char_boundary(text, (end_byte as usize).max(sb).min(len));
        let (line_idx, line_start, line_end) = match newlines {
            Some(nls) => {
                let idx = nls.partition_point(|&nl| (nl as usize) < sb);
                let start = if idx == 0 {
                    0
                } else {
                    nls[idx - 1] as usize + 1
                };
                let end = nls.get(idx).map_or(len, |&nl| nl as usize);
                (idx, start, end)
            }
            None => {
                let start = text[..sb].rfind('\n').map_or(0, |nl| nl + 1);
                let end = text[sb..].find('\n').map_or(len, |rel| sb + rel);
                let idx = text[..start].matches('\n').count();
                (idx, start, end)
            }
        };
        let line_text = text[line_start..line_end].trim_end_matches('\r');
        let line_limit = line_start + line_text.len();
        let clamped_start = sb.min(line_limit);
        let clamped_end = eb.min(line_limit).max(clamped_start);
        let line = line_idx + 1;
        let col = expanded_width(&text[line_start..clamped_start]) + 1;
        let width = expanded_width(&text[clamped_start..clamped_end]).max(1);
        Self {
            file,
            line,
            col,
            end_col: col + width,
            source_line: line_text.replace('\t', "    "),
            from_macro,
        }
    }

    /// Resolves a half-open token range `start_tok..end_tok` on `src`.
    pub(crate) fn from_tokens(src: &Source, start_tok: u32, end_tok: u32) -> Option<Self> {
        let (layer, sb, eb, from_macro) = src.token_byte_span(start_tok, end_tok)?;
        Some(Self::from_bytes_indexed(
            layer.path.clone(),
            &layer.text,
            Some(layer.line_starts()),
            sb,
            eb,
            from_macro,
        ))
    }

    /// Resolves a single token index `tok` on `src`.
    pub(crate) fn from_token(src: &Source, tok: u32) -> Option<Self> {
        Self::from_tokens(src, tok, tok.saturating_add(1))
    }

    /// Resolves the exact span of parsed expression `expr` in `body` on `src`.
    pub(crate) fn from_expr(src: &Source, body: &Body, expr: ExprId) -> Option<Self> {
        let (start, end) = body.expr_span(expr)?;
        Self::from_tokens(src, start, end)
    }

    /// Resolves the exact span of parsed pattern `pat` in `body` on `src`.
    pub(crate) fn from_pat(src: &Source, body: &Body, pat: PatId) -> Option<Self> {
        let (start, end) = body.pat_span(pat)?;
        Self::from_tokens(src, start, end)
    }
}

/// Secondary annotated span attached to a `Diagnostic`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    span: Span,
    message: String,
}

impl Label {
    /// Source span of this secondary annotation.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Explanatory message for this secondary annotation.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// A structured compiler error with optional primary span, secondary labels, note, and help text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    kind: DiagnosticKind,
    message: String,
    primary: Option<Span>,
    primary_label: Option<String>,
    labels: Vec<Label>,
    note: Option<String>,
    help: Option<String>,
}

impl Diagnostic {
    /// Creates a new diagnostic of `kind` with headline `message`.
    pub(crate) fn new(kind: DiagnosticKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            primary: None,
            primary_label: None,
            labels: Vec::new(),
            note: None,
            help: None,
        }
    }

    /// Attaches the primary source span.
    pub(crate) fn with_span(mut self, span: Option<Span>) -> Self {
        self.primary = span;
        self
    }

    /// Attaches inline text printed next to the primary `^^^` underline.
    pub(crate) fn with_primary_label(mut self, label: impl Into<String>) -> Self {
        self.primary_label = Some(label.into());
        self
    }

    /// Attaches a secondary `---` label when `span` is present.
    pub(crate) fn with_label(mut self, span: Option<Span>, message: impl Into<String>) -> Self {
        if let Some(span) = span {
            self.labels.push(Label {
                span,
                message: message.into(),
            });
        }
        self
    }

    /// Attaches a trailing `= note:` line.
    pub(crate) fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Attaches a trailing `= help:` line.
    pub(crate) fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Diagnostic category.
    pub fn kind(&self) -> DiagnosticKind {
        self.kind
    }

    /// Headline error message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Primary source span, if available.
    pub fn span(&self) -> Option<&Span> {
        self.primary.as_ref()
    }

    /// Secondary labels attached to this diagnostic.
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }
}

/// Whether `text` is already formatted as a structured Haste `Diagnostic`.
pub(crate) fn is_rendered(text: &str) -> bool {
    const PREFIXES: [&str; 9] = [
        "error[borrow]: ",
        "error[unsafe]: ",
        "error[type]: ",
        "error[name]: ",
        "error[trait]: ",
        "error[exhaustive]: ",
        "error[privacy]: ",
        "error[syntax]: ",
        "error[lower]: ",
    ];
    PREFIXES.iter().any(|p| text.starts_with(p))
}

/// Formats an `anyhow::Error` as a rendered `Diagnostic` when it carries one anywhere in its chain, else `<name>: <err>`.
pub(crate) fn format_error(name: &str, error: &anyhow::Error) -> String {
    if let Some(diag) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<Diagnostic>())
    {
        if diag.span().is_some() {
            return diag.to_string();
        }
        let mut with_name = diag.clone();
        with_name.message = format!("{name}: {}", diag.message);
        return with_name.to_string();
    }
    format!("{name}: {error:#}")
}

impl std::error::Error for Diagnostic {}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]: {}", self.kind.code(), self.message)?;
        let max_line = self
            .labels
            .iter()
            .map(|l| l.span.line)
            .fold(self.primary.as_ref().map_or(0, |s| s.line), usize::max);
        let gutter = if max_line == 0 { 0 } else { digits(max_line) };
        let pad = " ".repeat(gutter);
        let mut other_files: Vec<&Label> = Vec::new();
        if let Some(primary) = &self.primary {
            write!(
                f,
                "\n{pad}--> {}:{}:{}",
                primary.file.display(),
                primary.line,
                primary.col
            )?;
            let mut same_file: Vec<(&Span, &str, bool)> = Vec::new();
            for label in &self.labels {
                if label.span.file == primary.file {
                    same_file.push((&label.span, &label.message, false));
                } else {
                    other_files.push(label);
                }
            }
            same_file.push((primary, self.primary_label.as_deref().unwrap_or(""), true));
            same_file.sort_by_key(|(s, _, is_primary)| (s.line, s.col, u8::from(*is_primary)));
            render_file_group(f, gutter, &same_file)?;
        } else {
            other_files.extend(self.labels.iter());
        }
        render_other_files(f, gutter, &mut other_files)?;
        let macro_note = self
            .primary
            .as_ref()
            .is_some_and(|p| p.from_macro)
            .then_some("this error originates in a macro expansion");
        write_footer(f, gutter, macro_note, None)?;
        write_footer(f, gutter, self.note.as_deref(), self.help.as_deref())
    }
}

fn render_other_files(
    f: &mut fmt::Formatter<'_>,
    gutter: usize,
    other_files: &mut Vec<&Label>,
) -> fmt::Result {
    if other_files.is_empty() {
        return Ok(());
    }
    other_files.sort_by(|a, b| {
        a.span
            .file
            .cmp(&b.span.file)
            .then_with(|| (a.span.line, a.span.col).cmp(&(b.span.line, b.span.col)))
    });
    let pad = " ".repeat(gutter);
    let mut start = 0;
    while start < other_files.len() {
        let file = &other_files[start].span.file;
        let mut end = start + 1;
        while end < other_files.len() && &other_files[end].span.file == file {
            end += 1;
        }
        let first = &other_files[start].span;
        write!(
            f,
            "\n{pad}::: {}:{}:{}",
            first.file.display(),
            first.line,
            first.col
        )?;
        let group: Vec<(&Span, &str, bool)> = other_files[start..end]
            .iter()
            .map(|l| (&l.span, l.message.as_str(), false))
            .collect();
        render_file_group(f, gutter, &group)?;
        start = end;
    }
    Ok(())
}

fn render_file_group(
    f: &mut fmt::Formatter<'_>,
    gutter: usize,
    items: &[(&Span, &str, bool)],
) -> fmt::Result {
    let pad = " ".repeat(gutter);
    write!(f, "\n{pad} |")?;
    let mut prev_line: Option<usize> = None;
    for &(span, label, is_primary) in items {
        if let Some(prev) = prev_line {
            if span.line > prev + 1 {
                write!(f, "\n...")?;
            }
            if span.line != prev {
                write!(f, "\n{:>gutter$} | {}", span.line, span.source_line)?;
            }
        } else {
            write!(f, "\n{:>gutter$} | {}", span.line, span.source_line)?;
        }
        let indent = " ".repeat(span.col.saturating_sub(1));
        let mark = if is_primary { '^' } else { '-' };
        let width = span.end_col.saturating_sub(span.col).max(1);
        let underline: String = std::iter::repeat_n(mark, width).collect();
        let clean_label = label.lines().next().unwrap_or("");
        if clean_label.is_empty() {
            write!(f, "\n{pad} | {indent}{underline}")?;
        } else {
            write!(f, "\n{pad} | {indent}{underline} {clean_label}")?;
        }
        prev_line = Some(span.line);
    }
    write!(f, "\n{pad} |")
}

fn expanded_width(text: &str) -> usize {
    text.chars().map(|c| if c == '\t' { 4 } else { 1 }).sum()
}

fn write_footer(
    f: &mut fmt::Formatter<'_>,
    gutter: usize,
    note: Option<&str>,
    help: Option<&str>,
) -> fmt::Result {
    if let Some(note) = note {
        write_footer_entry(f, gutter, "note", note)?;
    }
    if let Some(help) = help {
        write_footer_entry(f, gutter, "help", help)?;
    }
    Ok(())
}

fn write_footer_entry(
    f: &mut fmt::Formatter<'_>,
    gutter: usize,
    tag: &str,
    text: &str,
) -> fmt::Result {
    let pad = " ".repeat(gutter);
    let cont = " ".repeat(tag.len() + 5);
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            write!(f, "\n{pad} = {tag}: {line}")?;
        } else {
            write!(f, "\n{pad}{cont}{line}")?;
        }
    }
    Ok(())
}

fn digits(mut n: usize) -> usize {
    let mut count = 1;
    while n >= 10 {
        n /= 10;
        count += 1;
    }
    count
}

fn clamp_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::body::{self, Config, Expr};
    use crate::outline::source::{Origins, Source, Tokens};
    use crate::outline::tree::File;

    use super::*;

    #[test]
    fn renders_single_span_diagnostic_snapshot() {
        let text = "fn main() {\n    let x: i32 = \"hi\";\n}\n";
        let sb = text.find("\"hi\"").expect("literal present") as u32;
        let span = Span::from_bytes(PathBuf::from("src/main.rs"), text, sb, sb + 4, false);
        let diag = Diagnostic::new(DiagnosticKind::Type, "mismatched types")
            .with_span(Some(span))
            .with_primary_label("expected `i32`, found `&str`")
            .with_help("convert the string or change the binding type");
        let expected = "\
error[type]: mismatched types
 --> src/main.rs:2:18
  |
2 |     let x: i32 = \"hi\";
  |                  ^^^^ expected `i32`, found `&str`
  |
  = help: convert the string or change the binding type";
        assert_eq!(diag.to_string(), expected);
        assert!(is_rendered(&diag.to_string()));
        assert!(!is_rendered("error[E0308]: foreign rustc error"));
    }

    #[test]
    fn renders_multi_span_borrow_diagnostic_snapshot() {
        let text = "fn main() {\n    let mut v = vec![1];\n    let r = &v;\n    let _a = 1;\n    v.push(2);\n    println!(\"{r:?}\");\n}\n";
        let imm_start = text.find("&v").expect("&v") as u32;
        let mut_start = text.find("v.push(2)").expect("push") as u32;
        let imm = Span::from_bytes(
            PathBuf::from("src/lib.rs"),
            text,
            imm_start,
            imm_start + 2,
            false,
        );
        let conflict = Span::from_bytes(
            PathBuf::from("src/lib.rs"),
            text,
            mut_start,
            mut_start + 9,
            false,
        );
        let diag = Diagnostic::new(
            DiagnosticKind::Borrow,
            "cannot borrow `v` as mutable because it is also borrowed as immutable",
        )
        .with_span(Some(conflict))
        .with_primary_label("mutable borrow occurs here")
        .with_label(Some(imm), "immutable borrow occurs here");
        let expected = "\
error[borrow]: cannot borrow `v` as mutable because it is also borrowed as immutable
 --> src/lib.rs:5:5
  |
3 |     let r = &v;
  |             -- immutable borrow occurs here
...
5 |     v.push(2);
  |     ^^^^^^^^^ mutable borrow occurs here
  |";
        assert_eq!(diag.to_string(), expected);
    }

    #[test]
    fn handles_crlf_and_tab_indented_lines_without_panicking() {
        let text = "fn f() {\r\n\tlet x = bad;\r\n}\r\n";
        let cr = text.find('\r').expect("cr") as u32;
        let cr_span = Span::from_bytes(PathBuf::from("win.rs"), text, cr, cr + 1, false);
        assert_eq!(cr_span.line(), 1);
        let bad = text.find("bad").expect("bad") as u32;
        let tab_span = Span::from_bytes(PathBuf::from("win.rs"), text, bad, bad + 3, false);
        let rendered = Diagnostic::new(DiagnosticKind::Name, "not found")
            .with_span(Some(tab_span))
            .to_string();
        assert!(rendered.contains("2 |     let x = bad;\n  |             ^^^"));
    }

    #[test]
    fn resolves_body_expr_and_pat_spans_exactly() {
        let code = "{ let (a, b) = (10, 20); a + b }";
        let mut src = Source::parse(PathBuf::from("sample.rs"), code.to_owned()).expect("lex");
        src.origins = Origins::Read;
        let body = body::parse(&src, 0, Config::default()).expect("parse");
        assert_eq!(body.exprs.len(), body.expr_starts.len());
        assert_eq!(body.exprs.len(), body.expr_ends.len());
        assert_eq!(body.pats.len(), body.pat_spans.len());

        let add_id = body
            .exprs
            .iter()
            .enumerate()
            .find_map(|(i, e)| matches!(e, Expr::Binary(..)).then_some(ExprId(i as u32)))
            .expect("binary add expr");
        let add_span = Span::from_expr(&src, &body, add_id).expect("expr span");
        assert_eq!(add_span.line(), 1);
        assert_eq!(add_span.column(), code.find("a + b").expect("pos") + 1);
        assert_eq!(add_span.end_col - add_span.col, "a + b".len());

        let tuple_pat_id = body
            .pats
            .iter()
            .enumerate()
            .find_map(|(i, p)| matches!(p, body::Pat::Tuple(..)).then_some(PatId(i as u32)))
            .expect("tuple pat");
        let pat_span = Span::from_pat(&src, &body, tuple_pat_id).expect("pat span");
        assert_eq!(pat_span.column(), code.find("(a, b)").expect("pos") + 1);
        assert_eq!(pat_span.end_col - pat_span.col, "(a, b)".len());
        assert!(Span::from_token(&src, src.len() + 5).is_none());
    }

    #[test]
    fn renders_macro_origin_diagnostic_snapshot() {
        let base_text = "fn demo() {\n    bad_macro!(alpha);\n}\n";
        let mut base_src =
            Source::parse(PathBuf::from("src/macro.rs"), base_text.to_owned()).expect("lex");
        base_src.origins = Origins::Read;
        let alpha_tok = (0..base_src.len())
            .find(|&t| base_src.text(t) == "alpha")
            .expect("alpha token");
        let base_file = Arc::new(File::new(
            base_src,
            crate::outline::item::Items::default(),
            false,
        ));
        let mut tokens = Tokens::default();
        tokens.copy_range(&base_file.source, alpha_tok, alpha_tok + 1);
        let layered = tokens.finish_over(Arc::clone(&base_file)).expect("layer");
        let copied_tok = base_file.source.len();
        let span = Span::from_token(&layered, copied_tok).expect("macro span");
        assert!(span.from_macro());
        let diag = Diagnostic::new(
            DiagnosticKind::Name,
            "cannot find value `alpha` in this scope",
        )
        .with_span(Some(span))
        .with_primary_label("not found in this scope");
        let expected = "\
error[name]: cannot find value `alpha` in this scope
 --> src/macro.rs:2:16
  |
2 |     bad_macro!(alpha);
  |                ^^^^^ not found in this scope
  |
  = note: this error originates in a macro expansion";
        assert_eq!(diag.to_string(), expected);
    }
}
