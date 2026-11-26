use anyhow::{Result, anyhow, bail};
use rustc_lexer::TokenKind;

use crate::outline::source::{Source, Tokens};
use crate::outline::tree::Truth;
use crate::sema::Context;
use crate::sema::builtin::{self, Builtin};
use crate::sema::format;

use super::writer::{Run, Writer, call_path};

#[derive(Clone, Copy)]
pub(super) struct Site {
    pub(super) builtin: Builtin,
    pub(super) krate: u32,
    pub(super) start: u32,
    pub(super) open: u32,
}

impl Writer {
    pub(super) fn builtin(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        site: Site,
        out: &mut Tokens,
    ) -> Result<bool> {
        let (src, open) = (run.src, site.open);
        let input = (open + 1, src.pair(open));
        match site.builtin {
            Builtin::Concat => return self.concat(cx, run, open, out),
            Builtin::Env | Builtin::OptionEnv => return self.env(cx, run, site, out),
            Builtin::Include | Builtin::IncludeStr | Builtin::IncludeBytes => {
                return self.include(cx, run, site, out);
            }
            Builtin::FormatArgs | Builtin::FormatArgsNl => {
                return self.format_args(cx, run, site, out);
            }
            Builtin::PatternType => return Ok(false),
            Builtin::Line | Builtin::Column | Builtin::File | Builtin::Stringify => {
                position(run, site, input, out)
            }
            Builtin::Cfg => self.cfg(cx, src, input, out)?,
            Builtin::ModulePath => out.lexed(&format!("{:?}", cx.outline.module_path(self.home))),
            Builtin::CfgSelect => return self.cfg_select(cx, run, open, out),
            Builtin::Assert => builtin::assert(src, &src.args(open), site.krate, out),
            Builtin::Asm => self.asm(cx, run, open, out)?,
            Builtin::Panic | Builtin::Unreachable => self.panic(cx, run, site, out),
        }
        Ok(true)
    }

    fn cfg(
        &self,
        cx: &Context<'_>,
        src: &Source,
        (from, to): (u32, u32),
        out: &mut Tokens,
    ) -> Result<()> {
        let predicate = (from..to)
            .map(|at| src.text(at))
            .collect::<Vec<_>>()
            .join(" ");
        let holds = cx.outline.crates[self.home.krate as usize]
            .env
            .eval(&crate::graph::cfg::parse(&predicate)?)
            == Truth::Yes;
        out.lexed(if holds { "true" } else { "false" });
        Ok(())
    }

    fn cfg_select(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        open: u32,
        out: &mut Tokens,
    ) -> Result<bool> {
        let env = &cx.outline.crates[self.home.krate as usize].env;
        let Some(arm) = crate::outline::mbe::select_arm(run.src, open, env)? else {
            bail!("no cfg_select! arm applies");
        };
        if !arm.braced {
            out.lexed("(");
        }
        out.copy_range(run.src, arm.from, arm.to);
        if !arm.braced {
            out.lexed(")");
        }
        Ok(true)
    }

    fn panic(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) {
        let edition = if cx.names.crates[self.home.krate as usize].edition.as_str() < "2021" {
            "2015"
        } else {
            "2021"
        };
        let name = if site.builtin == Builtin::Panic {
            "panic"
        } else {
            "unreachable"
        };
        builtin::crate_path(out, site.krate, &format!("::panic::{name}_{edition}!"));
        out.lexed("(");
        out.copy_range(run.src, site.open + 1, run.src.pair(site.open));
        out.lexed(")");
    }

    fn asm(&self, cx: &mut Context<'_>, run: Run<'_>, open: u32, out: &mut Tokens) -> Result<()> {
        let args = run.src.args(open);
        let mut templates = Vec::new();
        for &range in &args {
            let Some(template) = self.literal(cx, run, range)? else {
                break;
            };
            templates.push(template);
        }
        builtin::asm(run.src, &templates, &args[templates.len()..], out)
    }

    fn concat(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        open: u32,
        out: &mut Tokens,
    ) -> Result<bool> {
        let mut text = String::new();
        for range in run.src.args(open) {
            let Some(piece) = self.literal(cx, run, range)? else {
                return Ok(false);
            };
            text.push_str(&piece);
        }
        out.lexed(&format!("{text:?}"));
        Ok(true)
    }

    fn env(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        site: Site,
        out: &mut Tokens,
    ) -> Result<bool> {
        let Some(&name) = run.src.args(site.open).first() else {
            bail!("env! takes the name of a variable");
        };
        let Some(name) = self.literal(cx, run, name)? else {
            return Ok(false);
        };
        match (
            site.builtin,
            cx.outline.crates[self.home.krate as usize].env.var(&name),
        ) {
            (Builtin::Env, Some(value)) => out.lexed(&format!("{value:?}")),
            (Builtin::Env, None) => {
                bail!("environment variable `{name}` is not defined at compile time")
            }
            (_, Some(value)) => {
                builtin::crate_path(out, site.krate, "::option::Option::Some(");
                out.lexed(&format!("{value:?})"));
            }
            (_, None) => {
                builtin::crate_path(out, site.krate, "::option::Option::None::<&'static str>")
            }
        }
        Ok(true)
    }

    fn include(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        site: Site,
        out: &mut Tokens,
    ) -> Result<bool> {
        let Some(&path) = run.src.args(site.open).first() else {
            bail!("include!, include_str! and include_bytes! take a path");
        };
        let Some(path) = self.literal(cx, run, path)? else {
            return Ok(false);
        };
        let path = run.call_at(site.start).0.beside(&path);
        let bytes = std::fs::read(&path)
            .map_err(|error| anyhow!("cannot read {}: {error}", path.display()))?;
        let text = |bytes: Vec<u8>| {
            String::from_utf8(bytes)
                .map_err(|error| anyhow!("{} is not UTF-8: {error}", path.display()))
        };
        match site.builtin {
            Builtin::Include => out.lexed(&text(bytes)?),
            Builtin::IncludeStr => out.lexed(&format!("{:?}", text(bytes)?)),
            _ => out.lexed(&builtin::byte_string(&bytes)),
        }
        Ok(true)
    }

    fn format_args(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        site: Site,
        out: &mut Tokens,
    ) -> Result<bool> {
        let args = run.src.args(site.open);
        let Some((&template, rest)) = args.split_first() else {
            return Ok(false);
        };
        let start = template.0;
        let Some(mut template) = self.literal(cx, run, template)? else {
            return Ok(false);
        };
        if site.builtin == Builtin::FormatArgsNl {
            template.push('\n');
        }
        if rest.is_empty() && builtin::format_literal(&template, site.krate, out) {
            return Ok(true);
        }
        format::expand(
            &template,
            &format::Call::new(run.src, start, rest),
            site.krate,
            out,
        )?;
        Ok(true)
    }

    pub(super) fn literal(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        (from, to): (u32, u32),
    ) -> Result<Option<String>> {
        let src = run.src;
        if to > from + 1 && src.kind(from) == TokenKind::OpenParen && src.pair(from) == to - 1 {
            return self.literal(cx, run, (from + 1, to - 1));
        }
        if to == from + 1 {
            return src.literal_text(from);
        }
        if to == from + 2 && src.text(from) == "-" {
            return Ok(src.literal_text(from + 1)?.map(|text| format!("-{text}")));
        }
        let open = src.pair(to - 1);
        if !matches!(
            src.kind(to - 1),
            TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace
        ) || open <= from
            || src.kind(open - 1) != TokenKind::Bang
        {
            return Ok(None);
        }
        let Some(path) = call_path(src, open - 1, from).filter(|path| path.0 == from) else {
            return Ok(None);
        };
        let call = run.call_at(path.0);
        let Some(output) = self.expand_one(
            cx,
            Run {
                depth: run.depth + 1,
                call,
                ..run
            },
            path,
            open,
        )?
        else {
            return Ok(None);
        };
        self.literal(
            cx,
            Run {
                src: &output,
                from: 0,
                shift: 0,
                depth: run.depth + 1,
                call,
            },
            (0, output.len()),
        )
    }
}

fn position(run: Run<'_>, site: Site, input: (u32, u32), out: &mut Tokens) {
    let (src, at) = run.call_at(site.start);
    match site.builtin {
        Builtin::Line => out.lexed(&format!("{}u32", src.line(at))),
        Builtin::Column => out.lexed(&format!("{}u32", src.column(at))),
        Builtin::File => out.lexed(&format!("{:?}", src.path.display().to_string())),
        _ => out.lexed(&format!(
            "{:?}",
            crate::outline::build::spelled_fragment(run.src, input)
        )),
    }
}
