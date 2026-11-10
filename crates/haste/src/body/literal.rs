use std::iter::Peekable;
use std::str::Chars;

use anyhow::{Result, anyhow, bail};

const ESCAPES: [(char, char); 7] = [('n', '\n'), ('t', '\t'), ('r', '\r'), ('0', '\0'), ('\\', '\\'), ('\'', '\''), ('"', '"')];

/// The value of a char literal, quotes included.
pub fn char_value(text: &str) -> Result<char> {
    let out = match unit(text)? {
        Ok(ch) => return Ok(ch),
        Err(out) => out,
    };
    let decoded = String::from_utf8(out).map_err(|_| anyhow!("bad char {text}"))?;
    decoded.chars().next().ok_or_else(|| anyhow!("bad char {text}"))
}

/// The value of a byte literal, quotes included.
pub fn byte_value(text: &str) -> Result<u8> {
    match unit(text)? {
        Ok(ch) => u8::try_from(ch).map_err(|_| anyhow!("bad byte {text}")),
        Err(out) => out.first().copied().ok_or_else(|| anyhow!("bad byte {text}")),
    }
}

/// The unescaped character of a char or byte literal, or the bytes its escape decodes to.
fn unit(text: &str) -> Result<std::result::Result<char, Vec<u8>>> {
    let inner = text.trim_start_matches('b').strip_prefix('\'').and_then(|rest| rest.strip_suffix('\'')).ok_or_else(|| anyhow!("bad char literal {text}"))?;
    let mut chars = inner.chars().peekable();
    let mut out = Vec::new();
    match chars.next() {
        Some('\\') => escape(&mut chars, &mut out)?,
        Some(ch) => return Ok(Ok(ch)),
        None => bail!("empty char literal"),
    }
    Ok(Err(out))
}

/// The bytes of a string or byte string literal, prefix and quotes included, with its escapes decoded.
pub fn string_bytes(text: &str) -> Result<Vec<u8>> {
    let text = text.strip_prefix(['b', 'c']).unwrap_or(text);
    if let Some(raw) = text.strip_prefix('r') {
        let hashes = raw.len() - raw.trim_start_matches('#').len();
        return raw.get(hashes + 1..raw.len() - hashes - 1).map(|inner| inner.as_bytes().to_vec()).ok_or_else(|| anyhow!("bad raw string {text}"));
    }
    let inner = text.strip_prefix('"').and_then(|rest| rest.strip_suffix('"')).ok_or_else(|| anyhow!("bad string literal {text}"))?;
    let mut out = Vec::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => escape(&mut chars, &mut out)?,
            '\r' if chars.peek() == Some(&'\n') => {}
            ch => out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }
    Ok(out)
}

fn escape(chars: &mut Peekable<Chars<'_>>, out: &mut Vec<u8>) -> Result<()> {
    let Some(kind) = chars.next() else { bail!("escape at the end of a literal") };
    match kind {
        '\n' | '\r' => {
            while chars.peek().is_some_and(|ch| ch.is_whitespace()) {
                chars.next();
            }
        }
        'x' => {
            let digits: String = chars.by_ref().take(2).collect();
            out.push(u8::from_str_radix(&digits, 16)?);
        }
        'u' => {
            let digits: String = chars.by_ref().skip(1).take_while(|ch| *ch != '}').filter(|ch| *ch != '_').collect();
            let ch = char::from_u32(u32::from_str_radix(&digits, 16)?).ok_or_else(|| anyhow!("bad unicode escape"))?;
            out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes());
        }
        other => {
            let &(_, ch) = ESCAPES.iter().find(|(name, _)| *name == other).ok_or_else(|| anyhow!("bad escape \\{other}"))?;
            out.push(ch as u8);
        }
    }
    Ok(())
}
