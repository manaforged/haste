use proc_macro::{Delimiter, TokenStream, TokenTree};

fn name_after(input: &TokenStream, keyword: &str) -> String {
    let mut tokens = input.clone().into_iter();
    while let Some(token) = tokens.next() {
        if let TokenTree::Ident(ident) = &token
            && ident.to_string() == keyword
            && let Some(TokenTree::Ident(name)) = tokens.next()
        {
            return name.to_string();
        }
    }
    panic!("no {keyword}")
}

fn fields(input: &TokenStream) -> Vec<String> {
    let body = input.clone().into_iter().find_map(|token| match token {
        TokenTree::Group(group) if group.delimiter() == Delimiter::Brace => Some(group.stream()),
        _ => None,
    });
    let tokens: Vec<TokenTree> = body.expect("braced struct").into_iter().collect();
    let mut found = Vec::new();
    for pair in tokens.windows(2) {
        if let (TokenTree::Ident(name), TokenTree::Punct(colon)) = (&pair[0], &pair[1])
            && colon.as_char() == ':'
        {
            found.push(name.to_string());
        }
    }
    found
}

#[proc_macro_derive(Total, attributes(skip))]
pub fn derive_total(input: TokenStream) -> TokenStream {
    let name = name_after(&input, "struct");
    let sum: String = fields(&input).iter().map(|field| format!(" + self.{field} as i64")).collect();
    format!("impl {name} {{ pub fn total(&self) -> i64 {{ 0{sum} }} pub fn field_count() -> usize {{ {} }} }}", fields(&input).len())
        .parse()
        .unwrap()
}

#[proc_macro_attribute]
pub fn doubled(attr: TokenStream, item: TokenStream) -> TokenStream {
    let name = name_after(&item, "fn");
    let factor = attr.to_string();
    let factor = if factor.is_empty() { "2".to_owned() } else { factor };
    let extra: TokenStream = format!("pub fn {name}_doubled(x: i64) -> i64 {{ {name}(x) * {factor} }}").parse().unwrap();
    let mut out = item;
    out.extend(extra);
    out
}

#[proc_macro]
pub fn make_const(input: TokenStream) -> TokenStream {
    let parts: Vec<String> = input.to_string().split(',').map(|part| part.trim().to_owned()).collect();
    format!("pub const {}: i64 = {} * 3;", parts[0], parts[1]).parse().unwrap()
}

#[proc_macro]
pub fn literals(_input: TokenStream) -> TokenStream {
    let mut out = TokenStream::new();
    out.extend([TokenTree::Literal(proc_macro::Literal::i64_suffixed(40)), TokenTree::Punct(proc_macro::Punct::new('+', proc_macro::Spacing::Alone)), TokenTree::Literal(proc_macro::Literal::u8_unsuffixed(2))]);
    out
}
