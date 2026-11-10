use std::cell::RefCell;
use winnow::combinator::{alt, cut_err, eof, opt, peek, repeat};
use winnow::prelude::*;
use winnow::stream::LocatingSlice;
use winnow::token::{any, take_while};

type Input<'i> = LocatingSlice<&'i [u8]>;

fn comment<'i>(input: &mut Input<'i>) -> ModalResult<&'i [u8]> {
    (b'#', take_while(0.., |byte: u8| byte != b'\n')).take().parse_next(input)
}

fn line_ending<'i>(input: &mut Input<'i>) -> ModalResult<&'i [u8]> {
    alt((b"\n".value(&b"\n"[..]), eof.value(&b""[..]))).parse_next(input)
}

fn parse_comment<'s, 'i>(state: &'s RefCell<Vec<usize>>) -> impl ModalParser<Input<'i>, (), winnow::error::ContextError> + 's {
    move |i: &mut Input<'i>| {
        (comment, line_ending).span().map(|span| state.borrow_mut().push(span.start)).parse_next(i)
    }
}

fn other<'i>(input: &mut Input<'i>) -> ModalResult<()> {
    take_while(1.., |byte: u8| byte != b'#').void().parse_next(input)
}

fn document<'s, 'i>(state: &'s RefCell<Vec<usize>>) -> impl ModalParser<Input<'i>, (), winnow::error::ContextError> + 's {
    move |i: &mut Input<'i>| {
        (
            opt(b"\xEF\xBB\xBF"),
            repeat(0.., (
                winnow::combinator::dispatch! {peek(any);
                    b'#' => cut_err(parse_comment(state)),
                    _ => other,
                },
                opt(b" "),
            ))
            .map(|()| ()),
            eof,
        ).void().parse_next(i)
    }
}

fn main() {
    let state = RefCell::new(Vec::new());
    document(&state).parse(LocatingSlice::new(&b"# note\nrest# two"[..])).unwrap();
    println!("{:?}", state.into_inner());
}
