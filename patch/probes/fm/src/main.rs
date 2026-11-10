use std::marker::PhantomData;

mod combinator {
    use super::*;

    pub trait Parser<I, O, E> {
        fn parse_next(&mut self, input: &mut I) -> Result<O, E>;
    
        fn span(self) -> Span<Self, I, O, E>
        where
            Self: Sized,
            I: Stream + Location,
        {
            Span { parser: self, i: PhantomData, o: PhantomData, e: PhantomData }
        }
    
        fn map<G, O2>(self, map: G) -> Map<Self, G, O>
        where
            G: FnMut(O) -> O2,
            Self: Sized,
        {
            Map { parser: self, map, o: PhantomData }
        }
    }
    
    impl<I: Stream, O, E, F> Parser<I, O, E> for F
    where
        F: FnMut(&mut I) -> Result<O, E>,
    {
        fn parse_next(&mut self, input: &mut I) -> Result<O, E> {
            self(input)
        }
    }
}

mod prelude {
    pub(crate) use crate::combinator::Parser as _;
    pub(crate) use crate::Modal;
}

#[derive(Debug)]
pub struct Wrapped<E>(E);

#[derive(Debug, Default)]
pub struct Failure;

pub trait Modal<I, O, E>: combinator::Parser<I, O, Wrapped<E>> {}

impl<I, O, E, P: combinator::Parser<I, O, Wrapped<E>>> Modal<I, O, E> for P {}

fn modal_letter<'i>() -> impl Modal<Input<'i>, char, Failure> {
    move |i: &mut Input<'i>| i.take().ok_or(Wrapped(Failure))
}

use prelude::*;

trait Stream {
    fn take(&mut self) -> Option<char>;
}

trait Location {
    fn at(&self) -> usize;
}

struct Located<'a> {
    text: &'a str,
    used: usize,
}

impl Stream for Located<'_> {
    fn take(&mut self) -> Option<char> {
        let first = self.text.chars().next()?;
        self.text = &self.text[first.len_utf8()..];
        self.used += first.len_utf8();
        Some(first)
    }
}

impl Location for Located<'_> {
    fn at(&self) -> usize {
        self.used
    }
}

type Outcome<O, E = ()> = Result<O, E>;

struct Span<P, I, O, E> {
    parser: P,
    i: PhantomData<I>,
    o: PhantomData<O>,
    e: PhantomData<E>,
}

impl<P: combinator::Parser<I, O, E>, I: Stream + Location, O, E> combinator::Parser<I, std::ops::Range<usize>, E> for Span<P, I, O, E> {
    fn parse_next(&mut self, input: &mut I) -> Result<std::ops::Range<usize>, E> {
        let start = input.at();
        self.parser.parse_next(input)?;
        Ok(start..input.at())
    }
}

struct Map<P, G, O> {
    parser: P,
    map: G,
    o: PhantomData<O>,
}

impl<P: combinator::Parser<I, O, E>, G: FnMut(O) -> O2, I, O, O2, E> combinator::Parser<I, O2, E> for Map<P, G, O> {
    fn parse_next(&mut self, input: &mut I) -> Result<O2, E> {
        self.parser.parse_next(input).map(&mut self.map)
    }
}

type Input<'b> = Located<'b>;

fn letter<'i>(input: &mut Input<'i>) -> Outcome<char> {
    input.take().ok_or(())
}

fn spans(seen: &std::cell::RefCell<Vec<std::ops::Range<usize>>>) -> impl combinator::Parser<Input<'_>, (), ()> + '_ {
    move |i: &mut Input<'_>| letter.span().map(|span| seen.borrow_mut().push(span)).parse_next(i)
}

fn cut<I: Stream, O, E, P: combinator::Parser<I, O, E>>(mut parser: P) -> impl combinator::Parser<I, O, E> {
    move |input: &mut I| parser.parse_next(input)
}

fn skip<'s, 'i>(seen: &'s std::cell::RefCell<Vec<std::ops::Range<usize>>>) -> impl combinator::Parser<Input<'i>, (), ()> + 's {
    move |i: &mut Input<'i>| letter.span().map(|span| seen.borrow_mut().push(span)).parse_next(i)
}

fn pick<'s, 'i>(seen: &'s std::cell::RefCell<Vec<std::ops::Range<usize>>>) -> impl combinator::Parser<Input<'i>, (), ()> + 's {
    move |i: &mut Input<'i>| {
        let mut chosen = move |input: &mut _| {
            let first = letter.parse_next(input)?;
            match first {
                'h' => cut(skip(seen)).parse_next(input),
                _ => Ok(()),
            }
        };
        chosen.parse_next(i)
    }
}

fn main() {
    let mut modal_text = Located { text: "zx", used: 0 };
    let modal = (|| modal_letter())().parse_next(&mut modal_text);
    println!("modal {modal:?} rest {}", modal_text.text);
    let picked = std::cell::RefCell::new(Vec::new());
    let mut text = Located { text: "hello", used: 0 };
    pick(&picked).parse_next(&mut text).unwrap();
    println!("picked {:?} rest {}", picked.borrow(), text.text);
    let seen = std::cell::RefCell::new(Vec::new());
    let mut input = Located { text: "héllo", used: 0 };
    let mut parser = spans(&seen);
    for _ in 0..3 {
        parser.parse_next(&mut input).unwrap();
    }
    println!("{:?} rest {}", seen.borrow(), input.text);
}
