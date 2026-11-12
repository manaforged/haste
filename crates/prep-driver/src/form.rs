//! How a key writes a type from the keys of its parts, the one form the driver prints and Haste prints and reads.

/// The key of the never type.
pub const NEVER: &str = "!";
/// The prefixes of a shared and a mutable reference, and of a const and a mutable raw pointer.
pub const SHARED_REF: &str = "&";
pub const MUT_REF: &str = "&mut ";
pub const CONST_PTR: &str = "*const ";
pub const MUT_PTR: &str = "*mut ";
/// What opens a function pointer type's parameters and what closes them before its return type.
pub const FN_PTR: &str = "fn(";
pub const RETURNS: &str = ")->";

/// A reference to `inner`, mutable or shared.
pub fn reference(mutable: bool, inner: &str) -> String {
    format!("{}{inner}", if mutable { MUT_REF } else { SHARED_REF })
}

/// A raw pointer to `inner`, mutable or const.
pub fn raw(mutable: bool, inner: &str) -> String {
    format!("{}{inner}", if mutable { MUT_PTR } else { CONST_PTR })
}

/// An array of `len` elements of `inner`.
pub fn array(inner: &str, len: &str) -> String {
    format!("[{inner};{len}]")
}

/// A slice of `inner`.
pub fn slice(inner: &str) -> String {
    format!("[{inner}]")
}

/// A tuple of `items`, each followed by a comma.
pub fn tuple(items: &[String]) -> String {
    format!("({})", terminated(items))
}

/// Definition `def` with the arguments `args`.
pub fn named(def: &str, args: &[String]) -> String {
    format!("{def}<{}>", args.join(","))
}

/// The function item type of definition `def` with the arguments `args`.
pub fn fn_def(def: &str, args: &[String]) -> String {
    format!("fn {}", named(def, args))
}

/// A function pointer type that takes `params` and returns `ret`.
pub fn fn_ptr(params: &[String], ret: &str) -> String {
    format!("{FN_PTR}{}{RETURNS}{ret}", terminated(params))
}

/// A trait object type of trait `def` with the arguments `args`, fixing each associated type of `fixed` by name, in name order.
pub fn object(def: &str, args: &[String], fixed: &[(String, String)]) -> String {
    let mut bound: Vec<String> = fixed.iter().map(|(name, ty)| format!("+{name}={ty}")).collect();
    bound.sort();
    format!("dyn {}{}", named(def, args), bound.concat())
}

/// Closure number `order` of `count` in the body of `root`, with the root's arguments `args`.
pub fn closure(root: &str, (order, count): (usize, usize), args: &[String]) -> String {
    format!("closure {root}#{order}/{count}<{}>", args.join(","))
}

fn terminated(items: &[String]) -> String {
    items.iter().map(|item| format!("{item},")).collect()
}
