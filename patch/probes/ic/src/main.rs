fn main() {
    let names: Vec<String> = vec!["x".to_string(), "y".to_string()].into_iter().map(|name| name + "!").collect();
    let same: Vec<String> = names.clone().into_iter().collect();
    let lens: Vec<usize> = names.into_iter().map(|name| name.len()).collect();
    let kept: Vec<u64> = vec![1u64, 2, 3, 4].into_iter().filter(|n| n % 2 == 0).collect();
    println!("{same:?} {lens:?} {kept:?}");
}
