fn field_i(i: usize) -> String {
    format!("__field{i}")
}

fn main() {
    let vars = (0..3).map(field_i);
    let pairs: Vec<(String, char)> = vars.clone().zip("abc".chars()).collect();
    println!("{pairs:?} {}", vars.count());
    let offset = 10;
    let shifted = (0..2).map(move |i| i + offset);
    closures();
    pointers();
    println!("{:?} {:?}", shifted.clone().collect::<Vec<_>>(), shifted.sum::<usize>());
}

fn closures() {
    let name = String::from("x");
    let greet = move || format!("hi {name}");
    let again = greet.clone();
    println!("{} {}", greet(), again());
}

fn pointers() {
    let parse: fn(&str) -> usize = str::len;
    let copies: Vec<fn(&str) -> usize> = vec![parse; 2];
    let again = copies[1].clone();
    println!("{} {}", copies[0]("abc"), again("hello"));
}
