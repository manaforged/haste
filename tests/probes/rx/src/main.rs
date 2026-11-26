use regex::Regex;

fn main() {
    let re = Regex::new(r"(?P<key>[a-z_]+)=(?P<val>[0-9]+)").expect("valid regex");
    let text = "alpha=10 beta=25 gamma=300 delta=42 skip=abc eps_1=7";
    let mut sum = 0u64;
    let mut names = Vec::new();
    for cap in re.captures_iter(text) {
        names.push(cap["key"].to_owned());
        sum += cap["val"].parse::<u64>().expect("digits");
    }
    let replaced = re.replace_all("a=1 b=2", "$key:$val");
    println!("{} {} {}", names.join(","), sum, replaced);
}
