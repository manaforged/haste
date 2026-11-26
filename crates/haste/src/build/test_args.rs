pub(super) struct TestArgs<'a> {
    pub(super) filters: Vec<&'a str>,
    pub(super) skips: Vec<&'a str>,
    pub(super) exact: bool,
    pub(super) ignored_only: bool,
    pub(super) include_ignored: bool,
}

pub(super) fn parse_test_args(args: &[String]) -> TestArgs<'_> {
    let mut parsed = TestArgs { filters: Vec::new(), skips: Vec::new(), exact: false, ignored_only: false, include_ignored: false };
    let mut iter = args.iter().map(String::as_str);
    while let Some(arg) = iter.next() {
        if arg == "--" {
            continue;
        } else if arg == "--exact" {
            parsed.exact = true;
        } else if arg == "--ignored" {
            parsed.ignored_only = true;
        } else if arg == "--include-ignored" {
            parsed.include_ignored = true;
        } else if let Some(val) = arg.strip_prefix("--skip=") {
            parsed.skips.push(val);
        } else if arg == "--skip" {
            if let Some(val) = iter.next() {
                parsed.skips.push(val);
            }
        } else if matches!(arg, "--test-threads" | "--color" | "--format" | "--logfile" | "--shuffle-seed" | "-Z") {
            let _ = iter.next();
        } else if !arg.starts_with('-') {
            parsed.filters.push(arg);
        }
    }
    parsed
}
