use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

fn handle() -> &'static str {
    "v1: edit this string in examples/live-server/src/main.rs and save\n"
}

fn serve(mut stream: TcpStream) {
    let mut buf = [0u8; 512];
    let _ = stream.read(&mut buf);
    let body = format!("pid={} | {}", std::process::id(), handle());
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(resp.as_bytes());
}

fn main() {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:8080".to_owned());
    let listener = TcpListener::bind(&addr).expect("bind listener");
    eprintln!("live-server listening on http://{addr} (pid={})", std::process::id());
    for stream in listener.incoming().flatten() {
        serve(stream);
    }
}
