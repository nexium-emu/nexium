use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const REGISTERED_PORT: u16 = 47823;
const POLL_INTERVAL: Duration = Duration::from_millis(80);
const REQUEST_LIMIT: usize = 16 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthError {
    Cancelled,
    TimedOut,
    Denied,
    StateMismatch,
    Server(String),
    Local(String),
}

impl AuthError {
    pub fn describe(&self) -> String {
        match self {
            AuthError::Cancelled => "Sign-in was cancelled.".into(),
            AuthError::TimedOut => "Sign-in timed out. Try again when you're ready.".into(),
            AuthError::Denied => "You declined NeXium's request in the browser.".into(),
            AuthError::StateMismatch => {
                "The browser returned a sign-in NeXium didn't start, so it was ignored.".into()
            }
            AuthError::Server(message) => message.clone(),
            AuthError::Local(message) => message.clone(),
        }
    }
}

pub struct Grant {
    pub code: String,
    pub verifier: String,
    pub redirect_uri: String,
}

pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

pub fn random_token(bytes: usize) -> String {
    let mut buffer = vec![0u8; bytes];
    if getrandom::fill(&mut buffer).is_err() {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default();
        for (index, byte) in buffer.iter_mut().enumerate() {
            *byte = (seed >> ((index % 16) * 8)) as u8 ^ (index as u8).wrapping_mul(0x5B);
        }
    }
    URL_SAFE_NO_PAD.encode(buffer)
}

pub fn challenge_for(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

impl Pkce {
    pub fn new() -> Self {
        let verifier = random_token(32);
        let challenge = challenge_for(&verifier);
        Self {
            verifier,
            challenge,
        }
    }
}

pub fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 3);
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

pub fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => out.push(b' '),
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(value) => {
                        out.push(value);
                        index += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            byte => out.push(byte),
        }
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn authorize_url(base: &str, redirect_uri: &str, challenge: &str, state: &str) -> String {
    let params = [
        ("client_id", super::api::CLIENT_ID),
        ("redirect_uri", redirect_uri),
        ("response_type", "code"),
        ("scope", super::api::SCOPES),
        ("state", state),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
    ];
    let query: Vec<String> = params
        .iter()
        .map(|(key, value)| format!("{key}={}", percent_encode(value)))
        .collect();
    format!("{base}/oauth/authorize?{}", query.join("&"))
}

pub fn parse_query(query: &str) -> HashMap<String, String> {
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(key), percent_decode(value))
        })
        .collect()
}

pub fn request_target(request: &str) -> Option<(&str, &str)> {
    let line = request.lines().next()?;
    let mut parts = line.split_whitespace();
    if parts.next()? != "GET" {
        return None;
    }
    let target = parts.next()?;
    Some(target.split_once('?').unwrap_or((target, "")))
}

pub struct LoopbackServer {
    listener: TcpListener,
    port: u16,
}

impl LoopbackServer {
    pub fn bind() -> io::Result<Self> {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, REGISTERED_PORT))
            .or_else(|_| TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)))?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        Ok(Self { listener, port })
    }

    pub fn redirect_uri(&self) -> String {
        format!("http://127.0.0.1:{}/callback", self.port)
    }

    pub fn wait(&self, state: &str, cancel: &AtomicBool, timeout: Duration) -> Result<String, AuthError> {
        let deadline = Instant::now() + timeout;
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(AuthError::Cancelled);
            }
            if Instant::now() >= deadline {
                return Err(AuthError::TimedOut);
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if let Some(outcome) = handle_connection(stream, state) {
                        return outcome;
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(POLL_INTERVAL);
                }
                Err(error) => return Err(AuthError::Local(error.to_string())),
            }
        }
    }
}

fn read_request(stream: &mut TcpStream) -> io::Result<String> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    let mut request = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    while request.len() < REQUEST_LIMIT {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        request.extend_from_slice(&chunk[..read]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    Ok(String::from_utf8_lossy(&request).into_owned())
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn handle_connection(mut stream: TcpStream, state: &str) -> Option<Result<String, AuthError>> {
    let request = read_request(&mut stream).ok()?;
    let Some((path, query)) = request_target(&request) else {
        respond(&mut stream, "400 Bad Request", "");
        return None;
    };
    if path != "/callback" {
        respond(&mut stream, "404 Not Found", "");
        return None;
    }
    let params = parse_query(query);
    let outcome = if let Some(error) = params.get("error") {
        if error == "access_denied" {
            Err(AuthError::Denied)
        } else {
            let detail = params
                .get("error_description")
                .filter(|detail| !detail.is_empty())
                .unwrap_or(error);
            Err(AuthError::Server(format!("Nextendo refused the sign-in: {detail}")))
        }
    } else if params.get("state").map(String::as_str) != Some(state) {
        Err(AuthError::StateMismatch)
    } else {
        match params.get("code").filter(|code| !code.is_empty()) {
            Some(code) => Ok(code.clone()),
            None => Err(AuthError::Server("The browser returned without an authorization code.".into())),
        }
    };
    let page = match &outcome {
        Ok(_) => page(
            true,
            "You're signed in",
            "NeXium is now connected to your Nextendo account. You can close this tab and head back to NeXium.",
        ),
        Err(AuthError::Denied) => page(
            false,
            "Sign-in cancelled",
            "No changes were made. You can close this tab and try again from NeXium whenever you like.",
        ),
        Err(error) => page(false, "Sign-in didn't finish", &error.describe()),
    };
    respond(&mut stream, "200 OK", &page);
    if outcome == Err(AuthError::StateMismatch) {
        return None;
    }
    Some(outcome)
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn page(success: bool, title: &str, detail: &str) -> String {
    let (accent, glyph) = if success {
        ("#2FB4EF", "<path d=\"M14 25.5l7 7 14-15\" />")
    } else {
        ("#F05252", "<path d=\"M17 17l14 14M31 17L17 31\" />")
    };
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeXium · Nextendo</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:radial-gradient(1100px 700px at 50% 0%,#18222f 0%,#0f0f11 60%);color:#ececf0;font:16px/1.55 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif}}
.card{{width:min(440px,100%);background:#18181c;border:1px solid #2a2a32;border-radius:20px;padding:40px 36px 32px;text-align:center;box-shadow:0 24px 64px rgba(0,0,0,.55)}}
.badge{{width:72px;height:72px;margin:0 auto 22px;border-radius:50%;display:grid;place-items:center;background:color-mix(in srgb,{accent} 14%,transparent);border:1px solid color-mix(in srgb,{accent} 45%,transparent)}}
svg{{width:40px;height:40px;fill:none;stroke:{accent};stroke-width:3.4;stroke-linecap:round;stroke-linejoin:round}}
h1{{margin:0 0 10px;font-size:24px;font-weight:650;letter-spacing:.2px}}
p{{margin:0;color:#a3a3b2}}
.brand{{margin-top:28px;padding-top:18px;border-top:1px solid #2a2a32;font-size:13px;color:#707080;letter-spacing:.3px}}
.brand b{{color:#ececf0;font-weight:600}}
</style></head>
<body><main class="card">
<div class="badge"><svg viewBox="0 0 48 48">{glyph}</svg></div>
<h1>{title}</h1>
<p>{detail}</p>
<div class="brand"><b>NeXium</b> &middot; Nextendo Network</div>
</main></body></html>"#,
        accent = accent,
        glyph = glyph,
        title = html_escape(title),
        detail = html_escape(detail),
    )
}

pub fn authorize(
    base: &str,
    cancel: &AtomicBool,
    on_url: impl FnOnce(&str),
) -> Result<Grant, AuthError> {
    let server = LoopbackServer::bind()
        .map_err(|error| AuthError::Local(format!("NeXium couldn't open a local sign-in port: {error}")))?;
    let pkce = Pkce::new();
    let state = random_token(24);
    let redirect_uri = server.redirect_uri();
    on_url(&authorize_url(base, &redirect_uri, &pkce.challenge, &state));
    let code = server.wait(&state, cancel, Duration::from_secs(14 * 60))?;
    Ok(Grant {
        code,
        verifier: pkce.verifier,
        redirect_uri,
    })
}

#[cfg(windows)]
pub fn open_browser(url: &str) -> bool {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
    let wide = |text: &str| -> Vec<u16> { text.encode_utf16().chain(std::iter::once(0)).collect() };
    let operation = wide("open");
    let target = wide(url);
    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            operation.as_ptr(),
            target.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    };
    result as isize > 32
}

#[cfg(target_os = "macos")]
pub fn open_browser(url: &str) -> bool {
    std::process::Command::new("open").arg(url).spawn().is_ok()
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn open_browser(url: &str) -> bool {
    std::process::Command::new("xdg-open").arg(url).spawn().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_matches_the_rfc_example() {
        assert_eq!(
            challenge_for("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        let pkce = Pkce::new();
        assert_eq!(pkce.verifier.len(), 43);
        assert_eq!(pkce.challenge, challenge_for(&pkce.verifier));
    }

    #[test]
    fn authorize_url_carries_every_parameter() {
        let url = authorize_url(
            "https://nextendo.network",
            "http://127.0.0.1:47823/callback",
            "abc",
            "xyz",
        );
        assert!(url.starts_with("https://nextendo.network/oauth/authorize?client_id=nxc_E-rgPv2x_uY9&"));
        assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A47823%2Fcallback"));
        assert!(url.contains("scope=identity%20friends%20presence%20game.matchmaking"));
        assert!(url.contains("code_challenge=abc&code_challenge_method=S256"));
        assert!(url.contains("state=xyz"));
    }

    #[test]
    fn callback_queries_decode() {
        let request = "GET /callback?code=a%2Bb%3D&state=s1 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
        let (path, query) = request_target(request).unwrap();
        assert_eq!(path, "/callback");
        let params = parse_query(query);
        assert_eq!(params["code"], "a+b=");
        assert_eq!(params["state"], "s1");
        assert!(request_target("POST /callback HTTP/1.1").is_none());
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
    }

    #[test]
    fn pages_escape_messages() {
        let html = page(false, "<b>", "a & b");
        assert!(html.contains("&lt;b&gt;"));
        assert!(html.contains("a &amp; b"));
    }

    #[test]
    fn loopback_flow_returns_the_code() {
        let server = LoopbackServer::bind().unwrap();
        let port = server.port;
        let client = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
            stream
                .write_all(b"GET /favicon.ico HTTP/1.1\r\nHost: x\r\n\r\n")
                .unwrap();
            let mut ignored = String::new();
            let _ = stream.read_to_string(&mut ignored);
            let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
            stream
                .write_all(b"GET /callback?code=the-code&state=st HTTP/1.1\r\nHost: x\r\n\r\n")
                .unwrap();
            let mut body = String::new();
            let _ = stream.read_to_string(&mut body);
            body
        });
        let cancel = AtomicBool::new(false);
        let code = server.wait("st", &cancel, Duration::from_secs(10)).unwrap();
        assert_eq!(code, "the-code");
        assert!(client.join().unwrap().contains("You're signed in"));
    }
}
