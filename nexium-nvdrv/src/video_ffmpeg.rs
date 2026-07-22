use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;

pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let opted_out = std::env::var("NEXIUM_VIDEO_FFMPEG")
            .map(|value| value == "0")
            .unwrap_or(false);
        !opted_out && resolve_binary().is_some()
    })
}

pub fn resolve_binary() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("NEXIUM_FFMPEG") {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    for dir in std::env::split_paths(&std::env::var_os("PATH")?) {
        let candidate = dir.join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub struct FfmpegDecoder {
    child: Child,
    stdin: Option<ChildStdin>,
    frames: mpsc::Receiver<Vec<u8>>,
    width: u32,
    height: u32,
    primed: bool,
}

impl FfmpegDecoder {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(format!("unsupported dimensions {}x{}", width, height));
        }
        let binary = resolve_binary().ok_or("ffmpeg binary not found")?;
        let mut child = Command::new(binary)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-probesize",
                "32",
                "-analyzeduration",
                "0",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("spawn ffmpeg: {}", error))?;
        let mut stdout = child.stdout.take().ok_or("ffmpeg stdout unavailable")?;
        let stdin = child.stdin.take();
        let (tx, rx) = mpsc::channel();
        let frame_len = width as usize * height as usize * 3 / 2;
        std::thread::Builder::new()
            .name("ffmpeg-video-out".into())
            .spawn(move || {
                let mut buf = vec![0u8; frame_len];
                loop {
                    let mut filled = 0usize;
                    while filled < frame_len {
                        match stdout.read(&mut buf[filled..]) {
                            Ok(0) => return,
                            Ok(read) => filled += read,
                            Err(_) => return,
                        }
                    }
                    if tx.send(buf.clone()).is_err() {
                        return;
                    }
                }
            })
            .map_err(|error| format!("spawn ffmpeg reader: {}", error))?;
        Ok(Self {
            child,
            stdin: Some(stdin.ok_or("ffmpeg stdin unavailable")?),
            frames: rx,
            width,
            height,
            primed: false,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let stdin = self.stdin.as_mut().ok_or("ffmpeg stdin closed")?;
        stdin
            .write_all(packet)
            .and_then(|_| stdin.flush())
            .map_err(|error| format!("ffmpeg write: {}", error))?;
        let wait = if self.primed { 700 } else { 250 };
        match self
            .frames
            .recv_timeout(std::time::Duration::from_millis(wait))
        {
            Ok(frame) => {
                self.primed = true;
                Ok(Some(frame))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err("ffmpeg exited".into()),
        }
    }
}

impl Drop for FfmpegDecoder {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn i420_frame(
    width: u32,
    height: u32,
    raw: &[u8],
) -> Result<crate::video_decode::OwnedI420Frame, String> {
    let w = width as usize;
    let h = height as usize;
    let y_len = w * h;
    let c_len = y_len / 4;
    if raw.len() < y_len + 2 * c_len {
        return Err(format!("short i420 buffer {} for {}x{}", raw.len(), w, h));
    }
    crate::video_decode::OwnedI420Frame::from_strided_planes(
        width as usize,
        height as usize,
        (w, w / 2, w / 2),
        &raw[..y_len],
        &raw[y_len..y_len + c_len],
        &raw[y_len + c_len..y_len + 2 * c_len],
    )
    .map_err(|error| format!("i420 copy: {}", error))
}
