use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FfmpegCodec {
    H264,
    Vp9,
}

impl FfmpegCodec {
    fn input_format(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Vp9 => "ivf",
        }
    }
}

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
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(dir) = current_exe.parent() {
            let candidate = dir.join(exe);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
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
    codec: FfmpegCodec,
    ivf_header_sent: bool,
    ivf_pts: u64,
}

impl FfmpegDecoder {
    pub fn new(width: u32, height: u32, codec: FfmpegCodec) -> Result<Self, String> {
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
                "-flags",
                "low_delay",
                "-threads",
                "1",
                "-f",
                codec.input_format(),
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-threads:v",
                "1",
                "-fps_mode",
                "passthrough",
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
            codec,
            ivf_header_sent: false,
            ivf_pts: 0,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn codec(&self) -> FfmpegCodec {
        self.codec
    }

    pub fn submit(&mut self, packet: &[u8]) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("ffmpeg stdin closed")?;
        if self.codec == FfmpegCodec::Vp9 {
            if !self.ivf_header_sent {
                let header =
                    crate::video_vp9::ivf_file_header(self.width as u16, self.height as u16);
                stdin
                    .write_all(&header)
                    .map_err(|error| format!("ffmpeg write ivf header: {}", error))?;
                self.ivf_header_sent = true;
            }
            let frame_header =
                crate::video_vp9::ivf_frame_header(packet.len() as u32, self.ivf_pts);
            self.ivf_pts += 1;
            stdin
                .write_all(&frame_header)
                .map_err(|error| format!("ffmpeg write ivf frame header: {}", error))?;
        }
        stdin
            .write_all(packet)
            .and_then(|_| stdin.flush())
            .map_err(|error| format!("ffmpeg write: {}", error))
    }

    pub fn receive(&mut self) -> Result<Option<Vec<u8>>, String> {
        match self.frames.try_recv() {
            Ok(frame) => Ok(Some(frame)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("ffmpeg exited".into()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const VP9_FIXTURE: &[u8] = include_bytes!("../tests/fixtures/vp9_hidden_frames.ivf");

    fn fixture_packets() -> Vec<&'static [u8]> {
        assert_eq!(&VP9_FIXTURE[..4], b"DKIF");
        let mut offset = 32;
        let mut packets = Vec::new();
        while offset < VP9_FIXTURE.len() {
            let size =
                u32::from_le_bytes(VP9_FIXTURE[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 12;
            packets.push(&VP9_FIXTURE[offset..offset + size]);
            offset += size;
        }
        assert_eq!(packets.len(), 24);
        packets
    }

    fn assert_fixture_frame(frame: &[u8], packet_index: usize) {
        assert_eq!(frame.len(), 16 * 16 * 3 / 2);
        let luma = 16 + 8 * packet_index as u8;
        assert!(
            frame[..256].iter().all(|&value| value == luma),
            "packet {packet_index}: expected luma {luma}, got {}",
            frame[0]
        );
        assert!(frame[256..].iter().all(|&value| value == 128));
    }

    #[test]
    #[ignore = "requires FFmpeg; set NEXIUM_FFMPEG and run with --ignored"]
    fn vp9_shown_frames_arrive_without_later_input() {
        let mut decoder = FfmpegDecoder::new(16, 16, FfmpegCodec::Vp9).unwrap();
        for (index, packet) in fixture_packets().into_iter().enumerate() {
            decoder.submit(packet).unwrap();
            if index % 3 == 1 {
                assert_eq!(
                    decoder.frames.recv_timeout(Duration::from_millis(20)),
                    Err(mpsc::RecvTimeoutError::Timeout),
                    "hidden packet {index} must not produce a picture"
                );
            } else {
                let frame = decoder
                    .frames
                    .recv_timeout(Duration::from_secs(3))
                    .unwrap_or_else(|error| panic!("packet {index} needed later input: {error}"));
                assert_fixture_frame(&frame, index);
            }
        }
        decoder.stdin.take();
        assert_eq!(
            decoder.frames.recv_timeout(Duration::from_secs(3)),
            Err(mpsc::RecvTimeoutError::Disconnected),
            "all shown frames must arrive before EOF, with no duplicates"
        );
    }

    #[test]
    #[ignore = "requires FFmpeg; set NEXIUM_FFMPEG and run with --ignored"]
    fn vp9_hidden_frames_do_not_duplicate_output() {
        let mut decoder = FfmpegDecoder::new(16, 16, FfmpegCodec::Vp9).unwrap();
        for packet in fixture_packets() {
            decoder.submit(packet).unwrap();
        }
        decoder.stdin.take();
        for index in (0..24).filter(|index| index % 3 != 1) {
            let frame = decoder
                .frames
                .recv_timeout(Duration::from_secs(3))
                .unwrap_or_else(|error| panic!("missing shown packet {index}: {error}"));
            assert_fixture_frame(&frame, index);
        }
        assert_eq!(
            decoder.frames.recv_timeout(Duration::from_secs(3)),
            Err(mpsc::RecvTimeoutError::Disconnected),
            "24 input packets contain exactly 16 shown pictures"
        );
    }
}
