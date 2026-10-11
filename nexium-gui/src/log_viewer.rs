//! The in-app log viewer (Debug > Logs).
//!
//! Lines arrive from the logger over a bounded queue and land in a ring
//! (`LogStore`) where every line gets a sequence number that never changes.
//! Filtering builds a list of rows (`RowView`) that refer to those numbers,
//! and the selection is a pair of row keys, so it stays on the same lines
//! while new lines arrive, old lines are evicted or the filter changes.
//! Only the visible rows are laid out each frame.

use crate::app_settings::{AppSettings, LogLevel, LogViewerSettings, LOG_RING_MAX, LOG_RING_MIN};
use egui::text::LayoutJob;
use egui::{Color32, FontId, Rect, Sense, TextFormat};
use log::Level;
use nexium_common::file_logger::{self, LogFeed, LogFilter};
use std::collections::VecDeque;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

/// Upper bound on lines moved from the logger queue per frame, so a log
/// flood cannot stall the UI thread.
const MAX_DRAIN_PER_FRAME: usize = 200_000;
/// How often an open window checks the logger queue when nothing else
/// redraws the app.
const LIVE_POLL: std::time::Duration = std::time::Duration::from_millis(100);
const LEVEL_NAMES: [&str; 5] = ["Error", "Warn", "Info", "Debug", "Trace"];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LineSpans {
    pub level: Option<Level>,
    /// `[HH:MM:SS.mmm]`, brackets included.
    pub timestamp: Option<Range<usize>>,
    /// `[ INFO]`, brackets included.
    pub level_tag: Option<Range<usize>>,
    /// The target name, brackets excluded.
    pub target: Option<Range<usize>>,
    /// Byte offset where the message starts.
    pub message: usize,
}

fn bracket_at(line: &str, at: usize) -> Option<(Range<usize>, usize)> {
    let rest = line.get(at..)?;
    if !rest.starts_with('[') {
        return None;
    }
    let close = rest.find(']')?;
    let inner = at + 1..at + close;
    let mut next = at + close + 1;
    if line[next..].starts_with(' ') {
        next += 1;
    }
    Some((inner, next))
}

fn parse_level_tag(tag: &str) -> Option<Level> {
    match tag.trim() {
        "ERROR" => Some(Level::Error),
        "WARN" => Some(Level::Warn),
        "INFO" => Some(Level::Info),
        "DEBUG" => Some(Level::Debug),
        "TRACE" => Some(Level::Trace),
        _ => None,
    }
}

/// Splits a line in the logger format `[HH:MM:SS.mmm] [LEVEL] [target] message`.
/// The timestamp and target are optional; a target is only recognised right
/// after a level tag and only when it looks like a Rust module path.
pub fn parse_line(line: &str) -> LineSpans {
    let mut spans = LineSpans::default();
    let mut pos = 0;
    if let Some((inner, next)) = bracket_at(line, pos) {
        let stamp = &line[inner.clone()];
        if stamp.contains(':')
            && stamp
                .bytes()
                .all(|b| b.is_ascii_digit() || b == b':' || b == b'.')
        {
            spans.timestamp = Some(pos..inner.end + 1);
            pos = next;
        }
    }
    if let Some((inner, next)) = bracket_at(line, pos) {
        if let Some(level) = parse_level_tag(&line[inner.clone()]) {
            spans.level = Some(level);
            spans.level_tag = Some(pos..inner.end + 1);
            pos = next;
            if let Some((inner, next)) = bracket_at(line, pos) {
                let target = &line[inner.clone()];
                if !target.is_empty()
                    && target
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b':')
                {
                    spans.target = Some(inner);
                    pos = next;
                }
            }
        }
    }
    spans.message = pos;
    spans
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub seq: u64,
    pub text: String,
    pub spans: LineSpans,
    /// Target of the record a continuation line belongs to.
    inherited_target: Option<Box<str>>,
    pub chars: usize,
}

impl Entry {
    pub fn level(&self) -> Option<Level> {
        self.spans.level
    }

    pub fn target(&self) -> &str {
        match (&self.inherited_target, &self.spans.target) {
            (Some(target), _) => target,
            (None, Some(range)) => &self.text[range.clone()],
            (None, None) => "",
        }
    }

    pub fn message(&self) -> &str {
        &self.text[self.spans.message..]
    }

    /// Same level, target and message; the timestamp may differ.
    fn same_content(&self, other: &Entry) -> bool {
        self.level() == other.level()
            && self.target() == other.target()
            && self.message() == other.message()
    }
}

/// Ring of received lines. Sequence numbers keep counting across evictions
/// and clears, so a number always names the same line.
#[derive(Default)]
pub struct LogStore {
    entries: VecDeque<Entry>,
    next_seq: u64,
}

impl LogStore {
    /// Adds one logger line; a multi-line message becomes several entries
    /// that share the first line's level and target.
    pub fn push_line(&mut self, line: &str) {
        let mut parts = line.split('\n');
        let first = parts.next().unwrap_or("").trim_end_matches('\r');
        let spans = parse_line(first);
        let level = spans.level;
        let target: Option<Box<str>> = spans.target.clone().map(|range| first[range].into());
        self.push_entry(first.to_string(), spans, None);
        let rest: Vec<&str> = parts.collect();
        let keep = if rest.last() == Some(&"") {
            rest.len() - 1
        } else {
            rest.len()
        };
        for part in &rest[..keep] {
            self.push_entry(
                part.trim_end_matches('\r').to_string(),
                LineSpans {
                    level,
                    ..LineSpans::default()
                },
                target.clone(),
            );
        }
    }

    fn push_entry(&mut self, text: String, spans: LineSpans, inherited_target: Option<Box<str>>) {
        let chars = if text.is_ascii() {
            text.len()
        } else {
            text.chars().count()
        };
        self.entries.push_back(Entry {
            seq: self.next_seq,
            text,
            spans,
            inherited_target,
            chars,
        });
        self.next_seq += 1;
    }

    pub fn first_seq(&self) -> u64 {
        self.entries.front().map_or(self.next_seq, |entry| entry.seq)
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get(&self, seq: u64) -> Option<&Entry> {
        let index = seq.checked_sub(self.first_seq())?;
        self.entries.get(usize::try_from(index).ok()?)
    }

    /// Drops the oldest lines until at most `limit` remain; returns how many
    /// were dropped.
    pub fn evict_to(&mut self, limit: usize) -> usize {
        let excess = self.entries.len().saturating_sub(limit);
        self.entries.drain(..excess);
        excess
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[derive(Clone, Debug)]
pub enum Search {
    Plain { needle: String, case_sensitive: bool },
    Regex(regex::Regex),
}

impl Search {
    /// `Ok(None)` for an empty query, `Err` for an invalid regex.
    pub fn new(query: &str, regex: bool, case_sensitive: bool) -> Result<Option<Self>, String> {
        if query.is_empty() {
            return Ok(None);
        }
        if regex {
            return regex::RegexBuilder::new(query)
                .case_insensitive(!case_sensitive)
                .build()
                .map(|re| Some(Search::Regex(re)))
                .map_err(|error| error.to_string());
        }
        Ok(Some(Search::Plain {
            needle: if case_sensitive {
                query.to_string()
            } else {
                query.to_ascii_lowercase()
            },
            case_sensitive,
        }))
    }

    pub fn is_match(&self, haystack: &str) -> bool {
        match self {
            Search::Plain {
                needle,
                case_sensitive: true,
            } => haystack.contains(needle.as_str()),
            Search::Plain { needle, .. } => haystack.to_ascii_lowercase().contains(needle.as_str()),
            Search::Regex(re) => re.is_match(haystack),
        }
    }

    /// Byte ranges of every match in `haystack`.
    pub fn find_ranges(&self, haystack: &str) -> Vec<Range<usize>> {
        match self {
            Search::Plain {
                needle,
                case_sensitive,
            } => {
                let lowered;
                let hay = if *case_sensitive {
                    haystack
                } else {
                    lowered = haystack.to_ascii_lowercase();
                    lowered.as_str()
                };
                hay.match_indices(needle.as_str())
                    .map(|(at, found)| at..at + found.len())
                    .collect()
            }
            Search::Regex(re) => re
                .find_iter(haystack)
                .filter(|found| !found.is_empty())
                .map(|found| found.range())
                .collect(),
        }
    }
}

/// Splits `nexium_kernel, -wgpu` into include and exclude substrings.
pub fn parse_target_filter(spec: &str) -> (Vec<String>, Vec<String>) {
    let mut include = Vec::new();
    let mut exclude = Vec::new();
    for term in spec.split(',').map(str::trim).filter(|term| !term.is_empty()) {
        match term.strip_prefix('-') {
            Some(rest) if !rest.trim().is_empty() => exclude.push(rest.trim().to_ascii_lowercase()),
            Some(_) => {}
            None => include.push(term.to_ascii_lowercase()),
        }
    }
    (include, exclude)
}

fn level_from_filter(level: log::LevelFilter) -> LogLevel {
    match level {
        log::LevelFilter::Off | log::LevelFilter::Error => LogLevel::Error,
        log::LevelFilter::Warn => LogLevel::Warn,
        log::LevelFilter::Info => LogLevel::Info,
        log::LevelFilter::Debug => LogLevel::Debug,
        log::LevelFilter::Trace => LogLevel::Trace,
    }
}

fn level_index(level: Option<Level>) -> usize {
    level.map_or(2, |level| level as usize - 1)
}

#[derive(Clone, Debug)]
pub struct ViewFilter {
    pub levels: [bool; 5],
    pub include_targets: Vec<String>,
    pub exclude_targets: Vec<String>,
    pub search: Option<Search>,
    pub only_matches: bool,
    pub collapse: bool,
}

impl Default for ViewFilter {
    fn default() -> Self {
        Self {
            levels: [true; 5],
            include_targets: Vec::new(),
            exclude_targets: Vec::new(),
            search: None,
            only_matches: false,
            collapse: false,
        }
    }
}

impl ViewFilter {
    pub fn from_settings(prefs: &LogViewerSettings, search: Option<Search>) -> Self {
        let (include_targets, exclude_targets) = parse_target_filter(&prefs.target_filter);
        Self {
            levels: [
                prefs.show_error,
                prefs.show_warn,
                prefs.show_info,
                prefs.show_debug,
                prefs.show_trace,
            ],
            include_targets,
            exclude_targets,
            search,
            only_matches: prefs.only_matches,
            collapse: prefs.collapse_repeats,
        }
    }

    pub fn accepts(&self, entry: &Entry) -> bool {
        if !self.levels[level_index(entry.level())] {
            return false;
        }
        if !self.include_targets.is_empty() || !self.exclude_targets.is_empty() {
            let target = entry.target().to_ascii_lowercase();
            if !self.include_targets.is_empty()
                && !self.include_targets.iter().any(|term| target.contains(term.as_str()))
            {
                return false;
            }
            if self.exclude_targets.iter().any(|term| target.contains(term.as_str())) {
                return false;
            }
        }
        !self.only_matches || self.search.is_none() || self.matches(entry)
    }

    pub fn matches(&self, entry: &Entry) -> bool {
        self.search.as_ref().is_some_and(|search| search.is_match(&entry.text))
    }
}

/// One visible row: a line, or a run of identical lines when collapsing.
/// `key` is the sequence number of the first line and never changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub key: u64,
    pub last: u64,
    pub count: u32,
    pub matched: bool,
}

fn suffix_text(count: u32) -> String {
    if count > 1 {
        format!("  (×{count})")
    } else {
        String::new()
    }
}

fn suffix_chars(count: u32) -> usize {
    if count > 1 {
        5 + count.to_string().len()
    } else {
        0
    }
}

/// The entry a row displays: its first line, or its newest line once the
/// first has been evicted.
pub fn row_entry<'a>(store: &'a LogStore, row: &Row) -> Option<&'a Entry> {
    store.get(row.key).or_else(|| store.get(row.last))
}

/// Filtered rows over a `LogStore`, kept sorted by key.
#[derive(Default)]
pub struct RowView {
    rows: VecDeque<Row>,
    scanned: u64,
    matches: usize,
    max_chars: usize,
    generation: u64,
}

impl RowView {
    pub fn rebuild(&mut self, store: &LogStore, filter: &ViewFilter) {
        self.rows.clear();
        self.matches = 0;
        self.max_chars = 0;
        self.scanned = store.first_seq();
        self.generation += 1;
        self.extend(store, filter, 0);
    }

    /// Adds the lines that arrived since the last call. Identical lines merge
    /// into the previous row when collapsing, but never into a row whose key
    /// is below `merge_floor` (the frozen part of a paused view).
    pub fn extend(&mut self, store: &LogStore, filter: &ViewFilter, merge_floor: u64) {
        let start = self.scanned.max(store.first_seq());
        let end = store.next_seq();
        let mut changed = false;
        for seq in start..end {
            let Some(entry) = store.get(seq) else {
                continue;
            };
            if !filter.accepts(entry) {
                continue;
            }
            changed = true;
            if filter.collapse {
                if let Some(last) = self.rows.back_mut() {
                    if last.key >= merge_floor
                        && store.get(last.last).is_some_and(|prev| prev.same_content(entry))
                    {
                        last.last = seq;
                        last.count = last.count.saturating_add(1);
                        self.max_chars = self.max_chars.max(entry.chars + suffix_chars(last.count));
                        continue;
                    }
                }
            }
            let matched = filter.matches(entry);
            self.matches += usize::from(matched);
            self.max_chars = self.max_chars.max(entry.chars);
            self.rows.push_back(Row {
                key: seq,
                last: seq,
                count: 1,
                matched,
            });
        }
        self.scanned = end;
        if changed {
            self.generation += 1;
        }
    }

    /// Removes rows whose lines have all been evicted.
    pub fn prune(&mut self, first_seq: u64) -> usize {
        let mut removed = 0;
        while self.rows.front().is_some_and(|row| row.last < first_seq) {
            if let Some(row) = self.rows.pop_front() {
                self.matches -= usize::from(row.matched);
            }
            removed += 1;
        }
        if removed > 0 {
            self.generation += 1;
        }
        removed
    }

    pub fn clear(&mut self, next_seq: u64) {
        self.rows.clear();
        self.matches = 0;
        self.max_chars = 0;
        self.scanned = next_seq;
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn get(&self, index: usize) -> Option<&Row> {
        self.rows.get(index)
    }

    pub fn match_count(&self) -> usize {
        self.matches
    }

    pub fn max_chars(&self) -> usize {
        self.max_chars
    }

    /// Index of the first row with a key at or above `key`.
    pub fn lower_bound(&self, key: u64) -> usize {
        self.rows.partition_point(|row| row.key < key)
    }

    /// Index of the first row with a key above `key`.
    pub fn upper_bound(&self, key: u64) -> usize {
        self.rows.partition_point(|row| row.key <= key)
    }

    pub fn range(&self, range: Range<usize>) -> impl Iterator<Item = &Row> {
        self.rows.range(range)
    }
}

/// A contiguous range of rows between two row keys. Keys are line sequence
/// numbers, so the range keeps naming the same lines while the view changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub anchor: u64,
    pub cursor: u64,
}

impl Selection {
    pub fn single(key: u64) -> Self {
        Self {
            anchor: key,
            cursor: key,
        }
    }

    /// Row indices covered by the selection among the first `shown` rows.
    pub fn rows(&self, view: &RowView, shown: usize) -> Range<usize> {
        let low = self.anchor.min(self.cursor);
        let high = self.anchor.max(self.cursor);
        let start = view.lower_bound(low).min(shown);
        let end = view.upper_bound(high).min(shown);
        start..end.max(start)
    }
}

/// Plain text for a set of rows: each row's original line, with a note
/// after a collapsed row saying how often it repeated.
pub fn copy_text<'a>(store: &LogStore, rows: impl IntoIterator<Item = &'a Row>) -> String {
    let mut out = String::new();
    for row in rows {
        let Some(entry) = row_entry(store, row) else {
            continue;
        };
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&entry.text);
        if row.count > 1 {
            out.push_str(&format!(
                "\n    [previous line repeated {} more times]",
                row.count - 1
            ));
        }
    }
    out
}

pub fn wrapped_lines(chars: usize, cols: usize) -> usize {
    if cols == 0 {
        1
    } else {
        chars.div_ceil(cols).max(1)
    }
}

/// Maps rows to screen lines. Without wrapping every row is one line; with
/// wrapping a row takes `ceil(chars / cols)` lines of the monospace font.
#[derive(Default)]
pub struct VisualMap {
    cols: usize,
    prefix: Vec<usize>,
    built_for: Option<(u64, usize, usize)>,
}

impl VisualMap {
    pub fn update(&mut self, store: &LogStore, view: &RowView, shown: usize, cols: usize) {
        self.cols = cols;
        if cols == 0 {
            self.prefix.clear();
            self.built_for = None;
            return;
        }
        let key = (view.generation, shown, cols);
        if self.built_for == Some(key) {
            return;
        }
        self.prefix.clear();
        self.prefix.reserve(shown + 1);
        let mut total = 0;
        self.prefix.push(0);
        for row in view.range(0..shown.min(view.len())) {
            let chars = row_entry(store, row).map_or(0, |entry| entry.chars) + suffix_chars(row.count);
            total += wrapped_lines(chars, cols);
            self.prefix.push(total);
        }
        self.built_for = Some(key);
    }

    pub fn total(&self, shown: usize) -> usize {
        if self.cols == 0 {
            shown
        } else {
            self.prefix.last().copied().unwrap_or(0)
        }
    }

    pub fn line_of_row(&self, row: usize) -> usize {
        if self.cols == 0 {
            row
        } else {
            self.prefix
                .get(row)
                .or(self.prefix.last())
                .copied()
                .unwrap_or(0)
        }
    }

    /// The row a screen line belongs to and which wrapped piece of it.
    pub fn locate(&self, line: usize) -> (usize, usize) {
        if self.cols == 0 {
            return (line, 0);
        }
        let row = self.prefix.partition_point(|&start| start <= line).saturating_sub(1);
        let start = self.prefix.get(row).copied().unwrap_or(0);
        (row, line - start)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauseReason {
    Scroll,
    Selection,
    Manual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Follow {
    Live,
    /// Only rows with a key below `frozen_seq` are shown, so nothing moves.
    Paused { reason: PauseReason, frozen_seq: u64 },
}

/// Text of the button that leaves a paused view.
fn jump_label(pending: usize) -> String {
    match pending {
        0 => "Jump to latest (End)".to_string(),
        1 => "1 new line, jump to latest (End)".to_string(),
        n => format!("{n} new lines, jump to latest (End)"),
    }
}

pub fn shown_rows(view: &RowView, follow: Follow) -> usize {
    match follow {
        Follow::Live => view.len(),
        Follow::Paused { frozen_seq, .. } => view.lower_bound(frozen_seq),
    }
}

struct TraceSwitch {
    name: &'static str,
    value: &'static str,
    about: &'static str,
}

const TRACE_SWITCHES: &[TraceSwitch] = &[
    TraceSwitch {
        name: "NEXIUM_SVC_TRACE",
        value: "2000",
        about: "Logs the first N supervisor calls with their arguments (the value is N).",
    },
    TraceSwitch {
        name: "NEXIUM_IPC_TRACE",
        value: "1",
        about: "Logs every IPC request with port, command id and caller PC. NEXIUM_IPC_TRACE_FILTER narrows it, NEXIUM_IPC_TRACE_LIMIT caps it (20000).",
    },
    TraceSwitch {
        name: "NEXIUM_FS_TRACE",
        value: "1",
        about: "Logs filesystem opens and reads. NEXIUM_FS_TRACE_FILTER narrows it to matching paths.",
    },
    TraceSwitch {
        name: "NEXIUM_HID_TRACE",
        value: "1",
        about: "Logs the controller state handed to the game.",
    },
    TraceSwitch {
        name: "NEXIUM_MODE_TRACE",
        value: "1",
        about: "Logs docked/handheld mode and display resolution queries.",
    },
    TraceSwitch {
        name: "NEXIUM_GPFIFO_TRACE",
        value: "1",
        about: "Logs GPU command FIFO submissions.",
    },
    TraceSwitch {
        name: "NEXIUM_MME_TRACE",
        value: "1",
        about: "Logs GPU macro (MME) dispatches.",
    },
    TraceSwitch {
        name: "NEXIUM_SUBMIT_TRACE",
        value: "1",
        about: "Logs Vulkan queue submissions.",
    },
    TraceSwitch {
        name: "NEXIUM_UNKNOWN_METHOD_TRACE",
        value: "1",
        about: "Periodically summarises 3D engine methods the emulator does not handle.",
    },
    TraceSwitch {
        name: "NEXIUM_VIDEO_TRACE",
        value: "1",
        about: "Logs video decoder (nvdec/vic) ioctls.",
    },
    TraceSwitch {
        name: "NEXIUM_FRAME_QUEUE_TRACE",
        value: "1",
        about: "Logs frame pacing deadline rebases.",
    },
];

struct Preset {
    label: &'static str,
    rules: &'static str,
    about: &'static str,
}

const PRESETS: &[Preset] = &[
    Preset {
        label: "Kernel/SVC detail",
        rules: "nexium_kernel::kernel=debug, nexium_kernel::kernel::svc=trace, nexium_kernel::kernel::threads=trace",
        about: "Supervisor calls, threads and scheduling.",
    },
    Preset {
        label: "IPC detail",
        rules: "nexium_ipc=trace, nexium_cmif=trace, nexium_kernel::services=debug, nexium_kernel::kernel::session=trace",
        about: "Service requests and replies.",
    },
    Preset {
        label: "GPU detail",
        rules: "nexium_nvdrv=debug, nexium_gpu=debug, nexium_shader=debug, nexium_spirv=debug",
        about: "nvdrv, the Vulkan renderer and shader translation.",
    },
    Preset {
        label: "Filesystem detail",
        rules: "nexium_kernel::fs_host=debug, nexium_kernel::services::fs=debug, nexium_kernel::services::fsp=debug, nexium_loader=debug",
        about: "Content mounting and file access.",
    },
    Preset {
        label: "Everything (trace)",
        rules: "nexium=trace",
        about: "Every NeXium crate at trace. Very verbose.",
    },
];

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-:=,+@".contains(c))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', r"'\''"))
    }
}

/// A command that starts `exe` with the given environment variables set,
/// for a POSIX shell (or fish) or for PowerShell.
pub fn launch_command(vars: &[(&str, &str)], exe: &str, windows: bool) -> String {
    let mut command = String::new();
    if windows {
        for (name, value) in vars {
            command.push_str(&format!("$env:{name}='{}'; ", value.replace('\'', "''")));
        }
        command.push_str(&format!("& '{}'", exe.replace('\'', "''")));
    } else {
        for (name, value) in vars {
            command.push_str(&format!("{name}={} ", shell_quote(value)));
        }
        command.push_str(&shell_quote(exe));
    }
    command
}

struct Palette {
    muted: Color32,
    text: Color32,
    target: Color32,
    levels: [Color32; 5],
    suffix: Color32,
    selection: Color32,
    highlight: Color32,
    highlight_text: Color32,
    cursor: Color32,
}

fn palette(dark: bool) -> Palette {
    if dark {
        Palette {
            muted: Color32::from_rgb(0x78, 0x78, 0x88),
            text: Color32::from_rgb(0xDC, 0xDC, 0xE4),
            target: Color32::from_rgb(0x86, 0xA8, 0xD8),
            levels: [
                Color32::from_rgb(0xF2, 0x5C, 0x5C),
                Color32::from_rgb(0xF5, 0xA6, 0x23),
                Color32::from_rgb(0x5C, 0xC8, 0x7A),
                Color32::from_rgb(0x9C, 0x9C, 0xB4),
                Color32::from_rgb(0x74, 0x74, 0x88),
            ],
            suffix: Color32::from_rgb(0xF5, 0xA6, 0x23),
            selection: Color32::from_rgba_unmultiplied(0x2F, 0xB4, 0xEF, 0x55),
            highlight: Color32::from_rgb(0xF5, 0xD0, 0x4A),
            highlight_text: Color32::from_rgb(0x10, 0x10, 0x14),
            cursor: Color32::from_rgb(0x5C, 0xF2, 0xFF),
        }
    } else {
        Palette {
            muted: Color32::from_rgb(0x70, 0x70, 0x80),
            text: Color32::from_rgb(0x1C, 0x1C, 0x24),
            target: Color32::from_rgb(0x2A, 0x58, 0x9A),
            levels: [
                Color32::from_rgb(0xC0, 0x1E, 0x1E),
                Color32::from_rgb(0xA8, 0x62, 0x00),
                Color32::from_rgb(0x1E, 0x82, 0x3C),
                Color32::from_rgb(0x50, 0x50, 0x68),
                Color32::from_rgb(0x80, 0x80, 0x90),
            ],
            suffix: Color32::from_rgb(0xA8, 0x62, 0x00),
            selection: Color32::from_rgba_unmultiplied(0x2F, 0xB4, 0xEF, 0x60),
            highlight: Color32::from_rgb(0xFF, 0xD8, 0x4A),
            highlight_text: Color32::from_rgb(0x10, 0x10, 0x14),
            cursor: Color32::from_rgb(0x0D, 0x6F, 0xBB),
        }
    }
}

fn byte_of_char(text: &str, chars: usize) -> usize {
    if text.is_ascii() {
        return chars.min(text.len());
    }
    text.char_indices().nth(chars).map_or(text.len(), |(at, _)| at)
}

/// Lays out bytes `piece` of a row's display text (line plus repeat
/// suffix) with level colours and search highlights.
fn row_job(
    entry: &Entry,
    display: &str,
    piece: Range<usize>,
    highlights: &[Range<usize>],
    colors: &Palette,
    font: &FontId,
) -> LayoutJob {
    let level = level_index(entry.level());
    let message_color = match level {
        0 | 1 => colors.levels[level],
        2 => colors.text,
        _ => colors.levels[level],
    };
    let ts_end = entry.spans.timestamp.as_ref().map_or(0, |r| r.end);
    let level_end = entry.spans.level_tag.as_ref().map_or(ts_end, |r| r.end);
    let message = entry.spans.message.max(level_end);
    let line_end = entry.text.len();
    let sections = [
        (0..ts_end, colors.muted),
        (ts_end..level_end, colors.levels[level]),
        (
            level_end..message,
            if entry.spans.target.is_some() {
                colors.target
            } else {
                message_color
            },
        ),
        (message..line_end, message_color),
        (line_end..display.len(), colors.suffix),
    ];
    let mut cuts = vec![piece.start, piece.end];
    for (range, _) in &sections {
        cuts.push(range.start);
        cuts.push(range.end);
    }
    for range in highlights {
        cuts.push(range.start);
        cuts.push(range.end);
    }
    cuts.retain(|cut| (piece.start..=piece.end).contains(cut) && display.is_char_boundary(*cut));
    cuts.sort_unstable();
    cuts.dedup();
    let mut job = LayoutJob::default();
    for pair in cuts.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if start >= end {
            continue;
        }
        let mut color = sections
            .iter()
            .find(|(range, _)| range.contains(&start))
            .map_or(colors.text, |(_, color)| *color);
        let highlighted = highlights.iter().any(|range| range.contains(&start));
        let background = if highlighted {
            color = colors.highlight_text;
            colors.highlight
        } else {
            Color32::TRANSPARENT
        };
        job.append(
            &display[start..end],
            0.0,
            TextFormat {
                font_id: font.clone(),
                color,
                background,
                ..Default::default()
            },
        );
    }
    job
}

pub enum LogViewerAction {
    OpenPerformance,
    Reveal(PathBuf),
}

pub struct LogViewer {
    feed: Option<LogFeed>,
    store: LogStore,
    view: RowView,
    filter: ViewFilter,
    filter_dirty: bool,
    capacity: usize,
    follow: Follow,
    selection: Option<Selection>,
    search_text: String,
    search_error: Option<String>,
    current_match: Option<(usize, usize)>,
    target_levels_text: String,
    target_levels_errors: Vec<String>,
    dropped_total: u64,
    visual: VisualMap,
    top_anchor: Option<(u64, f32)>,
    last_offset: f32,
    last_top_row: usize,
    last_inner: Rect,
    /// Height the scroll bars took from the log area last frame.
    bar_height: f32,
    scroll_to_row: Option<(usize, bool)>,
    drag_active: bool,
    /// The last drain stopped at its limit with lines still queued.
    backlog: bool,
    status: Option<(String, f64)>,
    switch_selected: Vec<bool>,
}

impl LogViewer {
    pub fn new(feed: Option<LogFeed>, prefs: &LogViewerSettings) -> Self {
        let target_levels_text = feed
            .as_ref()
            .map(|feed| file_logger::format_target_levels(&feed.filter().rules()))
            .unwrap_or_default();
        Self {
            feed,
            store: LogStore::default(),
            view: RowView::default(),
            filter: ViewFilter::from_settings(prefs, None),
            filter_dirty: true,
            capacity: prefs.ring_lines.clamp(LOG_RING_MIN, LOG_RING_MAX),
            follow: Follow::Live,
            selection: None,
            search_text: String::new(),
            search_error: None,
            current_match: None,
            target_levels_text,
            target_levels_errors: Vec::new(),
            dropped_total: 0,
            visual: VisualMap::default(),
            top_anchor: None,
            last_offset: 0.0,
            last_top_row: 0,
            last_inner: Rect::NOTHING,
            bar_height: 0.0,
            scroll_to_row: None,
            drag_active: false,
            backlog: false,
            status: None,
            switch_selected: vec![false; TRACE_SWITCHES.len()],
        }
    }

    fn log_filter(&self) -> Option<Arc<LogFilter>> {
        self.feed
            .as_ref()
            .map(|feed| Arc::clone(feed.filter()))
            .or_else(|| file_logger::installed_filter().cloned())
    }

    fn log_path(&self) -> Option<PathBuf> {
        self.feed
            .as_ref()
            .and_then(|feed| feed.log_path().map(PathBuf::from))
            .or_else(|| file_logger::current_log_path().map(PathBuf::from))
    }

    /// Moves queued lines into the ring and updates the rows. Called every
    /// frame, also while the window is closed, so the history is complete.
    pub fn ingest(&mut self) {
        if let Some(feed) = &self.feed {
            let mut taken = 0;
            while taken < MAX_DRAIN_PER_FRAME {
                let Some(line) = feed.try_recv() else {
                    break;
                };
                self.store.push_line(&line);
                taken += 1;
            }
            self.backlog = taken == MAX_DRAIN_PER_FRAME;
            self.dropped_total += feed.take_dropped();
        }
        self.settle();
    }

    #[cfg(test)]
    fn push_lines(&mut self, lines: &[&str]) {
        for line in lines {
            self.store.push_line(line);
        }
        self.settle();
    }

    fn settle(&mut self) {
        let paused = !matches!(self.follow, Follow::Live);
        let limit = if paused {
            self.capacity.saturating_mul(2)
        } else {
            self.capacity
        };
        if self.store.evict_to(limit) > 0 {
            self.view.prune(self.store.first_seq());
        }
        if self.filter_dirty {
            self.filter_dirty = false;
            self.view.rebuild(&self.store, &self.filter);
            self.current_match = None;
        } else {
            let floor = match self.follow {
                Follow::Live => 0,
                Follow::Paused { frozen_seq, .. } => frozen_seq,
            };
            self.view.extend(&self.store, &self.filter, floor);
        }
    }

    fn shown(&self) -> usize {
        shown_rows(&self.view, self.follow)
    }

    fn pause(&mut self, reason: PauseReason) {
        self.follow = match self.follow {
            Follow::Live => Follow::Paused {
                reason,
                frozen_seq: self.store.next_seq(),
            },
            Follow::Paused {
                reason: old,
                frozen_seq,
            } => Follow::Paused {
                reason: if old == PauseReason::Scroll { reason } else { old },
                frozen_seq,
            },
        };
    }

    fn resume(&mut self) {
        self.follow = Follow::Live;
        self.top_anchor = None;
        self.drag_active = false;
    }

    /// Drops the selection. A view paused only for the selection becomes a
    /// scrolled view, which follows again once it is at the bottom.
    fn clear_selection(&mut self) {
        self.selection = None;
        if let Follow::Paused {
            reason: PauseReason::Selection,
            frozen_seq,
        } = self.follow
        {
            self.follow = Follow::Paused {
                reason: PauseReason::Scroll,
                frozen_seq,
            };
        }
    }

    fn clear(&mut self) {
        self.store.clear();
        self.view.clear(self.store.next_seq());
        self.clear_selection();
        self.current_match = None;
        self.top_anchor = None;
        if let Follow::Paused { reason, .. } = self.follow {
            self.follow = Follow::Paused {
                reason,
                frozen_seq: self.store.next_seq(),
            };
        }
    }

    fn selected_range(&self) -> Option<Range<usize>> {
        let range = self.selection?.rows(&self.view, self.shown());
        (!range.is_empty()).then_some(range)
    }

    fn selection_text(&self) -> Option<String> {
        let range = self.selected_range()?;
        Some(copy_text(&self.store, self.view.range(range)))
    }

    fn set_status(&mut self, ctx: &egui::Context, message: impl Into<String>) {
        self.status = Some((message.into(), ctx.input(|i| i.time)));
    }

    fn copy(&mut self, ctx: &egui::Context, text: String, rows: usize) {
        ctx.copy_text(text);
        self.set_status(ctx, format!("Copied {rows} line(s) to the clipboard"));
    }

    fn copy_selection(&mut self, ctx: &egui::Context) {
        if let (Some(range), Some(text)) = (self.selected_range(), self.selection_text()) {
            self.copy(ctx, text, range.len());
        }
    }

    fn copy_all(&mut self, ctx: &egui::Context) {
        let text = copy_text(&self.store, self.view.range(0..self.view.len()));
        let rows = self.view.len();
        self.copy(ctx, text, rows);
    }

    fn copy_last(&mut self, ctx: &egui::Context, count: usize) {
        let start = self.view.len().saturating_sub(count);
        let text = copy_text(&self.store, self.view.range(start..self.view.len()));
        let rows = self.view.len() - start;
        self.copy(ctx, text, rows);
    }

    fn save_to_file(&mut self, ctx: &egui::Context) {
        let (range, what) = match self.selected_range() {
            Some(range) => (range, "selected"),
            None => (0..self.view.len(), "filtered"),
        };
        let mut text = copy_text(&self.store, self.view.range(range.clone()));
        text.push('\n');
        let name = format!(
            "nexium-log-{}.txt",
            chrono::Local::now().format("%Y%m%d-%H%M%S")
        );
        let mut dialog = rfd::FileDialog::new()
            .set_file_name(name.as_str())
            .add_filter("Text", &["txt", "log"]);
        if let Some(dir) = self.log_path().as_deref().and_then(|path| path.parent()) {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else {
            return;
        };
        match std::fs::write(&path, text) {
            Ok(()) => self.set_status(
                ctx,
                format!("Saved {} {what} line(s) to {}", range.len(), path.display()),
            ),
            Err(error) => self.set_status(ctx, format!("Could not save {}: {error}", path.display())),
        }
    }

    fn select_all(&mut self) {
        let shown = self.shown();
        if shown == 0 {
            return;
        }
        if let (Some(first), Some(last)) = (self.view.get(0), self.view.get(shown - 1)) {
            self.selection = Some(Selection {
                anchor: first.key,
                cursor: last.key,
            });
            self.pause(PauseReason::Selection);
        }
    }

    fn cursor_row(&self) -> Option<usize> {
        let shown = self.shown();
        let selection = self.selection?;
        (shown > 0).then(|| self.view.lower_bound(selection.cursor).min(shown - 1))
    }

    fn move_cursor(&mut self, delta: isize, extend: bool, top_row: usize) {
        let shown = self.shown();
        if shown == 0 {
            return;
        }
        let from = self.cursor_row().unwrap_or(top_row.min(shown - 1));
        let to = (from as isize + delta).clamp(0, shown as isize - 1) as usize;
        self.select_row(to, extend);
    }

    fn select_row(&mut self, row: usize, extend: bool) {
        let Some(key) = self.view.get(row).map(|row| row.key) else {
            return;
        };
        match (&mut self.selection, extend) {
            (Some(selection), true) => selection.cursor = key,
            _ => self.selection = Some(Selection::single(key)),
        }
        self.pause(PauseReason::Selection);
        self.scroll_to_row = Some((row, false));
    }

    fn step_match(&mut self, forward: bool, top_row: usize) {
        let shown = self.shown();
        if shown == 0 || self.view.match_count() == 0 {
            return;
        }
        let from = self.cursor_row().unwrap_or(top_row.min(shown - 1));
        let found = if forward {
            (from + 1..shown)
                .chain(0..=from)
                .find(|&i| self.view.get(i).is_some_and(|row| row.matched))
        } else {
            (0..from)
                .rev()
                .chain((from..shown).rev())
                .find(|&i| self.view.get(i).is_some_and(|row| row.matched))
        };
        if let Some(row) = found {
            let ordinal = self.view.range(0..row).filter(|r| r.matched).count() + 1;
            self.current_match = Some((ordinal, self.view.match_count()));
            if let Some(key) = self.view.get(row).map(|r| r.key) {
                self.selection = Some(Selection::single(key));
            }
            self.pause(PauseReason::Selection);
            self.scroll_to_row = Some((row, true));
        }
    }

    /// Applies the search box. An invalid regex keeps the last valid search
    /// so the view does not go blank while a pattern is being typed.
    fn recompile_search(&mut self, prefs: &LogViewerSettings) {
        match Search::new(&self.search_text, prefs.search_regex, prefs.search_case_sensitive) {
            Ok(search) => {
                self.search_error = None;
                self.filter.search = search;
                self.filter_dirty = true;
            }
            Err(error) => self.search_error = Some(error),
        }
    }

    fn refresh_filter(&mut self, prefs: &LogViewerSettings) {
        let search = self.filter.search.take();
        self.filter = ViewFilter::from_settings(prefs, search);
        self.filter_dirty = true;
    }

    fn apply_target_levels(&mut self, settings: &mut AppSettings) -> bool {
        let (rules, errors) = file_logger::parse_target_levels(&self.target_levels_text);
        self.target_levels_errors = errors;
        if let Some(filter) = self.log_filter() {
            filter.set_rules(rules);
        }
        if settings.log_viewer.keep_target_levels {
            settings.log_viewer.target_levels = self.target_levels_text.clone();
            return true;
        }
        false
    }

    /// Draws the window. Returns an action the app has to carry out.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        open: &mut bool,
        settings: &mut AppSettings,
    ) -> Option<LogViewerAction> {
        self.capacity = settings.log_viewer.ring_lines.clamp(LOG_RING_MIN, LOG_RING_MAX);
        let mut action = None;
        let mut save = false;
        egui::Window::new("Logs")
            .open(open)
            .default_size([980.0, 580.0])
            .min_size([560.0, 280.0])
            .show(ctx, |ui| {
                let visuals = ui.visuals_mut();
                visuals.selection.bg_fill = Color32::from_rgb(0x1B, 0x5E, 0x82);
                visuals.selection.stroke = egui::Stroke::new(1.0, Color32::WHITE);
                self.toolbar(ui, settings, &mut action, &mut save);
                self.search_bar(ui, settings, &mut save);
                self.levels_section(ui, settings, &mut save);
                self.trace_switches_section(ui);
                egui::CollapsingHeader::new("Viewer settings")
                    .id_salt("log_viewer_settings")
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Lines kept in memory");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut settings.log_viewer.ring_lines)
                                        .range(LOG_RING_MIN..=LOG_RING_MAX)
                                        .speed(500.0),
                                )
                                .on_hover_text("Older lines stay in the log file only")
                                .changed()
                            {
                                save = true;
                            }
                            ui.label(
                                egui::RichText::new(format!("default {}", LogViewerSettings::default().ring_lines))
                                    .weak(),
                            );
                        });
                    });
                ui.separator();
                self.status_row(ui);
                self.log_view(ui, &settings.log_viewer);
            });
        if save && !cfg!(test) {
            let _ = settings.save();
        }
        if *open {
            if self.backlog {
                ctx.request_repaint();
            } else {
                ctx.request_repaint_after(LIVE_POLL);
            }
        }
        action
    }

    fn toolbar(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut AppSettings,
        action: &mut Option<LogViewerAction>,
        save: &mut bool,
    ) {
        let ctx = ui.ctx().clone();
        let colors = palette(ui.visuals().dark_mode);
        ui.horizontal_wrapped(|ui| {
            let live = matches!(self.follow, Follow::Live);
            if ui
                .button(if live { "Pause" } else { "Resume" })
                .on_hover_text("Pause freezes the view; new lines keep collecting")
                .clicked()
            {
                if live {
                    self.pause(PauseReason::Manual);
                } else {
                    self.resume();
                }
            }
            if ui
                .button("Clear")
                .on_hover_text("Empties the viewer. The log file keeps everything.")
                .clicked()
            {
                self.clear();
            }
            ui.separator();
            let prefs = &mut settings.log_viewer;
            let flags = [
                &mut prefs.show_error,
                &mut prefs.show_warn,
                &mut prefs.show_info,
                &mut prefs.show_debug,
                &mut prefs.show_trace,
            ];
            let mut filter_changed = false;
            for (index, flag) in flags.into_iter().enumerate() {
                let mut text = egui::text::LayoutJob::default();
                let font = egui::TextStyle::Button.resolve(ui.style());
                text.append(
                    "● ",
                    0.0,
                    egui::TextFormat::simple(font.clone(), colors.levels[index]),
                );
                text.append(
                    LEVEL_NAMES[index],
                    0.0,
                    egui::TextFormat::simple(font, ui.visuals().strong_text_color()),
                );
                if ui
                    .toggle_value(flag, text)
                    .on_hover_text(format!("Show {} lines", LEVEL_NAMES[index].to_lowercase()))
                    .changed()
                {
                    filter_changed = true;
                }
            }
            ui.separator();
            if ui
                .checkbox(&mut prefs.wrap, "Wrap")
                .on_hover_text("Wrap long lines instead of scrolling sideways")
                .changed()
            {
                *save = true;
            }
            if ui
                .checkbox(&mut prefs.collapse_repeats, "Collapse repeats")
                .on_hover_text("Show runs of identical lines once, with a count")
                .changed()
            {
                filter_changed = true;
            }
            if filter_changed {
                *save = true;
                let prefs = prefs.clone();
                self.refresh_filter(&prefs);
            }
            ui.separator();
            let has_selection = self.selected_range().is_some();
            let last_n = settings.log_viewer.copy_last_lines;
            ui.menu_button("Copy", |ui| {
                if ui
                    .add_enabled(has_selection, egui::Button::new("Copy selection (Ctrl+C)"))
                    .clicked()
                {
                    self.copy_selection(&ctx);
                    ui.close();
                }
                if ui.button("Copy all filtered lines").clicked() {
                    self.copy_all(&ctx);
                    ui.close();
                }
                if ui.button(format!("Copy last {last_n} lines")).clicked() {
                    self.copy_last(&ctx, last_n);
                    ui.close();
                }
                ui.horizontal(|ui| {
                    ui.label("Last N:");
                    if ui
                        .add(
                            egui::DragValue::new(&mut settings.log_viewer.copy_last_lines)
                                .range(1..=LOG_RING_MAX),
                        )
                        .changed()
                    {
                        *save = true;
                    }
                });
                ui.separator();
                if ui.button("Copy log file path").clicked() {
                    match self.log_path() {
                        Some(path) => {
                            ctx.copy_text(path.display().to_string());
                            self.set_status(&ctx, format!("Copied {}", path.display()));
                        }
                        None => self.set_status(&ctx, "No log file in this session"),
                    }
                    ui.close();
                }
            });
            if ui
                .button("Save...")
                .on_hover_text("Saves the selection, or every filtered line when nothing is selected")
                .clicked()
            {
                self.save_to_file(&ctx);
            }
            if ui.button("Open logs folder").clicked() {
                let dir = self
                    .log_path()
                    .and_then(|path| path.parent().map(PathBuf::from))
                    .unwrap_or_else(|| nexium_common::paths::root().join("logs"));
                *action = Some(LogViewerAction::Reveal(dir));
            }
            if ui.button("Performance Tuning").clicked() {
                *action = Some(LogViewerAction::OpenPerformance);
            }
        });
    }

    fn search_bar(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings, save: &mut bool) {
        let top_row = self.last_top_row;
        ui.horizontal_wrapped(|ui| {
            ui.label("Search");
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.search_text)
                    .desired_width(220.0)
                    .hint_text(if settings.log_viewer.search_regex {
                        "regular expression"
                    } else {
                        "text"
                    }),
            );
            let mut recompile = response.changed();
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let back = ui.input(|i| i.modifiers.shift);
                self.step_match(!back, top_row);
                response.request_focus();
            }
            let prefs = &mut settings.log_viewer;
            if ui
                .toggle_value(&mut prefs.search_case_sensitive, "Aa")
                .on_hover_text("Match case")
                .changed()
            {
                recompile = true;
                *save = true;
            }
            if ui
                .toggle_value(&mut prefs.search_regex, ".*")
                .on_hover_text("Regular expression")
                .changed()
            {
                recompile = true;
                *save = true;
            }
            if ui
                .toggle_value(&mut prefs.only_matches, "Only matching")
                .on_hover_text("Hide lines that do not match the search")
                .changed()
            {
                *save = true;
                let prefs = prefs.clone();
                self.refresh_filter(&prefs);
            }
            if recompile {
                let prefs = settings.log_viewer.clone();
                self.recompile_search(&prefs);
            }
            if let Some(error) = &self.search_error {
                ui.colored_label(Color32::from_rgb(0xF2, 0x5C, 0x5C), "invalid regex, last valid search still applies")
                    .on_hover_text(error.as_str());
            } else if self.filter.search.is_some() {
                let count = self.view.match_count();
                let text = match self.current_match {
                    Some((at, total)) if total == count => format!("{at} of {count} matching lines"),
                    _ => format!("{count} matching lines"),
                };
                ui.label(text);
            }
            if ui.button("Prev").on_hover_text("Previous match (Shift+F3)").clicked() {
                self.step_match(false, top_row);
            }
            if ui.button("Next").on_hover_text("Next match (F3 or Enter)").clicked() {
                self.step_match(true, top_row);
            }
            ui.separator();
            ui.label("Target");
            if ui
                .add(
                    egui::TextEdit::singleline(&mut settings.log_viewer.target_filter)
                        .desired_width(200.0)
                        .hint_text("nexium_kernel, -wgpu"),
                )
                .on_hover_text("Comma separated parts of the target (module) to show; a leading - hides matches")
                .changed()
            {
                *save = true;
                let prefs = settings.log_viewer.clone();
                self.refresh_filter(&prefs);
            }
        });
    }

    fn levels_section(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings, save: &mut bool) {
        egui::CollapsingHeader::new("Log levels (apply immediately)")
            .id_salt("log_viewer_levels")
            .show(ui, |ui| {
                let filter = self.log_filter();
                ui.horizontal_wrapped(|ui| {
                    ui.label("Global level");
                    let active = filter
                        .as_ref()
                        .map_or(settings.log_level, |filter| level_from_filter(filter.default_level()));
                    let mut level = active;
                    egui::ComboBox::from_id_salt("log_viewer_global_level")
                        .selected_text(level.label())
                        .show_ui(ui, |ui| {
                            for option in LogLevel::all() {
                                ui.selectable_value(&mut level, *option, option.label());
                            }
                        });
                    if level != active {
                        settings.log_level = level;
                        file_logger::set_global_level(level.to_filter());
                        *save = true;
                    }
                    if let Some(filter) = &filter {
                        ui.label(
                            egui::RichText::new(format!(
                                "Most verbose active level: {}",
                                filter.max_level()
                            ))
                            .weak(),
                        );
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Per-target levels");
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.target_levels_text)
                            .desired_width(420.0)
                            .hint_text("nexium_kernel=debug, nexium_nvdrv=trace"),
                    );
                    let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("Apply").clicked() || enter {
                        *save |= self.apply_target_levels(settings);
                    }
                    if ui.button("Clear rules").clicked() {
                        self.target_levels_text.clear();
                        *save |= self.apply_target_levels(settings);
                    }
                });
                for error in &self.target_levels_errors {
                    ui.colored_label(Color32::from_rgb(0xF2, 0x5C, 0x5C), error.as_str());
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label("Presets");
                    for preset in PRESETS {
                        if ui
                            .button(preset.label)
                            .on_hover_text(format!("{}\n{}", preset.about, preset.rules))
                            .clicked()
                        {
                            self.target_levels_text = preset.rules.to_string();
                            *save |= self.apply_target_levels(settings);
                        }
                    }
                });
                if ui
                    .checkbox(
                        &mut settings.log_viewer.keep_target_levels,
                        "Keep per-target levels after restart",
                    )
                    .changed()
                {
                    settings.log_viewer.target_levels = if settings.log_viewer.keep_target_levels {
                        self.target_levels_text.clone()
                    } else {
                        String::new()
                    };
                    *save = true;
                }
                ui.label(
                    egui::RichText::new(
                        "Rules match the start of a target (module path); the longest match wins. \
                         They change both the log file and this view. Debug and trace on busy \
                         targets write a lot and can slow games down.",
                    )
                    .weak(),
                );
            });
    }

    fn trace_switches_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Trace switches (apply after restart)")
            .id_salt("log_viewer_trace_switches")
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "These are read once when NeXium starts, so changing them here does \
                         nothing to this session. Tick the ones you want and copy a launch \
                         command that sets them.",
                    )
                    .weak(),
                );
                egui::Grid::new("log_viewer_switch_grid")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        for (index, switch) in TRACE_SWITCHES.iter().enumerate() {
                            ui.checkbox(
                                &mut self.switch_selected[index],
                                format!("{}={}", switch.name, switch.value),
                            );
                            ui.horizontal(|ui| {
                                ui.label(switch.about);
                                if std::env::var_os(switch.name).is_some() {
                                    ui.colored_label(
                                        Color32::from_rgb(0x3C, 0xD4, 0x5C),
                                        "(set in this session)",
                                    );
                                }
                            });
                            ui.end_row();
                        }
                    });
                let picked: Vec<(&str, &str)> = TRACE_SWITCHES
                    .iter()
                    .zip(&self.switch_selected)
                    .filter(|(_, on)| **on)
                    .map(|(switch, _)| (switch.name, switch.value))
                    .collect();
                if ui
                    .add_enabled(!picked.is_empty(), egui::Button::new("Copy launch command"))
                    .clicked()
                {
                    let exe = std::env::current_exe()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|_| "nexium".to_string());
                    let command = launch_command(&picked, &exe, cfg!(windows));
                    ui.ctx().copy_text(command.clone());
                    self.set_status(ui.ctx(), format!("Copied: {command}"));
                }
            });
    }

    fn status_row(&mut self, ui: &mut egui::Ui) {
        let shown = self.shown();
        let pending = self.view.len() - shown;
        let now = ui.input(|i| i.time);
        ui.horizontal_wrapped(|ui| {
            match self.follow {
                Follow::Live => {
                    ui.colored_label(Color32::from_rgb(0x3C, 0xD4, 0x5C), "Following new lines");
                }
                Follow::Paused { reason, .. } => {
                    let why = match reason {
                        PauseReason::Scroll => "Paused (scrolled up)",
                        PauseReason::Selection => "Paused (selection)",
                        PauseReason::Manual => "Paused",
                    };
                    ui.colored_label(Color32::from_rgb(0xF5, 0xA6, 0x23), why);
                    if ui.button(jump_label(pending)).clicked() {
                        self.resume();
                    }
                }
            }
            ui.separator();
            ui.label(format!(
                "{} rows shown, {} lines in memory",
                shown,
                self.store.len()
            ));
            if let Some(range) = self.selected_range() {
                ui.label(format!("{} selected", range.len()));
            }
            if self.dropped_total > 0 {
                ui.colored_label(
                    Color32::from_rgb(0xF5, 0xA6, 0x23),
                    format!("{} lines skipped by the viewer (still in the log file)", self.dropped_total),
                );
            }
            if let Some((message, at)) = &self.status {
                if now - at < 4.0 {
                    ui.label(egui::RichText::new(message.as_str()).weak());
                }
            }
        });
        if self.status.as_ref().is_some_and(|(_, at)| now - at < 4.0) {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        }
    }

    fn log_view(&mut self, ui: &mut egui::Ui, prefs: &LogViewerSettings) {
        let ctx = ui.ctx().clone();
        let colors = palette(ui.visuals().dark_mode);
        let font = FontId::monospace(12.0);
        let (glyph_h, char_w) = ctx.fonts_mut(|f| (f.row_height(&font), f.glyph_width(&font, 'M')));
        let row_h = (glyph_h + 2.0).round();
        let shown = self.shown();
        let cols = if prefs.wrap {
            ((ui.available_width() - 24.0) / char_w).floor().max(20.0) as usize
        } else {
            0
        };
        self.visual.update(&self.store, &self.view, shown, cols);
        let total_lines = self.visual.total(shown);
        let content_h = total_lines as f32 * row_h;
        let available_h = ui.available_height();
        let viewport_h = (available_h - self.bar_height).max(1.0);
        let max_offset = (content_h - viewport_h).max(0.0);
        let user_scroll = ctx.input(|i| i.smooth_scroll_delta.y != 0.0 || i.pointer.primary_down());

        let mut forced = match self.follow {
            Follow::Live => Some(max_offset),
            Follow::Paused { .. } => self.top_anchor.map(|(key, delta)| {
                let row = self.view.lower_bound(key).min(shown);
                let exact = self.view.get(row).is_some_and(|r| r.key == key);
                self.visual.line_of_row(row) as f32 * row_h + if exact { delta } else { 0.0 }
            }),
        };
        if let Some((row, center)) = self.scroll_to_row.take() {
            let top = self.visual.line_of_row(row) as f32 * row_h;
            let bottom = self.visual.line_of_row(row + 1) as f32 * row_h;
            let current = forced.unwrap_or(self.last_offset);
            forced = Some(if center {
                top - (viewport_h - row_h) / 2.0
            } else if top < current {
                top
            } else if bottom > current + viewport_h {
                bottom - viewport_h
            } else {
                current
            });
        }
        if self.drag_active {
            let current = forced.unwrap_or(self.last_offset);
            let mut target = current;
            if let Some(pointer) = ctx.input(|i| i.pointer.interact_pos()) {
                let inner = self.last_inner;
                if pointer.y < inner.top() {
                    target -= (inner.top() - pointer.y).min(120.0) * 0.5 + 2.0;
                } else if pointer.y > inner.bottom() {
                    target += (pointer.y - inner.bottom()).min(120.0) * 0.5 + 2.0;
                }
            }
            target -= ctx.input(|i| i.smooth_scroll_delta.y);
            if target != current {
                forced = Some(target);
                ctx.request_repaint();
            }
        }
        let forced = forced.map(|offset| offset.clamp(0.0, max_offset));

        let mut area = if prefs.wrap {
            egui::ScrollArea::vertical()
        } else {
            egui::ScrollArea::both()
        }
        .id_salt("log_viewer_scroll")
        .auto_shrink([false; 2])
        .content_margin(egui::Margin::ZERO)
        .scroll_source(egui::scroll_area::ScrollSource::SCROLL_BAR | egui::scroll_area::ScrollSource::MOUSE_WHEEL);
        if let Some(offset) = forced {
            area = area.vertical_scroll_offset(offset);
        }
        let view_id = ui.id().with("log_viewer_rows");
        let selected = self.selected_range();
        let cursor_row = self.cursor_row();
        let content_w = if prefs.wrap {
            0.0
        } else {
            self.view.max_chars() as f32 * char_w + 16.0
        };
        let search = self.filter.search.clone();
        let output = area.show_viewport(ui, |ui, viewport| {
            ui.set_height(content_h);
            ui.set_min_width(content_w);
            let origin = ui.max_rect().min;
            let width = ui.max_rect().width().max(content_w).max(viewport.width());
            let full = Rect::from_min_size(origin, egui::vec2(width, content_h.max(viewport.height())));
            let response = ui.interact(full, view_id, Sense::click_and_drag());
            if total_lines == 0 {
                ui.painter().text(
                    origin + egui::vec2(8.0, 6.0),
                    egui::Align2::LEFT_TOP,
                    if self.store.len() == 0 {
                        "No log lines yet."
                    } else {
                        "No lines match the current filters."
                    },
                    FontId::proportional(13.0),
                    colors.muted,
                );
                return (response, origin);
            }
            let first = (viewport.min.y / row_h).floor().max(0.0) as usize;
            let last = ((viewport.max.y / row_h).ceil().max(0.0) as usize).min(total_lines);
            let mut jobs = Vec::with_capacity(last.saturating_sub(first));
            for line in first..last {
                let (row_index, piece) = self.visual.locate(line);
                let Some(row) = self.view.get(row_index) else {
                    continue;
                };
                let Some(entry) = row_entry(&self.store, row) else {
                    continue;
                };
                let display = format!("{}{}", entry.text, suffix_text(row.count));
                let bytes = if cols == 0 {
                    0..display.len()
                } else {
                    byte_of_char(&display, piece * cols)..byte_of_char(&display, (piece + 1) * cols)
                };
                let highlights = search
                    .as_ref()
                    .map(|search| search.find_ranges(&entry.text))
                    .unwrap_or_default();
                let job = row_job(entry, &display, bytes, &highlights, &colors, &font);
                jobs.push((line, row_index, job));
            }
            let galleys: Vec<_> = ctx.fonts_mut(|f| {
                jobs.into_iter()
                    .map(|(line, row, job)| (line, row, f.layout_job(job)))
                    .collect()
            });
            let painter = ui.painter();
            for (line, row, galley) in galleys {
                let y = origin.y + line as f32 * row_h;
                let line_rect = Rect::from_min_size(egui::pos2(origin.x, y), egui::vec2(width, row_h));
                if selected.as_ref().is_some_and(|range| range.contains(&row)) {
                    painter.rect_filled(line_rect, 0.0, colors.selection);
                }
                if cursor_row == Some(row) && response.has_focus() {
                    painter.rect_stroke(
                        line_rect.shrink(0.5),
                        0.0,
                        egui::Stroke::new(1.0, colors.cursor),
                        egui::StrokeKind::Inside,
                    );
                }
                painter.galley(egui::pos2(origin.x + 4.0, y + 1.0), galley, colors.text);
            }
            (response, origin)
        });

        let (response, origin) = output.inner;
        let offset = output.state.offset.y;
        let actual_max = (content_h - output.inner_rect.height()).max(0.0);
        let at_bottom = offset >= actual_max - 1.0;
        self.last_inner = output.inner_rect;
        self.bar_height = (available_h - output.inner_rect.height()).max(0.0);
        self.last_offset = offset;

        match self.follow {
            Follow::Live => {
                if user_scroll && !at_bottom && forced.is_some_and(|f| offset < f - 1.0) {
                    self.pause(PauseReason::Scroll);
                }
            }
            Follow::Paused {
                reason: PauseReason::Scroll,
                ..
            } => {
                if at_bottom {
                    self.resume();
                }
            }
            Follow::Paused { .. } => {}
        }

        if matches!(self.follow, Follow::Paused { .. }) {
            let label = jump_label(self.view.len() - self.shown());
            let label = format!("Paused, {}{}", label[..1].to_lowercase(), &label[1..]);
            let button_font = egui::TextStyle::Button.resolve(ui.style());
            let text_w = ctx.fonts_mut(|f| f.layout_no_wrap(label.clone(), button_font, Color32::WHITE).size().x);
            let text = egui::RichText::new(label)
                .color(Color32::from_rgb(0x10, 0x10, 0x14))
                .strong();
            let button = egui::Button::new(text).fill(Color32::from_rgb(0xF5, 0xA6, 0x23));
            let inner = output.inner_rect;
            let rect = Rect::from_center_size(
                egui::pos2(inner.center().x, inner.bottom() - 18.0),
                egui::vec2((text_w + 24.0).min(inner.width()), 26.0),
            );
            if ui.put(rect, button).clicked() {
                self.resume();
            }
        }

        let row_at = |visual: &VisualMap, y: f32| -> Option<usize> {
            if total_lines == 0 {
                return None;
            }
            let line = ((y - origin.y) / row_h).floor().clamp(0.0, (total_lines - 1) as f32) as usize;
            Some(visual.locate(line).0)
        };
        let top_row = row_at(&self.visual, output.inner_rect.top()).unwrap_or(0);
        self.last_top_row = top_row;
        let page = ((output.inner_rect.height() / row_h).floor() as isize - 1).max(1);

        let (primary_pressed, secondary_pressed, primary_down, shift, pointer) = ctx.input(|i| {
            (
                i.pointer.primary_pressed(),
                i.pointer.secondary_pressed(),
                i.pointer.primary_down(),
                i.modifiers.shift,
                i.pointer.interact_pos(),
            )
        });
        if response.hovered() && (primary_pressed || secondary_pressed) {
            response.request_focus();
            if let Some(pointer) = pointer {
                let below_content = pointer.y > origin.y + content_h;
                match row_at(&self.visual, pointer.y) {
                    Some(row) if !below_content => {
                        let inside = selected.as_ref().is_some_and(|range| range.contains(&row));
                        if primary_pressed {
                            self.select_row(row, shift && self.selection.is_some());
                            self.scroll_to_row = None;
                            self.drag_active = true;
                        } else if !inside {
                            self.select_row(row, false);
                            self.scroll_to_row = None;
                        }
                    }
                    _ if primary_pressed => self.clear_selection(),
                    _ => {}
                }
            }
        } else if self.drag_active {
            if primary_down {
                if let (Some(pointer), Some(selection)) = (pointer, self.selection.as_mut()) {
                    if let Some(key) = row_at(&self.visual, pointer.y).and_then(|row| self.view.get(row)).map(|r| r.key) {
                        selection.cursor = key;
                    }
                }
            } else {
                self.drag_active = false;
            }
        }

        response.context_menu(|ui| {
            if ui
                .add_enabled(self.selected_range().is_some(), egui::Button::new("Copy selection"))
                .clicked()
            {
                self.copy_selection(&ctx);
                ui.close();
            }
            if ui.button("Select all (Ctrl+A)").clicked() {
                self.select_all();
                ui.close();
            }
            if ui.button("Clear selection (Esc)").clicked() {
                self.clear_selection();
                ui.close();
            }
            if ui.button("Jump to latest (End)").clicked() {
                self.resume();
                ui.close();
            }
        });

        if response.has_focus() {
            ui.memory_mut(|memory| {
                memory.set_focus_lock_filter(
                    view_id,
                    egui::EventFilter {
                        tab: false,
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        escape: true,
                    },
                )
            });
            let events = ctx.input(|i| i.events.clone());
            for event in events {
                match event {
                    egui::Event::Copy => self.copy_selection(&ctx),
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => match key {
                        egui::Key::A if modifiers.command => self.select_all(),
                        egui::Key::ArrowUp => self.move_cursor(-1, modifiers.shift, top_row),
                        egui::Key::ArrowDown => self.move_cursor(1, modifiers.shift, top_row),
                        egui::Key::PageUp => self.move_cursor(-page, modifiers.shift, top_row),
                        egui::Key::PageDown => self.move_cursor(page, modifiers.shift, top_row),
                        egui::Key::Home => self.move_cursor(isize::MIN / 2, modifiers.shift, top_row),
                        egui::Key::End if modifiers.shift => {
                            self.move_cursor(isize::MAX / 2, true, top_row)
                        }
                        egui::Key::End => self.resume(),
                        egui::Key::Escape => self.clear_selection(),
                        egui::Key::F3 => self.step_match(!modifiers.shift, top_row),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }

        self.top_anchor = if total_lines == 0 {
            None
        } else {
            let line = ((offset / row_h).floor().max(0.0) as usize).min(total_lines - 1);
            let row = self.visual.locate(line).0;
            self.view.get(row).map(|r| {
                (
                    r.key,
                    offset - self.visual.line_of_row(row) as f32 * row_h,
                )
            })
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: &str = "[12:00:01.250] [ WARN] [nexium_kernel::kernel::svc] svcBreak reason=0x1";

    fn store_with(lines: &[&str]) -> LogStore {
        let mut store = LogStore::default();
        for line in lines {
            store.push_line(line);
        }
        store
    }

    fn texts(store: &LogStore, view: &RowView) -> Vec<String> {
        view.range(0..view.len())
            .map(|row| row_entry(store, row).unwrap().message().to_string())
            .collect()
    }

    #[test]
    fn parses_timestamp_level_target_and_message() {
        let spans = parse_line(LINE);
        assert_eq!(spans.level, Some(Level::Warn));
        assert_eq!(&LINE[spans.timestamp.clone().unwrap()], "[12:00:01.250]");
        assert_eq!(&LINE[spans.level_tag.clone().unwrap()], "[ WARN]");
        assert_eq!(&LINE[spans.target.clone().unwrap()], "nexium_kernel::kernel::svc");
        assert_eq!(&LINE[spans.message..], "svcBreak reason=0x1");
    }

    #[test]
    fn parses_older_formats_without_inventing_targets() {
        let legacy = "[ INFO] [draw-drop] skipped draw";
        let spans = parse_line(legacy);
        assert_eq!(spans.level, Some(Level::Info));
        assert!(spans.timestamp.is_none());
        assert!(spans.target.is_none());
        assert_eq!(&legacy[spans.message..], "[draw-drop] skipped draw");

        let file = "[09:15:02.001] [ERROR] lost device";
        let spans = parse_line(file);
        assert_eq!(spans.level, Some(Level::Error));
        assert_eq!(&file[spans.message..], "lost device");

        let plain = "no brackets [ INFO] here";
        let spans = parse_line(plain);
        assert_eq!(spans.level, None);
        assert_eq!(spans.message, 0);

        let target_then_bracket = "[ INFO] [nexium_nvdrv] [draw-drop] x";
        let spans = parse_line(target_then_bracket);
        assert_eq!(&target_then_bracket[spans.target.clone().unwrap()], "nexium_nvdrv");
        assert_eq!(&target_then_bracket[spans.message..], "[draw-drop] x");
    }

    #[test]
    fn multi_line_records_split_and_inherit_level_and_target() {
        let store = store_with(&["[ ERROR] [nexium_gpu] first\nsecond\r\nthird\n"]);
        assert_eq!(store.len(), 3);
        let second = store.get(1).unwrap();
        assert_eq!(second.text, "second");
        assert_eq!(second.level(), Some(Level::Error));
        assert_eq!(second.target(), "nexium_gpu");
        assert_eq!(store.get(2).unwrap().text, "third");
    }

    #[test]
    fn store_keeps_sequence_numbers_across_eviction() {
        let mut store = store_with(&["a", "b", "c", "d"]);
        assert_eq!(store.evict_to(2), 2);
        assert_eq!(store.first_seq(), 2);
        assert_eq!(store.get(2).unwrap().text, "c");
        assert!(store.get(1).is_none());
        store.clear();
        store.push_line("e");
        assert_eq!(store.first_seq(), 4);
    }

    #[test]
    fn filter_by_level_target_and_search() {
        let store = store_with(&[
            "[ INFO] [nexium_kernel::svc] a",
            "[DEBUG] [nexium_nvdrv::gpu] b",
            "[ WARN] [wgpu_core] c",
            "untagged d",
        ]);
        let mut filter = ViewFilter::default();
        filter.levels[3] = false;
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        assert_eq!(texts(&store, &view), ["a", "c", "untagged d"]);

        filter.levels = [true; 5];
        let (include, exclude) = parse_target_filter("nexium, -nvdrv");
        filter.include_targets = include;
        filter.exclude_targets = exclude;
        view.rebuild(&store, &filter);
        assert_eq!(texts(&store, &view), ["a"]);

        filter.include_targets.clear();
        filter.exclude_targets.clear();
        filter.search = Search::new("WGPU", false, false).unwrap();
        filter.only_matches = true;
        view.rebuild(&store, &filter);
        assert_eq!(texts(&store, &view), ["c"]);
        assert_eq!(view.match_count(), 1);
    }

    #[test]
    fn only_matching_without_a_search_hides_nothing() {
        let store = store_with(&["[ INFO] a", "[ WARN] b"]);
        let filter = ViewFilter {
            only_matches: true,
            ..ViewFilter::default()
        };
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        assert_eq!(texts(&store, &view), ["a", "b"]);
    }

    #[test]
    fn search_ranges_plain_case_and_regex() {
        let plain = Search::new("draw", false, false).unwrap().unwrap();
        assert_eq!(plain.find_ranges("Draw a draw"), vec![0..4, 7..11]);
        let exact = Search::new("draw", false, true).unwrap().unwrap();
        assert_eq!(exact.find_ranges("Draw a draw"), vec![7..11]);
        let regex = Search::new(r"0x[0-9a-f]+", true, false).unwrap().unwrap();
        assert_eq!(regex.find_ranges("addr=0x1F end"), vec![5..9]);
        assert!(Search::new("(", true, false).is_err());
        assert!(Search::new("", true, false).unwrap().is_none());
    }

    #[test]
    fn collapse_merges_identical_consecutive_lines() {
        let store = store_with(&[
            "[00:00:00.001] [ WARN] [nexium_nvdrv] [draw-drop] x",
            "[00:00:00.002] [ WARN] [nexium_nvdrv] [draw-drop] x",
            "[00:00:00.003] [ WARN] [nexium_nvdrv] [draw-drop] x",
            "[00:00:00.004] [ INFO] [nexium_nvdrv] other",
            "[00:00:00.005] [ WARN] [nexium_nvdrv] [draw-drop] x",
        ]);
        let filter = ViewFilter {
            collapse: true,
            ..ViewFilter::default()
        };
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        let counts: Vec<u32> = view.range(0..view.len()).map(|row| row.count).collect();
        assert_eq!(counts, [3, 1, 1]);
        assert_eq!(view.get(0).unwrap().key, 0);
        assert_eq!(view.get(0).unwrap().last, 2);
    }

    #[test]
    fn collapse_does_not_merge_into_frozen_rows() {
        let mut store = store_with(&["[ WARN] same", "[ WARN] same"]);
        let filter = ViewFilter {
            collapse: true,
            ..ViewFilter::default()
        };
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        assert_eq!(view.len(), 1);
        let frozen = store.next_seq();
        store.push_line("[ WARN] same");
        view.extend(&store, &filter, frozen);
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0).unwrap().count, 2);
        store.push_line("[ WARN] same");
        view.extend(&store, &filter, frozen);
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(1).unwrap().count, 2);
    }

    #[test]
    fn selection_maps_to_filtered_rows_and_survives_filter_changes() {
        let store = store_with(&[
            "[ INFO] a0", "[DEBUG] d1", "[ INFO] a2", "[DEBUG] d3", "[ INFO] a4", "[ INFO] a5",
        ]);
        let mut filter = ViewFilter::default();
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        let selection = Selection {
            anchor: 4,
            cursor: 1,
        };
        assert_eq!(selection.rows(&view, view.len()), 1..5);
        let selected: Vec<_> = view.range(1..5).map(|row| row.key).collect();
        assert_eq!(selected, [1, 2, 3, 4]);

        filter.levels[3] = false;
        view.rebuild(&store, &filter);
        let range = selection.rows(&view, view.len());
        let keys: Vec<_> = view.range(range).map(|row| row.key).collect();
        assert_eq!(keys, [2, 4]);
        assert_eq!(
            copy_text(&store, view.range(selection.rows(&view, view.len()))),
            "[ INFO] a2\n[ INFO] a4"
        );
    }

    #[test]
    fn selection_and_view_hold_still_while_lines_arrive() {
        let (tx, feed) = LogFeed::detached(64);
        let mut viewer = LogViewer::new(Some(feed), &LogViewerSettings::default());
        for n in 0..10 {
            tx.send(format!("[ INFO] line {n}")).unwrap();
        }
        viewer.ingest();
        assert_eq!(viewer.shown(), 10);
        viewer.select_row(3, false);
        viewer.select_row(6, true);
        assert!(matches!(viewer.follow, Follow::Paused { reason: PauseReason::Selection, .. }));
        let before = viewer.selection_text().unwrap();
        for n in 10..40 {
            tx.send(format!("[ INFO] line {n}")).unwrap();
        }
        viewer.ingest();
        assert_eq!(viewer.shown(), 10);
        assert_eq!(viewer.view.len() - viewer.shown(), 30);
        assert_eq!(viewer.selected_range(), Some(3..7));
        assert_eq!(viewer.selection_text().unwrap(), before);
        assert_eq!(before, "[ INFO] line 3\n[ INFO] line 4\n[ INFO] line 5\n[ INFO] line 6");
        viewer.resume();
        assert_eq!(viewer.shown(), 40);
        assert_eq!(viewer.selected_range(), Some(3..7));
    }

    #[test]
    fn paused_view_defers_eviction_and_live_view_trims() {
        let prefs = LogViewerSettings {
            ring_lines: LOG_RING_MIN,
            ..LogViewerSettings::default()
        };
        let mut viewer = LogViewer::new(None, &prefs);
        let lines: Vec<String> = (0..LOG_RING_MIN).map(|n| format!("[ INFO] {n}")).collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        viewer.push_lines(&refs);
        viewer.pause(PauseReason::Manual);
        viewer.push_lines(&refs[..10]);
        assert_eq!(viewer.store.len(), LOG_RING_MIN + 10);
        assert_eq!(viewer.store.first_seq(), 0);
        viewer.resume();
        viewer.push_lines(&[]);
        assert_eq!(viewer.store.len(), LOG_RING_MIN);
        assert_eq!(viewer.view.get(0).unwrap().key, 10);
    }

    #[test]
    fn selection_clamps_when_its_lines_are_evicted() {
        let mut store = store_with(&["a", "b", "c", "d", "e"]);
        let filter = ViewFilter::default();
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        let selection = Selection {
            anchor: 1,
            cursor: 3,
        };
        store.evict_to(3);
        assert_eq!(view.prune(store.first_seq()), 2);
        let keys: Vec<_> = view
            .range(selection.rows(&view, view.len()))
            .map(|row| row.key)
            .collect();
        assert_eq!(keys, [2, 3]);
    }

    #[test]
    fn copy_text_keeps_exact_lines_and_notes_repeats() {
        let store = store_with(&[LINE, LINE, "[ INFO] [nexium_gpu] done"]);
        let filter = ViewFilter {
            collapse: true,
            ..ViewFilter::default()
        };
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        assert_eq!(
            copy_text(&store, view.range(0..view.len())),
            format!("{LINE}\n    [previous line repeated 1 more times]\n[ INFO] [nexium_gpu] done")
        );
        let plain = ViewFilter::default();
        view.rebuild(&store, &plain);
        assert_eq!(
            copy_text(&store, view.range(0..2)),
            format!("{LINE}\n{LINE}")
        );
    }

    #[test]
    fn wrapped_visual_lines_map_back_to_rows() {
        let store = store_with(&["0123456789", "abc", "0123456789012345678901"]);
        let filter = ViewFilter::default();
        let mut view = RowView::default();
        view.rebuild(&store, &filter);
        let mut map = VisualMap::default();
        map.update(&store, &view, view.len(), 10);
        assert_eq!(map.total(view.len()), 1 + 1 + 3);
        assert_eq!(map.locate(0), (0, 0));
        assert_eq!(map.locate(1), (1, 0));
        assert_eq!(map.locate(4), (2, 2));
        assert_eq!(map.line_of_row(2), 2);
        map.update(&store, &view, view.len(), 0);
        assert_eq!(map.total(3), 3);
        assert_eq!(map.locate(2), (2, 0));
        assert_eq!(wrapped_lines(0, 8), 1);
        assert_eq!(wrapped_lines(17, 8), 3);
    }

    #[test]
    fn launch_command_quotes_for_each_shell() {
        let vars = [("NEXIUM_IPC_TRACE", "1"), ("NEXIUM_FS_TRACE_FILTER", "a b")];
        assert_eq!(
            launch_command(&vars, "/opt/nexium/nexium", false),
            "NEXIUM_IPC_TRACE=1 NEXIUM_FS_TRACE_FILTER='a b' /opt/nexium/nexium"
        );
        assert_eq!(
            launch_command(&vars[..1], r"C:\Program Files\NeXium\nexium.exe", true),
            r"$env:NEXIUM_IPC_TRACE='1'; & 'C:\Program Files\NeXium\nexium.exe'"
        );
    }

    #[test]
    fn row_job_colours_sections_and_highlights_inside_a_piece() {
        let mut store = LogStore::default();
        store.push_line(LINE);
        let entry = store.get(0).unwrap();
        let colors = palette(true);
        let font = FontId::monospace(12.0);
        let highlights = Search::new("svcbreak", false, false)
            .unwrap()
            .unwrap()
            .find_ranges(&entry.text);
        let job = row_job(entry, &entry.text, 0..entry.text.len(), &highlights, &colors, &font);
        assert_eq!(job.text, entry.text);
        assert!(job
            .sections
            .iter()
            .any(|section| section.format.background == colors.highlight
                && &job.text[section.byte_range.start.0..section.byte_range.end.0] == "svcBreak"));
        let piece = row_job(entry, &entry.text, 5..20, &[], &colors, &font);
        assert_eq!(piece.text, &entry.text[5..20]);
    }

    /// Drives the real window through egui frames, fed by a `FileLogger`.
    mod harness {
        use super::*;
        use egui::{Event, Modifiers, Pos2, RawInput, Vec2, ViewportId};
        use log::Log;
        use nexium_common::FileLogger;
        use std::time::Duration;

        const SCREEN: Vec2 = Vec2::new(1280.0, 800.0);

        struct Harness {
            ctx: egui::Context,
            viewer: LogViewer,
            settings: AppSettings,
            logger: FileLogger,
            path: PathBuf,
            open: bool,
            time: f64,
            output: Option<egui::FullOutput>,
            next: usize,
        }

        impl Drop for Harness {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.path);
            }
        }

        impl Harness {
            fn new() -> Self {
                static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let path = std::env::temp_dir().join(format!(
                    "nexium-log-viewer-{}-{}.log",
                    std::process::id(),
                    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                ));
                let (logger, feed) = FileLogger::with_path(&path, 4096).unwrap();
                let settings = AppSettings::default();
                let viewer = LogViewer::new(Some(feed), &settings.log_viewer);
                let mut harness = Self {
                    ctx: egui::Context::default(),
                    viewer,
                    settings,
                    logger,
                    path,
                    open: true,
                    time: 0.0,
                    output: None,
                    next: 0,
                };
                for n in 0..400 {
                    harness.log(Level::Info, "nexium_gui::boot", &format!("filler {n}"));
                }
                harness.frames(5);
                harness
            }

            fn frame(&mut self, events: Vec<Event>) {
                self.time += 1.0 / 60.0;
                let input = RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, SCREEN)),
                    time: Some(self.time),
                    predicted_dt: 1.0 / 60.0,
                    events,
                    ..Default::default()
                };
                let ctx = self.ctx.clone();
                let mut output = ctx.run_ui(input, |ui| {
                    self.viewer.ingest();
                    let ctx = ui.ctx().clone();
                    if self.open {
                        let _ = self.viewer.show(&ctx, &mut self.open, &mut self.settings);
                    }
                });
                output.textures_delta.clear();
                self.output = Some(output);
            }

            fn frames(&mut self, count: usize) {
                for _ in 0..count {
                    self.frame(Vec::new());
                }
            }

            fn log(&self, level: Level, target: &str, message: &str) {
                self.logger.log(
                    &log::Record::builder()
                        .level(level)
                        .target(target)
                        .args(format_args!("{message}"))
                        .build(),
                );
            }

            /// Logs a uniquely numbered line and returns its message.
            fn probe(&mut self, level: Level, target: &str) -> String {
                self.next += 1;
                let message = format!("probe-{} {}", self.next, level.as_str().to_lowercase());
                self.log(level, target, &message);
                message
            }

            fn texts(&self) -> Vec<(String, Rect, Rect)> {
                fn walk(shape: &egui::Shape, clip: Rect, out: &mut Vec<(String, Rect, Rect)>) {
                    match shape {
                        egui::Shape::Vec(shapes) => {
                            for shape in shapes {
                                walk(shape, clip, out);
                            }
                        }
                        egui::Shape::Text(text) => {
                            let rect = text.galley.rect.translate(text.pos.to_vec2());
                            out.push((text.galley.text().to_string(), rect, clip));
                        }
                        _ => {}
                    }
                }
                let mut out = Vec::new();
                for clipped in &self.output.as_ref().unwrap().shapes {
                    walk(&clipped.shape, clipped.clip_rect, &mut out);
                }
                out
            }

            /// Log rows drawn inside the visible part of the log area, top to bottom.
            fn visible_rows(&self) -> Vec<String> {
                let area = self.viewer.last_inner;
                let mut rows: Vec<(f32, String)> = self
                    .texts()
                    .into_iter()
                    .filter(|(text, rect, clip)| {
                        text.starts_with('[')
                            && clip.intersects(*rect)
                            && area.contains_rect(clip.intersect(*rect).shrink(0.5))
                    })
                    .map(|(text, rect, _)| (rect.top(), text))
                    .collect();
                rows.sort_by(|a, b| a.0.total_cmp(&b.0));
                rows.into_iter().map(|(_, text)| text).collect()
            }

            fn find(&self, label: &str) -> Pos2 {
                self.texts()
                    .into_iter()
                    .find(|(text, rect, clip)| text == label && clip.intersects(*rect))
                    .map(|(_, rect, _)| rect.center())
                    .unwrap_or_else(|| panic!("no '{label}' on screen"))
            }

            fn has_text(&self, needle: &str) -> bool {
                self.texts().iter().any(|(text, rect, clip)| text.contains(needle) && clip.intersects(*rect))
            }

            fn click_at(&mut self, pos: Pos2) {
                self.frame(vec![
                    Event::PointerMoved(pos),
                    Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: Modifiers::NONE,
                    },
                ]);
                self.frame(vec![Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Modifiers::NONE,
                }]);
                self.frame(vec![Event::PointerMoved(Pos2::new(SCREEN.x - 4.0, SCREEN.y - 4.0))]);
            }

            fn click(&mut self, label: &str) {
                let pos = self.find(label);
                self.click_at(pos);
            }

            fn type_into(&mut self, label: &str, text: &str) {
                self.click(label);
                self.key(egui::Key::End);
                self.frame(vec![Event::Text(text.to_string())]);
                self.frames(2);
            }

            fn key(&mut self, key: egui::Key) {
                self.frame(vec![Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Modifiers::NONE,
                }]);
                self.frame(vec![Event::Key {
                    key,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Modifiers::NONE,
                }]);
            }

            fn wheel(&mut self, delta_y: f32) {
                let pos = self.viewer.last_inner.center();
                self.frame(vec![
                    Event::PointerMoved(pos),
                    Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: Vec2::new(0.0, delta_y),
                        phase: egui::TouchPhase::Move,
                        modifiers: Modifiers::NONE,
                    },
                ]);
                self.frames(30);
            }

            fn repaint_delay(&self) -> Duration {
                self.output.as_ref().unwrap().viewport_output[&ViewportId::ROOT].repaint_delay
            }

            fn describe(&self) -> String {
                let rows = self.visible_rows();
                format!(
                    "follow {:?}, {} rows, {} shown, prefs {:?}, last visible {:?}",
                    self.viewer.follow,
                    self.viewer.view.len(),
                    self.viewer.shown(),
                    self.settings.log_viewer,
                    rows.iter().rev().take(3).collect::<Vec<_>>()
                )
            }

            /// Logs `level` at `target` and checks the line shows up at the
            /// bottom of a following view.
            fn expect_live(&mut self, step: &str, level: Level, target: &str) {
                let message = self.probe(level, target);
                self.frames(3);
                let rows = self.visible_rows();
                assert!(
                    matches!(self.viewer.follow, Follow::Live),
                    "{step}: view stopped following; {}",
                    self.describe()
                );
                assert!(
                    rows.last().is_some_and(|row| row.ends_with(&message)),
                    "{step}: '{message}' is not the newest visible row; {}",
                    self.describe()
                );
            }

            fn expect_hidden(&mut self, step: &str, level: Level, target: &str) {
                let message = self.probe(level, target);
                self.frames(3);
                assert!(
                    !self.visible_rows().iter().any(|row| row.ends_with(&message)),
                    "{step}: '{message}' should be filtered out; {}",
                    self.describe()
                );
            }
        }

        #[test]
        fn idle_app_repaints_while_the_window_is_open() {
            let mut harness = Harness::new();
            harness.frames(240);
            for round in 0..3 {
                let message = harness.probe(Level::Info, "nexium_gui::idle");
                let mut waited = Duration::ZERO;
                while !harness.visible_rows().iter().any(|row| row.ends_with(&message)) {
                    let delay = harness.repaint_delay();
                    assert!(
                        delay <= LIVE_POLL,
                        "round {round}: the open window scheduled no redraw ({delay:?}), so '{message}' waits for input"
                    );
                    waited += delay;
                    assert!(waited <= LIVE_POLL * 3, "round {round}: '{message}' never drawn");
                    harness.time += delay.as_secs_f64();
                    harness.frame(Vec::new());
                }
                harness.frames(240);
            }
            harness.open = false;
            harness.frames(240);
            assert_eq!(harness.repaint_delay(), Duration::MAX);
        }

        #[test]
        fn new_lines_keep_showing_through_every_filter_change() {
            let mut h = Harness::new();
            h.expect_live("start", Level::Info, "nexium_gui::boot");
            for (index, name) in LEVEL_NAMES.iter().enumerate() {
                let label = format!("● {name}");
                h.click(&label);
                let other = if index == 2 { Level::Warn } else { Level::Info };
                h.expect_live(&format!("{name} off"), other, "nexium_kernel::svc");
                h.click(&label);
                h.expect_live(&format!("{name} on"), other, "nexium_kernel::svc");
            }
            h.click("● Debug");
            h.click("● Trace");
            h.expect_live("debug and trace off", Level::Info, "nexium_kernel::svc");
            h.expect_hidden("builtin noise", Level::Info, "wgpu_core");
            h.expect_live("debug and trace off", Level::Error, "nexium_kernel::svc");

            h.click("Log levels (apply immediately)");
            h.frames(10);
            h.click("Everything (trace)");
            h.frames(2);
            h.expect_hidden("preset with debug off", Level::Debug, "nexium_kernel::svc");
            h.expect_live("preset with debug off", Level::Info, "nexium_kernel::svc");
            h.click("● Debug");
            h.expect_live("preset with debug on", Level::Debug, "nexium_kernel::svc");
            h.click("● Trace");
            h.expect_live("preset with trace on", Level::Trace, "nexium_kernel::svc");
            for preset in PRESETS {
                h.click(preset.label);
                h.expect_live(preset.label, Level::Info, "nexium_gpu::renderer");
            }
            h.click("Clear rules");
            h.expect_live("rules cleared", Level::Info, "nexium_gpu::renderer");
            h.expect_hidden("rules cleared", Level::Debug, "nexium_gpu::renderer");
            if let Some(filter) = h.viewer.log_filter() {
                filter.set_default_level(log::LevelFilter::Debug);
            }
            h.expect_live("global debug", Level::Debug, "nexium_gpu::renderer");
            if let Some(filter) = h.viewer.log_filter() {
                filter.set_default_level(log::LevelFilter::Info);
            }
            h.click("Log levels (apply immediately)");
            h.frames(10);

            h.type_into("text", "probe");
            h.expect_live("search", Level::Info, "nexium_kernel::svc");
            h.click("Only matching");
            h.expect_live("only matching", Level::Warn, "nexium_kernel::svc");
            h.click(".*");
            h.expect_live("regex", Level::Info, "nexium_kernel::svc");
            h.type_into("probe", "-[0-9]+ (info|warn)");
            h.expect_live("regex narrowed", Level::Warn, "nexium_kernel::svc");
            h.expect_hidden("regex narrowed", Level::Error, "nexium_kernel::svc");
            h.frame(vec![Event::Text("(".into())]);
            h.frames(2);
            assert!(h.has_text("invalid regex"));
            h.expect_live("invalid regex keeps the last valid search", Level::Info, "nexium_kernel::svc");
            h.expect_hidden("invalid regex keeps the last valid search", Level::Error, "nexium_kernel::svc");
            h.click("Only matching");
            h.expect_live("invalid regex, all lines", Level::Info, "nexium_kernel::svc");
            h.click(".*");
            h.expect_live("plain search again", Level::Info, "nexium_kernel::svc");

            h.type_into("nexium_kernel, -wgpu", "nexium_kernel");
            h.expect_live("target include", Level::Info, "nexium_kernel::svc");
            h.expect_hidden("target include", Level::Info, "nexium_gpu::renderer");
            h.frame(vec![Event::Text(", -svc".into())]);
            h.frames(2);
            h.expect_live("target exclude", Level::Info, "nexium_kernel::threads");
            h.expect_hidden("target exclude", Level::Info, "nexium_kernel::svc");

            h.click("Collapse repeats");
            for _ in 0..3 {
                h.log(Level::Warn, "nexium_kernel::threads", "same line");
            }
            h.frames(3);
            assert!(h.visible_rows().last().is_some_and(|row| row.ends_with("same line  (×3)")), "{}", h.describe());
            h.expect_live("collapse", Level::Info, "nexium_kernel::threads");
            h.click("Wrap");
            h.log(Level::Info, "nexium_kernel::threads", &"long ".repeat(120));
            h.expect_live("wrap", Level::Info, "nexium_kernel::threads");
            h.click("Wrap");
            h.log(Level::Info, "nexium_kernel::threads", &"wide ".repeat(400));
            h.expect_live("no wrap with a wide line", Level::Info, "nexium_kernel::threads");
            h.click("Collapse repeats");
            h.expect_live("collapse off", Level::Info, "nexium_kernel::threads");

            for step in 0..200 {
                h.log(Level::Info, "nexium_kernel::threads", &format!("burst {step}"));
                h.frame(Vec::new());
            }
            h.expect_live("after a burst", Level::Info, "nexium_kernel::threads");
        }

        #[test]
        fn scrolling_up_pauses_until_the_user_returns() {
            let mut h = Harness::new();
            h.expect_live("start", Level::Info, "nexium_gui::boot");
            h.wheel(200.0);
            assert!(
                matches!(h.viewer.follow, Follow::Paused { reason: PauseReason::Scroll, .. }),
                "{}",
                h.describe()
            );
            let held = h.probe(Level::Info, "nexium_gui::boot");
            h.frames(3);
            assert!(!h.visible_rows().iter().any(|row| row.ends_with(&held)));
            assert!(h.has_text("1 new line"), "paused view does not say lines are waiting");
            h.wheel(-100_000.0);
            assert!(matches!(h.viewer.follow, Follow::Live), "{}", h.describe());
            assert!(h.visible_rows().iter().any(|row| row.ends_with(&held)));
            h.expect_live("scrolled back down", Level::Info, "nexium_gui::boot");
        }

        #[test]
        fn a_filter_that_leaves_nothing_to_scroll_resumes_a_scrolled_view() {
            let mut h = Harness::new();
            h.wheel(200.0);
            assert!(matches!(h.viewer.follow, Follow::Paused { reason: PauseReason::Scroll, .. }));
            h.type_into("nexium_kernel, -wgpu", "nexium_kernel");
            h.expect_live("short filtered view", Level::Info, "nexium_kernel::svc");
        }

        #[test]
        fn jump_to_latest_resumes() {
            let mut h = Harness::new();
            h.wheel(300.0);
            let held = h.probe(Level::Info, "nexium_gui::boot");
            h.frames(3);
            let label = h
                .texts()
                .into_iter()
                .find(|(text, rect, clip)| text.starts_with("Paused, 1 new line, jump to latest") && clip.intersects(*rect))
                .map(|(text, _, _)| text)
                .expect("no jump to latest button over the paused view");
            h.click(&label);
            h.frames(2);
            assert!(matches!(h.viewer.follow, Follow::Live), "{}", h.describe());
            assert!(h.visible_rows().last().is_some_and(|row| row.ends_with(&held)), "{}", h.describe());
        }

        #[test]
        fn selecting_pauses_and_clearing_the_selection_follows_again() {
            let mut h = Harness::new();
            let area = h.viewer.last_inner;
            h.click_at(Pos2::new(area.left() + 40.0, area.bottom() - 30.0));
            assert!(
                matches!(h.viewer.follow, Follow::Paused { reason: PauseReason::Selection, .. }),
                "{}",
                h.describe()
            );
            let held = h.probe(Level::Info, "nexium_gui::boot");
            h.frames(3);
            assert!(!h.visible_rows().iter().any(|row| row.ends_with(&held)));
            h.key(egui::Key::Escape);
            h.frames(2);
            assert!(h.viewer.selection.is_none());
            assert!(h.visible_rows().iter().any(|row| row.ends_with(&held)), "{}", h.describe());
            h.expect_live("selection cleared", Level::Info, "nexium_gui::boot");
        }

        #[test]
        fn filter_changes_while_following_with_a_short_view_keep_following() {
            let mut h = Harness::new();
            h.type_into("nexium_kernel, -wgpu", "nexium_kernel");
            for n in 0..5 {
                h.expect_live(&format!("short view {n}"), Level::Info, "nexium_kernel::svc");
            }
            for _ in 0..60 {
                h.log(Level::Info, "nexium_kernel::svc", "fills the view");
                h.frame(Vec::new());
            }
            h.expect_live("view filled", Level::Info, "nexium_kernel::svc");
            h.click("Collapse repeats");
            h.expect_live("collapsed short view", Level::Info, "nexium_kernel::svc");
            h.click("Collapse repeats");
            h.expect_live("expanded again", Level::Info, "nexium_kernel::svc");
        }
    }
}
