use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use lofty::file::TaggedFileExt;
use lofty::tag::ItemKey;
use regex::Regex;
use std::sync::LazyLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LyricsFormat {
    WordLrc,
    LineLrc,
    Plain,
}

impl LyricsFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WordLrc => "word-lrc",
            Self::LineLrc => "line-lrc",
            Self::Plain => "plain",
        }
    }
}

#[derive(Clone, Debug)]
pub struct LyricWord {
    pub text: String,
    pub start: f64,
    pub end: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct LyricLine {
    pub text: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub words: Vec<LyricWord>,
    #[allow(dead_code)]
    pub timed: bool,
    pub aligned: bool,
    pub end_estimated: bool,
}

#[derive(Clone, Debug)]
pub struct LoadedLyrics {
    pub lines: Vec<LyricLine>,
    pub source: String,
    pub format: LyricsFormat,
}

static LRC_TIME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]").expect("valid regex")
});

static META_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[[A-Za-z]+:.*\]$").expect("valid regex"));

fn tag_seconds(minutes: &str, seconds: &str) -> f64 {
    let m: f64 = minutes.parse().unwrap_or(0.0);
    let s: f64 = seconds.replace(':', ".").parse().unwrap_or(0.0);
    m * 60.0 + s
}

pub fn normalize_text(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

pub fn mmss(seconds: f64) -> String {
    let minutes = (seconds / 60.0).floor() as i64;
    let secs = seconds - minutes as f64 * 60.0;
    format!("{minutes:02}:{secs:05.2}")
}

pub fn load_lyrics(audio: &Path, explicit: Option<&Path>) -> Result<LoadedLyrics> {
    let (text, source) = if let Some(path) = explicit {
        (
            fs::read_to_string(path)
                .with_context(|| format!("cannot read lyrics file {}", path.display()))?,
            path.display().to_string(),
        )
    } else if let Some((text, source)) = read_sidecar(audio)? {
        (text, source)
    } else if let Some((text, source)) = read_embedded(audio)? {
        (text, source)
    } else {
        anyhow::bail!(
            "no lyrics found: pass --lyrics, place a same-name .lrc/.txt beside the audio, \
             or use a file with embedded lyrics"
        );
    };

    let text = text.trim_start_matches('\u{feff}').to_string();
    if text.trim().is_empty() {
        anyhow::bail!("lyrics source is empty: {source}");
    }

    if LRC_TIME_RE.is_match(&text) {
        let lines = parse_lrc(&text);
        if lines.is_empty() {
            anyhow::bail!("no parsable LRC lines in {source}");
        }
        let format = if lines.iter().any(|line| !line.words.is_empty()) {
            LyricsFormat::WordLrc
        } else {
            LyricsFormat::LineLrc
        };
        let mut lines = lines;
        finalize_line_ends(&mut lines, &format);
        Ok(LoadedLyrics {
            lines,
            source,
            format,
        })
    } else {
        let lines = parse_plain_lyrics(&text);
        if lines.is_empty() {
            anyhow::bail!("no usable lyric lines in {source}");
        }
        Ok(LoadedLyrics {
            lines,
            source,
            format: LyricsFormat::Plain,
        })
    }
}

fn read_sidecar(audio: &Path) -> Result<Option<(String, String)>> {
    for suffix in [".lrc", ".LRC", ".txt", ".TXT"] {
        let candidate = audio.with_extension(&suffix[1..]);
        let candidate_upper = audio.with_extension(&suffix[1..].to_uppercase());
        for path in [candidate, candidate_upper] {
            if path.exists() {
                let text = fs::read_to_string(&path)
                    .with_context(|| format!("cannot read lyrics file {}", path.display()))?;
                return Ok(Some((text, path.display().to_string())));
            }
        }
    }
    Ok(None)
}

fn read_embedded(audio: &Path) -> Result<Option<(String, String)>> {
    let tags = match lofty::read_from_path(audio) {
        Ok(tagged) => tagged,
        Err(_) => return Ok(None),
    };

    let mut best: Option<(String, String)> = None;
    let mut best_score = -1i64;

    for tag in tags.tags() {
        for key in [ItemKey::UnsyncLyrics, ItemKey::Lyrics] {
            if let Some(value) = tag.get_string(key) {
                let score = LRC_TIME_RE.find_iter(value).count() as i64 * 1000
                    + value.len() as i64;
                if score > best_score {
                    best_score = score;
                    best = Some((value.to_string(), format!("embedded:{key:?}")));
                }
            }
        }

        for item in tag.items() {
            if let Some(value) = item.value().text() {
                if !looks_like_lyrics(value) {
                    continue;
                }
                let score = LRC_TIME_RE.find_iter(value).count() as i64 * 1000
                    + value.len() as i64;
                if score > best_score {
                    best_score = score;
                    best = Some((value.to_string(), format!("embedded:{:?}", item.key())));
                }
            }
        }
    }

    Ok(best)
}

fn looks_like_lyrics(value: &str) -> bool {
    !value.trim().is_empty()
        && (LRC_TIME_RE.is_match(value)
            || value.lines().count() > 1
            || value.to_lowercase().contains("lyric"))
}

pub fn parse_lrc(text: &str) -> Vec<LyricLine> {
    let mut out = Vec::new();

    for raw in text.lines() {
        let raw = raw.trim_start_matches('\u{feff}').trim();
        if raw.is_empty() || META_RE.is_match(raw) {
            continue;
        }

        let matches: Vec<_> = LRC_TIME_RE.find_iter(raw).collect();
        if matches.is_empty() {
            continue;
        }

        // Consecutive timestamps at the start: [t1][t2]text
        let mut leading = Vec::new();
        let mut cursor = 0usize;
        for m in &matches {
            if raw[cursor..m.start()].trim().is_empty() {
                leading.push(m);
                cursor = m.end();
            } else {
                break;
            }
        }

        if leading.len() >= 2 {
            let body = raw[leading.last().unwrap().end()..].trim();
            if body.is_empty() {
                continue;
            }
            for m in &leading {
                out.push(LyricLine {
                    text: body.to_string(),
                    start: Some(parse_match_seconds(raw, m)),
                    end: None,
                    words: Vec::new(),
                    timed: true,
                    end_estimated: false,
                    aligned: false,
                });
            }
            continue;
        }

        if matches.len() == 1 {
            let m = &matches[0];
            let body = raw[m.end()..].trim();
            if body.is_empty() {
                continue;
            }
            out.push(LyricLine {
                text: body.to_string(),
                start: Some(parse_match_seconds(raw, m)),
                end: None,
                words: Vec::new(),
                timed: true,
                end_estimated: false,
                aligned: false,
            });
            continue;
        }

        // Word-level: text between consecutive timestamps is one word.
        let mut words: Vec<LyricWord> = Vec::new();
        for (idx, m) in matches.iter().enumerate() {
            let seg_end = if idx + 1 < matches.len() {
                matches[idx + 1].start()
            } else {
                raw.len()
            };
            let segment = raw[m.end()..seg_end].trim();
            if segment.is_empty() {
                continue;
            }
            let word_end = if idx + 1 < matches.len() {
                Some(parse_match_seconds(raw, &matches[idx + 1]))
            } else {
                None
            };
            words.push(LyricWord {
                text: segment.to_string(),
                start: parse_match_seconds(raw, m),
                end: word_end,
            });
        }

        if words.is_empty() {
            continue;
        }
        let body: String = words.iter().map(|w| w.text.as_str()).collect();

        let final_end = if raw[matches.last().unwrap().end()..].trim().is_empty() {
            Some(parse_match_seconds(raw, matches.last().unwrap()))
        } else {
            None
        };

        if let Some(end) = final_end {
            if let Some(last) = words.last_mut() {
                last.end = Some(end);
            }
        } else if let Some(last) = words.last_mut() {
            if last.end.is_none() {
                last.end = Some(last.start + 0.3);
            }
        }

        let start = words.first().unwrap().start;
        let end = words.last().and_then(|w| w.end);
        out.push(LyricLine {
            text: body,
            start: Some(start),
            end,
            words,
            timed: true,
            end_estimated: false,
            aligned: false,
        });
    }

    out
}

fn parse_match_seconds(raw: &str, m: &regex::Match) -> f64 {
    let whole = &raw[m.start() + 1..m.end() - 1];
    let (minutes, seconds) = whole
        .split_once(':')
        .unwrap_or((whole, "0"));
    tag_seconds(minutes, seconds)
}

pub fn parse_plain_lyrics(text: &str) -> Vec<LyricLine> {
    text.lines()
        .map(|line| line.trim_start_matches('\u{feff}').trim())
        .filter(|line| !line.is_empty())
        .filter(|line| !META_RE.is_match(line))
        .filter(|line| {
            !(line.starts_with('[') && line.ends_with(']') && !LRC_TIME_RE.is_match(line))
        })
        .map(|line| LyricLine {
            text: line.to_string(),
            start: None,
            end: None,
            words: Vec::new(),
            timed: false,
            end_estimated: false,
            aligned: false,
        })
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 3.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

pub fn finalize_line_ends(lines: &mut [LyricLine], format: &LyricsFormat) {
    let starts: Vec<f64> = lines.iter().filter_map(|line| line.start).collect();
    let gaps: Vec<f64> = starts
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|gap| *gap > 0.0 && *gap <= 12.0)
        .collect();
    let typical = median(gaps);
    let max_line = (typical * 1.6).clamp(2.5, 8.0);

    for idx in 0..lines.len() {
        let Some(start) = lines[idx].start else {
            continue;
        };
        if lines[idx].end.is_some() && *format != LyricsFormat::LineLrc {
            continue;
        }

        let next_start = lines[idx + 1..]
            .iter()
            .find_map(|line| line.start);

        let end = match next_start {
            Some(next) => next.min(start + max_line),
            None => start + typical,
        };
        lines[idx].end = Some(end);
        lines[idx].end_estimated = true;
    }
}

/// Find the line after the last matched one that still belongs to the block.
pub fn extend_last_index(lines: &[LyricLine], idx: usize, target_end: f64) -> usize {
    let mut last = idx;
    while last + 1 < lines.len() {
        match lines[last + 1].start {
            Some(start) if start <= target_end + 1.0 => last += 1,
            _ => break,
        }
    }
    last
}

pub fn next_line_start(lines: &[LyricLine], idx: usize) -> Option<f64> {
    lines[idx + 1..].iter().find_map(|line| line.start)
}

