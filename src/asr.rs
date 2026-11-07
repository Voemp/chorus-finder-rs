use std::path::Path;

use anyhow::{Context, Result};
use pinyin::ToPinyin;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::decode_mono;
use crate::lyrics::LyricLine;

#[derive(Clone, Debug)]
pub struct AsrWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

pub fn transcribe_words(audio_path: &Path, model_path: &Path) -> Result<Vec<AsrWord>> {
    whisper_rs::install_logging_hooks();
    let (mono, sample_rate) = decode_mono(audio_path)?;
    if mono.is_empty() {
        anyhow::bail!("decoded audio is empty");
    }
    let audio = resample_linear(&mono, sample_rate, 16000);

    let context = WhisperContext::new_with_params(
        model_path,
        WhisperContextParameters::default(),
    )
    .with_context(|| format!("cannot load Whisper model {}", model_path.display()))?;
    let mut state = context.create_state().context("cannot create Whisper state")?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_translate(false);
    params.set_language(None);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(true);

    state
        .full(params, &audio)
        .context("Whisper transcription failed")?;

    let mut words = Vec::new();
    for segment in state.as_iter() {
        for token_index in 0..segment.n_tokens() {
            let Some(token) = segment.get_token(token_index) else {
                continue;
            };
            let text = token.to_str_lossy().unwrap_or_default().trim().to_string();
            if text.is_empty()
                || text.starts_with("<|")
                || text.starts_with("[_")
                || text == "[BLANK_AUDIO]"
            {
                continue;
            }
            let data = token.token_data();
            let start = data.t0 as f64 / 100.0;
            let end = (data.t1 as f64 / 100.0).max(start + 0.05);
            words.push(AsrWord { text, start, end });
        }
    }
    Ok(words)
}

fn resample_linear(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == 0 || source_rate == target_rate {
        return input.to_vec();
    }
    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = ((input.len() as f64) / ratio).round() as usize;
    let mut output = vec![0.0f32; output_len];
    for (idx, value) in output.iter_mut().enumerate() {
        let position = idx as f64 * ratio;
        let lower = position.floor() as usize;
        let upper = (lower + 1).min(input.len().saturating_sub(1));
        let weight = (position - lower as f64) as f32;
        *value = input[lower.min(input.len() - 1)] * (1.0 - weight)
            + input[upper] * weight;
    }
    output
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x9fff | 0x3040..=0x30ff | 0xac00..=0xd7af)
}

/// Convert text into comparable units: toneless pinyin for CJK, lowercase
/// alphanumerics elsewhere.
fn units(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for c in text.chars() {
        if is_cjk(c) {
            if let Some(pinyin) = c.to_pinyin() {
                out.push(pinyin.plain().to_string());
            } else {
                out.push(c.to_string());
            }
        } else if c.is_alphanumeric() {
            out.push(c.to_lowercase().to_string());
        }
    }
    out
}

/// Align untimed lyric lines to ASR tokens using LCS over pinyin units.
/// Returns alignment coverage in `[0, 1]`.
pub fn align_plain_lyrics(lines: &mut [LyricLine], words: &[AsrWord]) -> f64 {
    let mut lyric_units: Vec<String> = Vec::new();
    let mut unit_line: Vec<usize> = Vec::new();
    for (line_idx, line) in lines.iter().enumerate() {
        for unit in units(&line.text) {
            lyric_units.push(unit);
            unit_line.push(line_idx);
        }
    }

    let mut asr_units: Vec<String> = Vec::new();
    let mut asr_times: Vec<(f64, f64)> = Vec::new();
    for word in words {
        let word_units = units(&word.text);
        if word_units.is_empty() {
            continue;
        }
        let span = (word.end - word.start).max(0.01);
        for (idx, unit) in word_units.iter().enumerate() {
            asr_units.push(unit.clone());
            asr_times.push((
                word.start + span * idx as f64 / word_units.len() as f64,
                word.start + span * (idx + 1) as f64 / word_units.len() as f64,
            ));
        }
    }

    if lyric_units.is_empty() || asr_units.is_empty() {
        return 0.0;
    }

    let n = lyric_units.len();
    let m = asr_units.len();
    let width = m + 1;
    let mut table = vec![0u32; (n + 1) * width];
    for i in 1..=n {
        for j in 1..=m {
            table[i * width + j] = if lyric_units[i - 1] == asr_units[j - 1] {
                table[(i - 1) * width + (j - 1)] + 1
            } else {
                table[(i - 1) * width + j].max(table[i * width + (j - 1)])
            };
        }
    }

    let mut matches: Vec<(usize, usize)> = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if lyric_units[i - 1] == asr_units[j - 1] {
            matches.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if table[(i - 1) * width + j] >= table[i * width + (j - 1)] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    matches.reverse();

    let mut line_start: Vec<Option<f64>> = vec![None; lines.len()];
    let mut line_end: Vec<Option<f64>> = vec![None; lines.len()];
    let mut matched_units = 0usize;
    for (lyric_idx, asr_idx) in matches {
        let line_idx = unit_line[lyric_idx];
        let (start, end) = asr_times[asr_idx];
        line_start[line_idx] = Some(
            line_start[line_idx]
                .map(|value| value.min(start))
                .unwrap_or(start),
        );
        line_end[line_idx] = Some(
            line_end[line_idx]
                .map(|value| value.max(end))
                .unwrap_or(end),
        );
        matched_units += 1;
    }

    let known: Vec<usize> = (0..lines.len())
        .filter(|idx| line_start[*idx].is_some() && line_end[*idx].is_some())
        .collect();

    for idx in 0..lines.len() {
        let start = line_start[idx];
        let end = line_end[idx];
        if let (Some(start), Some(end)) = (start, end) {
            lines[idx].start = Some(start);
            lines[idx].end = Some(end.max(start + 0.1));
        } else {
            let prev = known.iter().rev().find(|value| **value < idx).copied();
            let next = known.iter().find(|value| **value > idx).copied();
            match (prev, next) {
                (Some(prev), Some(next)) => {
                    let prev_end = line_end[prev].unwrap_or(0.0);
                    let next_start = line_start[next].unwrap_or(prev_end + 1.0);
                    let span = (next_start - prev_end).max(0.5);
                    let count = (next - prev).max(1) as f64;
                    let offset = (idx - prev) as f64;
                    lines[idx].start = Some(prev_end + span * (offset - 1.0) / count);
                    lines[idx].end = Some(prev_end + span * offset / count);
                }
                (Some(prev), None) => {
                    let start = line_end[prev].unwrap_or(0.0);
                    lines[idx].start = Some(start);
                    lines[idx].end = Some(start + 3.0);
                }
                (None, Some(next)) => {
                    let end = line_start[next].unwrap_or(3.0);
                    lines[idx].end = Some(end);
                    lines[idx].start = Some((end - 3.0).max(0.0));
                }
                (None, None) => {
                    lines[idx].start = Some(0.0);
                    lines[idx].end = Some(0.0);
                }
            }
        }
        lines[idx].aligned = true;
        lines[idx].end_estimated = true;
    }

    matched_units as f64 / lyric_units.len() as f64
}
