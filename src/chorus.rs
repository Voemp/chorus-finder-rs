use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::audio::{analyze_audio, mean_rms, percentile, AudioFeatures};
use crate::lyrics::{
    extend_last_index, load_lyrics, mmss, next_line_start, normalize_text, LyricLine, LyricsFormat,
};

const BAR_TARGETS: [f64; 4] = [8.0, 16.0, 24.0, 32.0];

#[derive(Clone, Debug)]
pub struct Candidate {
    pub start: f64,
    pub end: f64,
    pub start2: Option<f64>,
    pub end2: Option<f64>,
    pub body_bars: f64,
    pub support: usize,
    pub coverage: f64,
    pub energy: f32,
    pub score: f64,
}

#[derive(Serialize)]
pub struct Occurrence {
    pub start: f64,
    pub end: f64,
    pub start_mmss: String,
    pub end_mmss: String,
}

#[derive(Serialize)]
pub struct ChorusInfo {
    pub start: f64,
    pub end: f64,
    pub start_mmss: String,
    pub end_mmss: String,
    pub body_bars: f64,
    pub confidence: f64,
}

#[derive(Serialize)]
pub struct CandidateInfo {
    pub start: f64,
    pub end: f64,
    pub start2: Option<f64>,
    pub end2: Option<f64>,
    pub body_bars: f64,
    pub support: usize,
    pub coverage: f64,
    pub energy: f32,
    pub score: f64,
}

#[derive(Serialize)]
pub struct ChorusOutput {
    pub audio: String,
    pub lyrics_source: String,
    pub lyrics_format: String,
    pub bpm: f64,
    pub duration: f64,
    pub chorus: ChorusInfo,
    pub first_chorus: Occurrence,
    pub best_chorus: Occurrence,
    pub occurrences: Vec<Occurrence>,
    pub repeat_support: usize,
    pub line_coverage: f64,
    pub alignment_coverage: Option<f64>,
    pub candidates: Vec<CandidateInfo>,
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn occurrence(start: f64, end: f64) -> Occurrence {
    Occurrence {
        start: round3(start),
        end: round3(end),
        start_mmss: mmss(start),
        end_mmss: mmss(end),
    }
}

fn bar_score(body_bars: f64) -> f64 {
    let distance = BAR_TARGETS
        .iter()
        .map(|target| (body_bars - target).abs())
        .fold(f64::INFINITY, f64::min);
    if distance <= 1.0 {
        1.0
    } else if distance <= 2.0 {
        0.6
    } else {
        0.2
    }
}

fn refine_end(
    features: &AudioFeatures,
    last_line_start: f64,
    estimated_end: Option<f64>,
    max_end: Option<f64>,
) -> f64 {
    let mut latest = last_line_start + 7.0;
    if let Some(max_end) = max_end {
        latest = latest.min(max_end - 0.15);
    }
    latest = latest.max(last_line_start + 2.5);
    let earliest = last_line_start + 1.0;

    let mut best_time: Option<f64> = None;
    let mut best_value = f32::MIN;
    for (value, time) in features.onset.iter().zip(features.onset_times.iter()) {
        let time = *time as f64;
        if time < earliest || time > latest {
            continue;
        }
        if *value > best_value {
            best_value = *value;
            best_time = Some(time);
        }
    }

    if let Some(time) = best_time {
        if best_value >= percentile(&features.onset, 75.0) {
            return time;
        }
    }
    estimated_end.unwrap_or(last_line_start + 4.0)
}

pub fn find_candidates(
    lines: &[LyricLine],
    features: &AudioFeatures,
) -> Vec<Candidate> {
    if !lines.iter().any(|line| line.start.is_some()) {
        return Vec::new();
    }

    let occurrences: HashMap<String, Vec<usize>> = {
        let mut map: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, line) in lines.iter().enumerate() {
            let key = normalize_text(&line.text);
            if !key.is_empty() {
                map.entry(key).or_default().push(idx);
            }
        }
        map
    };

    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for indices in occurrences.values() {
        for (position, first) in indices.iter().enumerate() {
            for second in indices[position + 1..].iter() {
                let (Some(first_start), Some(second_start)) =
                    (lines[*first].start, lines[*second].start)
                else {
                    continue;
                };
                let delta = second_start - first_start;
                if delta >= 10.0 {
                    pairs.push((delta, *first, *second));
                }
            }
        }
    }

    if pairs.is_empty() {
        return Vec::new();
    }

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut clusters: Vec<Vec<(f64, usize, usize)>> = Vec::new();
    for pair in pairs {
        let mut best_cluster: Option<usize> = None;
        let mut best_distance = f64::INFINITY;
        for (idx, cluster) in clusters.iter().enumerate() {
            let mean = cluster.iter().map(|item| item.0).sum::<f64>() / cluster.len() as f64;
            let distance = (pair.0 - mean).abs();
            if distance <= 0.6 && distance < best_distance {
                best_cluster = Some(idx);
                best_distance = distance;
            }
        }
        match best_cluster {
            Some(idx) => clusters[idx].push(pair),
            None => clusters.push(vec![pair]),
        }
    }

    let p20 = percentile(&features.rms, 20.0);
    let p80 = percentile(&features.rms, 80.0);
    let rms_span = (p80 - p20).max(1e-9);

    let mut candidates = Vec::new();
    for cluster in &clusters {
        let support = cluster.len();
        if support < 2 {
            continue;
        }

        let first_idx: Vec<usize> = cluster
            .iter()
            .map(|item| item.1)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let second_idx: Vec<usize> = cluster
            .iter()
            .map(|item| item.2)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mut first_idx = first_idx;
        let mut second_idx = second_idx;
        first_idx.sort_unstable();
        second_idx.sort_unstable();
        if first_idx.len() < 2 || second_idx.len() < 2 {
            continue;
        }

        let start1 = lines[first_idx[0]].start.unwrap();
        let start2 = lines[second_idx[0]].start.unwrap();
        let mut last1 = first_idx[first_idx.len() - 1];
        let mut last2 = second_idx[second_idx.len() - 1];
        let mut body_end1 = lines[last1].start.unwrap();
        let mut body_end2 = lines[last2].start.unwrap();

        let mut body_bars = (body_end1 - start1) / features.bar_seconds;
        let target_bars = BAR_TARGETS
            .iter()
            .copied()
            .min_by(|a, b| {
                (body_bars - *a)
                    .abs()
                    .total_cmp(&(body_bars - *b).abs())
            })
            .unwrap_or(16.0);

        let (grid_end1, grid_end2) = if (target_bars - body_bars).abs() <= 2.0 {
            (
                start1 + target_bars * features.bar_seconds,
                start2 + target_bars * features.bar_seconds,
            )
        } else {
            (body_end1, body_end2)
        };

        if grid_end1 > body_end1 + 0.05 {
            last1 = extend_last_index(lines, last1, grid_end1);
            body_end1 = lines[last1].start.unwrap();
        }
        if grid_end2 > body_end2 + 0.05 {
            last2 = extend_last_index(lines, last2, grid_end2);
            body_end2 = lines[last2].start.unwrap();
        }

        let span_lines = last1 - first_idx[0] + 1;
        let coverage = first_idx.len() as f64 / span_lines.max(1) as f64;
        body_bars = (body_end1 - start1) / features.bar_seconds;
        let bar_score = bar_score(body_bars);
        let energy = mean_rms(features, start1, body_end1 + 2.0);
        let energy_score = (((energy - p20) / rms_span) as f64).clamp(0.0, 1.0);
        let position = start1 / features.duration.max(1e-9);
        let position_score = if (0.08..=0.85).contains(&position) {
            1.0
        } else {
            0.5
        };

        let score = (0.35 * (support as f64 / 4.0).min(1.0)
            + 0.25 * coverage
            + 0.25 * bar_score
            + 0.15 * energy_score)
            * (0.7 + 0.3 * position_score);

        let end1 = lines[last1].end;
        let end2 = lines[last2].end;
        let end1 = if lines[last1].end_estimated || end1.is_none() {
            refine_end(
                features,
                body_end1.max(grid_end1),
                end1,
                next_line_start(lines, last1),
            )
        } else {
            end1.unwrap()
        };
        let end2 = if lines[last2].end_estimated || end2.is_none() {
            refine_end(
                features,
                body_end2.max(grid_end2),
                end2,
                next_line_start(lines, last2),
            )
        } else {
            end2.unwrap()
        };

        candidates.push(Candidate {
            start: start1,
            end: end1,
            start2: Some(start2),
            end2: Some(end2),
            body_bars,
            support,
            coverage,
            energy,
            score: score.clamp(0.0, 1.0),
        });
    }

    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut unique: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        let duplicate = unique.iter().any(|other| {
            (candidate.start - other.start).abs() < 4.0
                && match (candidate.start2, other.start2) {
                    (Some(a), Some(b)) => (a - b).abs() < 4.0,
                    _ => true,
                }
        });
        if !duplicate {
            unique.push(candidate);
        }
    }
    unique
}

pub fn find_chorus(
    audio_path: &Path,
    lyrics_path: Option<&Path>,
    choose_best: bool,
    whisper_model: Option<&Path>,
) -> Result<ChorusOutput> {
    let mut loaded = load_lyrics(audio_path, lyrics_path)?;
    let mut alignment_coverage = None;
    if loaded.format == LyricsFormat::Plain {
        let model = whisper_model.ok_or_else(|| {
            anyhow::anyhow!(
                "plain lyrics need a Whisper model: pass --whisper-model ggml-base.bin \
                 (or ggml-small.bin for better Chinese accuracy)"
            )
        })?;
        let words = crate::asr::transcribe_words(audio_path, model)?;
        if words.is_empty() {
            anyhow::bail!("Whisper produced no words; cannot align plain lyrics");
        }
        alignment_coverage = Some(crate::asr::align_plain_lyrics(&mut loaded.lines, &words));
    }

    let features = analyze_audio(audio_path)?;
    let candidates = find_candidates(&loaded.lines, &features);
    if candidates.is_empty() {
        bail!("no repeated lyric block found; cannot determine the chorus");
    }

    let best = candidates
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap();

    let first = occurrence(best.start, best.end);
    let best_occurrence = match (best.start2, best.end2) {
        (Some(start), Some(end)) => occurrence(start, end),
        _ => occurrence(best.start, best.end),
    };
    let chosen = if choose_best {
        &best_occurrence
    } else {
        &first
    };

    let mut occurrences = vec![occurrence(best.start, best.end)];
    if let (Some(start), Some(end)) = (best.start2, best.end2) {
        occurrences.push(occurrence(start, end));
    }

    Ok(ChorusOutput {
        audio: audio_path.display().to_string(),
        lyrics_source: loaded.source,
        lyrics_format: loaded.format.as_str().to_string(),
        bpm: (features.bpm * 100.0).round() / 100.0,
        duration: (features.duration * 1000.0).round() / 1000.0,
        chorus: ChorusInfo {
            start: chosen.start,
            end: chosen.end,
            start_mmss: chosen.start_mmss.clone(),
            end_mmss: chosen.end_mmss.clone(),
            body_bars: (best.body_bars * 100.0).round() / 100.0,
            confidence: {
                let mut value = best.score;
                if let Some(coverage) = alignment_coverage {
                    value *= (coverage / 0.6).min(1.0);
                }
                (value.clamp(0.0, 1.0) * 1000.0).round() / 1000.0
            },
        },
        first_chorus: first,
        best_chorus: best_occurrence,
        occurrences,
        repeat_support: best.support,
        line_coverage: (best.coverage * 1000.0).round() / 1000.0,
        alignment_coverage: alignment_coverage.map(|value| (value * 1000.0).round() / 1000.0),
        candidates: candidates
            .iter()
            .map(|candidate| CandidateInfo {
                start: round3(candidate.start),
                end: round3(candidate.end),
                start2: candidate.start2.map(round3),
                end2: candidate.end2.map(round3),
                body_bars: (candidate.body_bars * 100.0).round() / 100.0,
                support: candidate.support,
                coverage: (candidate.coverage * 1000.0).round() / 1000.0,
                energy: (candidate.energy * 10000.0).round() / 10000.0,
                score: (candidate.score * 1000.0).round() / 1000.0,
            })
            .collect(),
    })
}
