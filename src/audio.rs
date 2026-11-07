use std::f32::consts::PI;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub struct AudioFeatures {
    pub bpm: f64,
    pub bar_seconds: f64,
    pub duration: f64,
    pub rms: Vec<f32>,
    pub rms_times: Vec<f32>,
    pub onset: Vec<f32>,
    pub onset_times: Vec<f32>,
}

pub fn analyze_audio(path: &Path) -> Result<AudioFeatures> {
    let (mono, sample_rate) = decode_mono(path)?;
    if mono.is_empty() || sample_rate == 0 {
        anyhow::bail!("decoded audio is empty: {}", path.display());
    }

    let hop = 512usize;
    let window = 1024usize;
    let duration = mono.len() as f64 / sample_rate as f64;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window);
    let mut hann = vec![0.0f32; window];
    for (idx, value) in hann.iter_mut().enumerate() {
        *value = 0.5 - 0.5 * (2.0 * PI * idx as f32 / (window - 1) as f32).cos();
    }

    let mut rms = Vec::new();
    let mut rms_times = Vec::new();
    let mut onset = Vec::new();
    let mut onset_times = Vec::new();
    let mut prev_magnitude = vec![0.0f32; window / 2 + 1];

    let mut start = 0usize;
    while start + window <= mono.len() {
        let mut buffer: Vec<Complex32> = mono[start..start + window]
            .iter()
            .zip(hann.iter())
            .map(|(sample, window_value)| Complex32::new(sample * window_value, 0.0))
            .collect();
        fft.process(&mut buffer);

        let mut magnitude = vec![0.0f32; window / 2 + 1];
        let mut flux = 0.0f32;
        for (idx, value) in buffer.iter().take(window / 2 + 1).enumerate() {
            let mag = value.norm();
            magnitude[idx] = mag;
            let diff = mag - prev_magnitude[idx];
            if diff > 0.0 {
                flux += diff;
            }
        }
        prev_magnitude = magnitude;

        let frame = &mono[start..start + window];
        let energy = frame.iter().map(|sample| sample * sample).sum::<f32>()
            / window as f32;
        let time = start as f32 / sample_rate as f32;

        rms.push(energy.sqrt());
        rms_times.push(time);
        onset.push(flux);
        onset_times.push(time);

        start += hop;
    }

    let bpm = estimate_bpm(&onset, sample_rate, hop);
    let bar_seconds = 4.0 * 60.0 / bpm;

    Ok(AudioFeatures {
        bpm,
        bar_seconds,
        duration,
        rms,
        rms_times,
        onset,
        onset_times,
    })
}

pub fn decode_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path)
        .with_context(|| format!("cannot open audio file {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .with_context(|| format!("unsupported audio container: {}", path.display()))?;

    let track = format
        .default_track(TrackType::Audio)
        .context("no audio track found")?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .context("missing audio codec parameters")?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .context("cannot create audio decoder")?;

    let mut sample_rate = params.sample_rate.unwrap_or(0);
    let mut mono: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(Error::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(Error::DecodeError(_)) => continue,
            Err(_) => break,
        };

        let spec = decoded.spec();
        sample_rate = spec.rate();
        let channels = spec.channels().count().max(1);
        let sample_count = decoded.samples_interleaved();
        interleaved.resize(sample_count, 0.0);
        decoded.copy_to_slice_interleaved(&mut interleaved);

        mono.reserve(sample_count / channels);
        for frame in interleaved.chunks_exact(channels) {
            let sum: f32 = frame.iter().copied().sum();
            mono.push(sum / channels as f32);
        }
    }

    Ok((mono, sample_rate))
}

fn estimate_bpm(onset: &[f32], sample_rate: u32, hop: usize) -> f64 {
    if onset.len() < 64 {
        return 120.0;
    }

    let mean = onset.iter().map(|value| *value as f64).sum::<f64>() / onset.len() as f64;
    let normalized: Vec<f64> = onset
        .iter()
        .map(|value| (*value as f64 - mean).max(0.0))
        .collect();

    let frame_rate = sample_rate as f64 / hop as f64;
    let min_lag = (frame_rate * 60.0 / 200.0).floor().max(1.0) as usize;
    let max_lag = (frame_rate * 60.0 / 50.0).ceil() as usize;
    let max_lag = max_lag.min(normalized.len().saturating_sub(1));

    let mut best_lag = 0usize;
    let mut best_score = f64::MIN;
    for lag in min_lag..=max_lag {
        let count = normalized.len() - lag;
        if count == 0 {
            continue;
        }
        let mut score = 0.0;
        for idx in 0..count {
            score += normalized[idx] * normalized[idx + lag];
        }
        score /= count as f64;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    if best_lag == 0 || best_score <= 0.0 {
        return 120.0;
    }

    let mut bpm = 60.0 * frame_rate / best_lag as f64;
    while bpm < 70.0 {
        bpm *= 2.0;
    }
    while bpm > 180.0 {
        bpm /= 2.0;
    }
    refine_bpm(bpm, &normalized, frame_rate)
}

fn comb_score(onset: &[f64], period_frames: f64, phase: f64) -> f64 {
    if period_frames <= 1.0 {
        return 0.0;
    }
    let mut position = phase;
    let mut sum = 0.0;
    let mut count = 0usize;
    while position < (onset.len() - 1) as f64 {
        let idx = position.round() as usize;
        if idx < onset.len() {
            sum += onset[idx];
            count += 1;
        }
        position += period_frames;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn refine_bpm(initial: f64, onset: &[f64], frame_rate: f64) -> f64 {
    let period = frame_rate * 60.0 / initial;
    let base_score = (0..4)
        .map(|phase| comb_score(onset, period, phase as f64 * period / 4.0))
        .fold(f64::MIN, f64::max);

    let mut best_bpm = initial;
    let mut best_score = base_score;
    let mut bpm = initial * 0.92;
    while bpm <= initial * 1.08 {
        let period = frame_rate * 60.0 / bpm;
        for phase in 0..4 {
            let score = comb_score(onset, period, phase as f64 * period / 4.0);
            if score > best_score {
                best_score = score;
                best_bpm = bpm;
            }
        }
        bpm += 0.1;
    }

    if best_score > base_score * 1.02 {
        best_bpm
    } else {
        initial
    }
}

pub fn percentile(values: &[f32], percent: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let position = (sorted.len() - 1) as f64 * percent / 100.0;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = (position - lower as f64) as f32;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

pub fn mean_rms(features: &AudioFeatures, start: f64, end: f64) -> f32 {
    let values: Vec<f32> = features
        .rms
        .iter()
        .zip(features.rms_times.iter())
        .filter(|(_, time)| **time >= start as f32 && **time < end as f32)
        .map(|(value, _)| *value)
        .collect();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

