use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};

mod audio;
mod chorus;
mod lyrics;
mod asr;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Choose {
    First,
    Best,
}

#[derive(Parser, Debug)]
#[command(
    name = "chorus-finder",
    version,
    about = "Find the chorus start/end of an audio file from its lyrics"
)]
struct Cli {
    /// Audio file (mp3, wav, flac, ogg, ...)
    audio: PathBuf,

    /// Explicit lyrics file (.lrc or .txt)
    #[arg(long)]
    lyrics: Option<PathBuf>,

    /// Whisper ggml model, required only for lyrics without timestamps
    #[arg(long)]
    whisper_model: Option<PathBuf>,

    /// Report the repeat of the best chorus block instead of its first occurrence
    #[arg(long, value_enum, default_value_t = Choose::First)]
    choose: Choose,

    /// Print the full result as JSON
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match chorus::find_chorus(
        &cli.audio,
        cli.lyrics.as_deref(),
        matches!(cli.choose, Choose::Best),
        cli.whisper_model.as_deref(),
    ) {
        Ok(result) => {
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("Audio: {}", result.audio);
                println!(
                    "Lyrics: {} ({}, BPM {:.2})",
                    result.lyrics_source, result.lyrics_format, result.bpm
                );
                println!(
                    "Chorus: {} - {} ({:.2} bars, confidence {:.2})",
                    result.chorus.start_mmss,
                    result.chorus.end_mmss,
                    result.chorus.body_bars,
                    result.chorus.confidence
                );
                println!(
                    "Occurrences: {}",
                    result
                        .occurrences
                        .iter()
                        .map(|item| format!("{}-{}", item.start_mmss, item.end_mmss))
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            Ok(())
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(1);
        }
    }
}
