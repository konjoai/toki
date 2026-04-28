mod config;
mod runner;

use clap::{Parser, Subcommand};
use config::TokiConfig;
use runner::ExperimentRunner;

#[derive(Parser)]
#[command(name = "toki", version, about = "Adversarial fine-tuning lab for small LLMs")]
struct Cli {
    #[arg(short, long, help = "Config file path")]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate adversarial prompts
    Generate {
        #[arg(short, long, help = "Number of prompts per category")]
        count: Option<usize>,
    },
    /// Evaluate model robustness
    Evaluate {
        #[arg(short, long, help = "Model name or path")]
        model: Option<String>,
    },
    /// Fine-tune model with LoRA on adversarial dataset
    Finetune {
        #[arg(short, long, help = "Output directory")]
        output: Option<String>,
    },
    /// Show current configuration
    Config,
    /// Run full pipeline: generate -> evaluate -> finetune -> evaluate
    Pipeline,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut config = if let Some(path) = &cli.config {
        TokiConfig::from_file(path)?
    } else {
        TokiConfig::default()
    };

    match &cli.command {
        Commands::Generate { count } => {
            if let Some(n) = count {
                config.num_adversarial_prompts = *n;
            }
            ExperimentRunner::new(config).run_generate()?;
        }
        Commands::Evaluate { model } => {
            if let Some(m) = model {
                config.model_name = m.clone();
            }
            ExperimentRunner::new(config).run_evaluate()?;
        }
        Commands::Finetune { output } => {
            if let Some(o) = output {
                config.output_dir = o.clone();
            }
            ExperimentRunner::new(config).run_finetune()?;
        }
        Commands::Config => {
            ExperimentRunner::new(config).show_config();
        }
        Commands::Pipeline => {
            let runner = ExperimentRunner::new(config);
            runner.run_generate()?;
            println!("---");
            runner.run_evaluate()?;
            println!("---");
            runner.run_finetune()?;
            println!("---");
            println!("Pipeline complete");
        }
    }

    Ok(())
}
