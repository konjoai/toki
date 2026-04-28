use crate::config::TokiConfig;
use anyhow::Result;

pub struct ExperimentRunner {
    config: TokiConfig,
}

impl ExperimentRunner {
    pub fn new(config: TokiConfig) -> Self {
        Self { config }
    }

    pub fn run_generate(&self) -> Result<()> {
        println!("Toki - Adversarial Generation");
        println!("Model: {}", self.config.model_name);
        println!("Categories: {}", self.config.categories.join(", "));
        println!("Prompts per category: {}", self.config.num_adversarial_prompts);
        println!("Seed: {}", self.config.seed);
        println!();
        // In production, this calls the Python generate module via subprocess
        println!(
            "Run: python -m toki generate --model {} --seed {}",
            self.config.model_name, self.config.seed
        );
        Ok(())
    }

    pub fn run_evaluate(&self) -> Result<()> {
        println!("Toki - Robustness Evaluation");
        println!("Model: {}", self.config.model_name);
        println!();
        println!(
            "Run: python -m toki evaluate --model {} --seed {}",
            self.config.model_name, self.config.seed
        );
        Ok(())
    }

    pub fn run_finetune(&self) -> Result<()> {
        println!("Toki - LoRA Fine-tuning");
        println!("Model: {}", self.config.model_name);
        println!("Output: {}", self.config.output_dir);
        println!();
        println!(
            "Run: python -m toki finetune --model {} --output {}",
            self.config.model_name, self.config.output_dir
        );
        Ok(())
    }

    pub fn show_config(&self) {
        let json = serde_json::to_string_pretty(&self.config).unwrap_or_default();
        println!("{}", json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_new() {
        let cfg = TokiConfig::default();
        let runner = ExperimentRunner::new(cfg);
        // show_config should not panic
        runner.show_config();
    }

    #[test]
    fn test_runner_run_generate() {
        let runner = ExperimentRunner::new(TokiConfig::default());
        assert!(runner.run_generate().is_ok());
    }

    #[test]
    fn test_runner_run_evaluate() {
        let runner = ExperimentRunner::new(TokiConfig::default());
        assert!(runner.run_evaluate().is_ok());
    }

    #[test]
    fn test_runner_run_finetune() {
        let runner = ExperimentRunner::new(TokiConfig::default());
        assert!(runner.run_finetune().is_ok());
    }
}
