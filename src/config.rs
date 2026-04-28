use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokiConfig {
    pub model_name: String,
    pub output_dir: String,
    pub num_adversarial_prompts: usize,
    pub categories: Vec<String>,
    pub seed: u64,
}

impl Default for TokiConfig {
    fn default() -> Self {
        Self {
            model_name: "gpt2".to_string(),
            output_dir: "experiments/runs".to_string(),
            num_adversarial_prompts: 100,
            categories: vec![
                "jailbreak".to_string(),
                "injection".to_string(),
                "edge_case".to_string(),
                "boundary".to_string(),
            ],
            seed: 42,
        }
    }
}

impl TokiConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = TokiConfig::default();
        assert_eq!(cfg.model_name, "gpt2");
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.categories.len(), 4);
        assert_eq!(cfg.num_adversarial_prompts, 100);
    }

    #[test]
    fn test_config_roundtrip_json() {
        let cfg = TokiConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: TokiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.model_name, restored.model_name);
        assert_eq!(cfg.seed, restored.seed);
    }

    #[test]
    fn test_config_save_and_load() {
        use std::io::Write;
        let cfg = TokiConfig::default();
        let tmp = std::env::temp_dir().join("toki_test_config.json");
        let path = tmp.to_str().unwrap();
        cfg.save(path).unwrap();
        let loaded = TokiConfig::from_file(path).unwrap();
        assert_eq!(cfg.model_name, loaded.model_name);
        assert_eq!(cfg.seed, loaded.seed);
        let _ = std::fs::remove_file(path);
    }
}
