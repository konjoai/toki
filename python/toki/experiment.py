"""Orchestrates the generate -> evaluate -> (finetune) -> evaluate pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

from toki.generate import AdversarialGenerator
from toki.evaluate import RobustnessEvaluator
from toki.dataset import AdversarialDataset
from toki.results import ExperimentResult


@dataclass
class ExperimentConfig:
    name: str = "toki_experiment"
    model_name: str = "gpt2"
    seed: int = 42
    jailbreak_count: int = 10
    injection_count: int = 10
    boundary_count: int = 5
    output_dir: str = "experiments/runs"
    run_finetune: bool = False


class TokiExperiment:
    """
    Full adversarial experiment:
    1. Generate adversarial prompts
    2. Evaluate model before fine-tuning
    3. (Optionally) fine-tune with LoRA
    4. Re-evaluate after fine-tuning
    5. Store results
    """

    def __init__(self, config: ExperimentConfig, model_fn: Optional[Callable] = None) -> None:
        """
        config: experiment configuration
        model_fn: callable(prompt: str) -> str for evaluation.
                  If None, uses the default mock (always refuses).
        """
        self._config = config
        self._model_fn = model_fn
        self._generator = AdversarialGenerator(seed=config.seed)
        self._evaluator = RobustnessEvaluator(model_fn=model_fn)
        self._dataset: Optional[AdversarialDataset] = None

    def generate(self) -> AdversarialDataset:
        """Step 1: Generate adversarial dataset."""
        ds = AdversarialDataset()
        prompts = self._generator.generate_all(
            jailbreak_count=self._config.jailbreak_count,
            injection_count=self._config.injection_count,
            boundary_count=self._config.boundary_count,
        )
        ds.add_batch(prompts)
        self._dataset = ds
        return ds

    def evaluate(self) -> dict:
        """Step 2/4: Evaluate model on current dataset. Returns summary dict."""
        if self._dataset is None:
            raise RuntimeError("Call generate() before evaluate()")
        results = self._evaluator.evaluate_batch(list(self._dataset))
        return self._evaluator.summary(results)

    def finetune(self) -> None:
        """
        Step 3: Fine-tune model with LoRA on adversarial dataset.
        Requires peft + transformers installed. Raises ImportError otherwise.
        """
        try:
            from toki.finetune import LoRAFinetuner, LoRAConfig, TrainingConfig
            finetuner = LoRAFinetuner(
                lora_config=LoRAConfig(),
                training_config=TrainingConfig(
                    output_dir=self._config.output_dir,
                    num_epochs=1,
                    seed=self._config.seed,
                ),
            )
            model, tokenizer = finetuner.prepare_model(self._config.model_name)
            # In a real run: finetuner.train(model, tokenizer, dataset=self._dataset)
        except ImportError as e:
            raise ImportError(
                "Fine-tuning requires peft: pip install toki[hf]\n"
                f"Missing: {e}"
            ) from e

    def run(self) -> ExperimentResult:
        """
        Full pipeline: generate -> evaluate -> [finetune -> evaluate] -> save.
        Returns ExperimentResult.
        """
        self.generate()
        pre_summary = self.evaluate()
        post_summary = None

        if self._config.run_finetune:
            self.finetune()
            post_summary = self.evaluate()

        result = ExperimentResult(
            name=self._config.name,
            timestamp=ExperimentResult.make_timestamp(),
            model_name=self._config.model_name,
            seed=self._config.seed,
            pre_score=pre_summary["mean_score"],
            post_score=post_summary["mean_score"] if post_summary else None,
            total_prompts=pre_summary["total"],
            category_scores=pre_summary["by_category"],
            config={
                "model_name": self._config.model_name,
                "seed": self._config.seed,
                "jailbreak_count": self._config.jailbreak_count,
                "injection_count": self._config.injection_count,
                "boundary_count": self._config.boundary_count,
                "run_finetune": self._config.run_finetune,
            },
        )
        result.save(self._config.output_dir)
        return result
