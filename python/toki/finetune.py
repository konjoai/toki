"""
LoRA fine-tuning wrapper.

Core config dataclasses are importable and testable without torch/transformers.
The ``LoRAFinetuner.prepare_model`` method requires the ``toki[hf]`` extras.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    r: int = 8                          # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Configuration for the fine-tuning training loop."""

    output_dir: str = "output"
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_seq_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    fp16: bool = False
    seed: int = 42


class LoRAFinetuner:
    """Wrapper that applies LoRA adapters to a HuggingFace causal-LM.

    Parameters
    ----------
    lora_config:
        LoRA adapter configuration.  Defaults to ``LoRAConfig()``.
    training_config:
        Training loop configuration.  Defaults to ``TrainingConfig()``.
    """

    def __init__(
        self,
        lora_config: LoRAConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) -> None:
        self._lora = lora_config or LoRAConfig()
        self._training = training_config or TrainingConfig()

    @property
    def lora_config(self) -> LoRAConfig:
        return self._lora

    @property
    def training_config(self) -> TrainingConfig:
        return self._training

    def prepare_model(self, model_name: str):
        """Load model and apply LoRA adapters.

        Requires ``pip install toki[hf]`` (torch + transformers + peft).

        Returns
        -------
        tuple[peft.PeftModel, transformers.PreTrainedTokenizer]
        """
        try:
            import torch
            from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LoRA fine-tuning requires: pip install toki[hf]\n"
                f"Missing: {exc}"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self._training.fp16 else torch.float32,
        )
        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self._lora.r,
            lora_alpha=self._lora.lora_alpha,
            lora_dropout=self._lora.lora_dropout,
            target_modules=self._lora.target_modules,
            bias=self._lora.bias,
        )
        model = get_peft_model(model, peft_config)
        return model, tokenizer

    def train(
        self,
        model,
        tokenizer,
        prompts=None,
        dataset=None,
    ) -> dict:
        """
        Fine-tune model on adversarial prompts.
        Returns dict with training_loss and num_steps.
        Requires: pip install toki[hf] (peft + datasets + transformers).

        Parameters
        ----------
        model:
            PEFT-wrapped model from prepare_model().
        tokenizer:
            HF tokenizer from prepare_model().
        prompts:
            Optional list[str] of raw text prompts.
        dataset:
            Optional AdversarialDataset; takes priority over prompts.
        """
        try:
            import torch
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            from datasets import Dataset as HFDataset
        except ImportError as e:
            raise ImportError(f"Training requires toki[hf]: {e}") from e

        # Collect text prompts
        if dataset is not None:
            texts = [p.text for p in dataset]
        elif prompts is not None:
            texts = prompts
        else:
            raise ValueError("Provide either dataset or prompts")

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self._training.max_seq_length,
                padding="max_length",
            )

        hf_dataset = HFDataset.from_dict({"text": texts})
        tokenized = hf_dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)

        training_args = TrainingArguments(
            output_dir=self._training.output_dir,
            num_train_epochs=self._training.num_epochs,
            per_device_train_batch_size=self._training.batch_size,
            learning_rate=self._training.learning_rate,
            warmup_steps=self._training.warmup_steps,
            save_steps=self._training.save_steps,
            logging_steps=self._training.logging_steps,
            fp16=self._training.fp16,
            seed=self._training.seed,
            no_cuda=True,  # always CPU in tests
            report_to=[],  # no wandb/tensorboard
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        train_result = trainer.train()
        return {
            "training_loss": train_result.training_loss,
            "num_steps": train_result.global_step,
        }

    def config_summary(self) -> dict:
        """Return a JSON-serialisable summary of current configuration."""
        return {
            "lora": {
                "r": self._lora.r,
                "alpha": self._lora.lora_alpha,
                "dropout": self._lora.lora_dropout,
                "target_modules": self._lora.target_modules,
            },
            "training": {
                "epochs": self._training.num_epochs,
                "lr": self._training.learning_rate,
                "batch_size": self._training.batch_size,
                "output_dir": self._training.output_dir,
            },
        }
