"""HF backbone loader used by the EVOLVE named-adapter runtime."""

from __future__ import annotations

import contextlib


class HFBackbone:
    name = "hf"

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_options = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "trust_remote_code": True,
        }
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **model_options
        )
        if self.config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        adapter_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, adapter_config)
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def set_inference_mode(self):
        self.model.eval()

    def set_training_mode(self):
        self.model.train()

    def disable_adapter(self):
        disabled = self.model.disable_adapter()
        return disabled if hasattr(disabled, "__enter__") else contextlib.nullcontext()


__all__ = ["HFBackbone"]
