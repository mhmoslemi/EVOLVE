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
            # One logical backbone may be sharded across every authoritative
            # generation GPU. It is still a single model and a single set of
            # three named role adapters, not one backbone per role/device.
            "device_map": "auto" if self.config.num_gpus > 1 else {"": 0},
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


class UnslothBackbone:
    """Unsloth training backend with the same named-PEFT adapter contract."""

    name = "unsloth"

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._fast_language_model = None

    def load(self):
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError(
                "backend=unsloth requires the unsloth package compatible with "
                "the installed PyTorch and CUDA versions"
            ) from exc

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
            full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            target_modules=list(self.config.target_modules),
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=False,
            loftq_config=None,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self._fast_language_model = FastLanguageModel
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def set_inference_mode(self):
        self._fast_language_model.for_inference(self.model)

    def set_training_mode(self):
        self._fast_language_model.for_training(self.model)

    def disable_adapter(self):
        disabled = self.model.disable_adapter()
        return disabled if hasattr(disabled, "__enter__") else contextlib.nullcontext()


__all__ = ["HFBackbone", "UnslothBackbone"]
