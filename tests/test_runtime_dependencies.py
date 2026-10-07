import importlib.metadata
from pathlib import Path
import tempfile
import unittest

from packaging.requirements import Requirement
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


class RuntimeDependencyTests(unittest.TestCase):
    def test_peft_is_declared_for_adapter_only_models(self):
        requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
        requirements = {
            requirement.name: requirement
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
            for requirement in [Requirement(line)]
        }

        self.assertIn("peft", requirements)
        self.assertIn(
            importlib.metadata.version("peft"), requirements["peft"].specifier
        )

    def test_auto_model_loads_local_adapter_weights_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            base_dir = Path(tmp) / "base"
            adapter_dir = Path(tmp) / "adapter"
            config = LlamaConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                tie_word_embeddings=False,
            )
            base = LlamaForCausalLM(config).eval()
            base.save_pretrained(base_dir)
            inputs = torch.tensor([[1, 2, 3, 4]])
            attention_mask = torch.ones_like(inputs)
            with torch.no_grad():
                base_logits = base(inputs, attention_mask=attention_mask).logits

            adapted = get_peft_model(
                base,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    target_modules=["v_proj"],
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0.0,
                    bias="none",
                ),
            ).eval()
            adapted.peft_config["default"].base_model_name_or_path = str(base_dir)
            with torch.no_grad():
                for name, parameter in adapted.named_parameters():
                    if "lora_" in name:
                        parameter.normal_(mean=0.0, std=0.2)
                expected_logits = adapted(inputs, attention_mask=attention_mask).logits
            expected_weights = {
                name.split("model.layers.", 1)[-1]: parameter.detach().clone()
                for name, parameter in adapted.named_parameters()
                if "lora_" in name
            }
            self.assertGreater((expected_logits - base_logits).abs().max().item(), 1e-5)
            adapted.save_pretrained(adapter_dir)

            loaded = AutoModelForCausalLM.from_pretrained(
                adapter_dir, local_files_only=True
            ).eval()
            loaded_weights = {
                name.split("model.layers.", 1)[-1]: parameter.detach()
                for name, parameter in loaded.named_parameters()
                if "lora_" in name
            }
            self.assertEqual(expected_weights.keys(), loaded_weights.keys())
            for name, expected in expected_weights.items():
                torch.testing.assert_close(loaded_weights[name], expected, rtol=0, atol=0)
            with torch.no_grad():
                actual_logits = loaded(inputs, attention_mask=attention_mask).logits
            torch.testing.assert_close(actual_logits, expected_logits)


if __name__ == "__main__":
    unittest.main()
