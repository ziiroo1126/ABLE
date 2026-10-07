import unittest
from unittest.mock import patch

import torch

from src.calculator.able import ABLECalculator
from src.prompting import ATTRIBUTION_VERSION


class _TokenizerWithSharedPadAndEos:
    pad_token_id = 2
    eos_token_id = 2
    chat_template = None

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        if text.endswith("A"):
            input_ids = torch.tensor([[1, 2, 3, 4]])
            offsets = torch.tensor([[[0, 1], [1, 2], [2, 3], [2, 3]]])
        else:
            input_ids = torch.tensor([[1, 2, 3]])
            offsets = torch.tensor([[[0, 1], [1, 2], [2, 3]]])

        encoded = {"input_ids": input_ids}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


class _RecordingCausalModel:
    def __init__(self):
        self.attention_masks = []

    def __call__(self, input_ids, attention_mask, use_cache=False):
        self.attention_masks.append(attention_mask.detach().cpu())
        logits = torch.zeros((*input_ids.shape, 6), device=input_ids.device)
        return type("ModelOutput", (), {"logits": logits})()

    def get_input_embeddings(self):
        return object()

    def zero_grad(self):
        return None


class _MetaOutputCausalModel:
    def __call__(self, input_ids, attention_mask, use_cache=False):
        logits = torch.empty((*input_ids.shape, 6), device="meta")
        return type("ModelOutput", (), {"logits": logits})()


class _CaptumLayerAdapter:
    def __init__(self, forward_func, layer):
        self.forward_func = forward_func

    def attribute(
        self,
        inputs,
        additional_forward_args,
        attribute_to_layer_input=False,
    ):
        return torch.ones((*inputs.shape, 2), device=inputs.device)


class _ReasoningChatTokenizer:
    chat_template = "reasoning-template"

    def __init__(self, supports_offsets=True):
        self.calls = []
        self.assistant_template_calls = 0
        self.supports_offsets = supports_offsets

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = f"<bos>user\n{messages[0]['content']}\nassistant\n"
        if messages[-1]["role"] == "assistant":
            self.assistant_template_calls += 1
            return prompt + "<think>\n\n</think>\n\n" + messages[-1]["content"]
        return prompt

    def __call__(
        self, text, return_tensors=None, return_offsets_mapping=False,
        add_special_tokens=True,
    ):
        self.calls.append((text, add_special_tokens))
        if return_offsets_mapping and not self.supports_offsets:
            raise NotImplementedError("Slow tokenizer does not provide offsets")
        ids = [ord(char) for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids.insert(0, 0)
            offsets.insert(0, (0, 0))
        encoded = {"input_ids": torch.tensor([ids])}
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor([offsets])
        return encoded


class _TokenPreferenceModel:
    def __call__(self, input_ids, attention_mask, use_cache=False):
        logits = torch.arange(128, dtype=torch.float32).expand(*input_ids.shape, 128)
        return type("ModelOutput", (), {"logits": logits})()


class AbleCalculatorTests(unittest.TestCase):
    def test_chat_scoring_targets_choices_without_inserting_reasoning_or_duplicate_bos(self):
        for supports_offsets in (True, False):
            with self.subTest(supports_offsets=supports_offsets):
                calculator = ABLECalculator.__new__(ABLECalculator)
                tokenizer = _ReasoningChatTokenizer(supports_offsets)
                calculator.tokenizer = tokenizer
                calculator.apply_chat_template = True
                calculator.is_chat_model = True
                calculator.max_length = 128
                calculator.model = _TokenPreferenceModel()

                (_, tensors, prompt_length, _, choice_lengths), = calculator._tokenize(
                    [(0, "Pick an option", ["A", "B"], 0)]
                )
                expected_scores = torch.log_softmax(
                    torch.arange(128, dtype=torch.float32), dim=0
                )
                self.assertEqual(choice_lengths, [1, 1])
                for choice, tokens, length in zip(
                    ["A", "B"], tensors, choice_lengths
                ):
                    self.assertEqual(
                        tokens[0, prompt_length:].tolist(), [ord(choice)]
                    )
                    actual_score = calculator._model_forward(
                        tokens,
                        torch.tensor([length]),
                        prompt_length,
                        torch.ones_like(tokens),
                    )
                    self.assertAlmostEqual(
                        actual_score.item(), expected_scores[ord(choice)].item()
                    )
                self.assertEqual(tokenizer.assistant_template_calls, 0)
                self.assertTrue(
                    all(not add_special for _, add_special in tokenizer.calls)
                )

    def test_choice_scoring_aligns_inputs_with_a_different_output_device(self):
        calculator = ABLECalculator.__new__(ABLECalculator)
        calculator.model = _MetaOutputCausalModel()
        input_ids = torch.tensor([[1, 2, 3]])

        scores = calculator._model_forward(
            input_ids=input_ids,
            choice_len_tensor=torch.tensor([1]),
            question_len=2,
            attention_mask=torch.ones_like(input_ids),
        )

        self.assertEqual(scores.device.type, "meta")

    def test_batch_mask_keeps_real_eos_when_eos_is_also_the_pad_token(self):
        calculator = ABLECalculator.__new__(ABLECalculator)
        calculator.apply_chat_template = False
        calculator.is_chat_model = False
        calculator.tokenizer = _TokenizerWithSharedPadAndEos()
        model = _RecordingCausalModel()
        calculator.model = model
        calculator.input_device = torch.device("cpu")
        calculator.pad_id = 2
        calculator.run_batch = True
        calculator.max_length = 32

        with patch(
            "src.calculator.able.LayerGradientXActivation",
            _CaptumLayerAdapter,
        ):
            results = calculator([(0, "Q", ["A", "B"], 0)])

        self.assertEqual(results[0]["attribution_version"], ATTRIBUTION_VERSION)
        self.assertFalse(results[0]["use_chat_template"])

        self.assertEqual(
            [[1, 1, 1, 1], [1, 1, 1, 0]],
            model.attention_masks[0].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
