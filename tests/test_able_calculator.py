import unittest
from unittest.mock import patch

import torch

from src.calculator.able import ABLECalculator


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


class AbleCalculatorTests(unittest.TestCase):
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
            calculator([(0, "Q", ["A", "B"], 0)])

        self.assertEqual(
            [[1, 1, 1, 1], [1, 1, 1, 0]],
            model.attention_masks[0].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
