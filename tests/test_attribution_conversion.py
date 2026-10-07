import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.calculator.able import ABLECalculator
from src.prompting import ATTRIBUTION_VERSION
from src.token_to_word_attribution import (
    character_to_word_attribution,
    get_question_offset_mapping,
    load_jsonl,
    main as conversion_main,
    parse_cli_args as parse_conversion_args,
    process_directory,
    save_jsonl,
    token_to_character_attribution,
)


class _OffsetRow:
    def __init__(self, offsets):
        self._offsets = offsets

    def tolist(self):
        return self._offsets


class _OffsetBatch:
    def __init__(self, offsets):
        self._offsets = offsets

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return _OffsetRow(self._offsets)


class _TokenizerAdapter:
    """Small adapter for the external tokenizer offset-mapping contract."""

    def __init__(self, offsets):
        self._offsets = offsets

    def __call__(self, *_args, **_kwargs):
        return {"offset_mapping": _OffsetBatch(self._offsets)}


class _CharacterChatTokenizer:
    chat_template = "reasoning-template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = f"<bos>user\n{messages[0]['content']}\nassistant\n"
        if messages[-1]["role"] == "assistant":
            return prompt + "<think>\n\n</think>\n\n" + messages[-1]["content"]
        return prompt

    def __call__(
        self, text, return_tensors=None, return_offsets_mapping=False,
        add_special_tokens=True,
    ):
        self.calls.append((text, add_special_tokens))
        offsets = [(index, index + 1) for index in range(len(text))]
        ids = [ord(char) for char in text]
        if add_special_tokens:
            offsets.insert(0, (0, 0))
            ids.insert(0, 0)
        return {
            "input_ids": torch.tensor([ids]),
            "offset_mapping": torch.tensor([offsets]),
        }


class AttributionConversionTests(unittest.TestCase):
    def test_chat_conversion_uses_the_same_prompt_tokens_as_calculator(self):
        tokenizer = _CharacterChatTokenizer()
        calculator = ABLECalculator.__new__(ABLECalculator)
        calculator.tokenizer = tokenizer
        calculator.apply_chat_template = True
        calculator.is_chat_model = True
        calculator.max_length = 128
        question, choices = "Pick one", ["A", "B"]
        (_, tensors, prompt_length, _, _), = calculator._tokenize(
            [(0, question, choices, 0)]
        )
        token_attrs = [float(index + 1) for index in range(prompt_length)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            dataset_path = root / "dataset.jsonl"
            save_jsonl(
                [dict(index=0, question=question, choices=choices, ans_idx=0)],
                str(dataset_path),
            )
            save_jsonl(
                [dict(
                    index=0,
                    ans_idx=0,
                    input_attrs=[token_attrs, token_attrs],
                    use_chat_template=True,
                    attribution_version=ATTRIBUTION_VERSION,
                )],
                str(input_dir / "org--model_float32.jsonl"),
            )
            tokenizer.calls.clear()
            with patch(
                "src.token_to_word_attribution.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ):
                succeeded = process_directory(
                    str(input_dir), str(dataset_path), apply_chat_template=True
                )
            output = load_jsonl(
                str(input_dir / "word_level" / "org--model_float32_word.jsonl")
            )

        self.assertTrue(succeeded)
        self.assertEqual(len(tokenizer.calls), len(choices))
        for (text, add_special_tokens), ids, word_attrs in zip(
            tokenizer.calls, tensors, output[0]["word_attrs_per_choice"]
        ):
            self.assertEqual([ord(char) for char in text], ids[0].tolist())
            self.assertFalse(add_special_tokens)
            self.assertNotIn("<think>", text)
            self.assertAlmostEqual(sum(word_attrs), sum(token_attrs))

    def test_conversion_rejects_mismatched_modes_and_legacy_chat_results(self):
        cases = [
            ({}, True),
            ({"use_chat_template": False}, True),
            (
                {"use_chat_template": True, "attribution_version": ATTRIBUTION_VERSION},
                False,
            ),
            ({"use_chat_template": True}, True),
            ({"use_chat_template": True, "attribution_version": 1}, True),
        ]
        for metadata, requested_chat in cases:
            with self.subTest(metadata=metadata, requested_chat=requested_chat):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    input_dir = root / "input"
                    input_dir.mkdir()
                    dataset_path = root / "dataset.jsonl"
                    save_jsonl(
                        [dict(index=0, question="Q", choices=["A"], ans_idx=0)],
                        str(dataset_path),
                    )
                    save_jsonl(
                        [dict(index=0, input_attrs=[[1.0]], **metadata)],
                        str(input_dir / "org--model_float32.jsonl"),
                    )
                    with patch(
                        "src.token_to_word_attribution.AutoTokenizer.from_pretrained",
                        return_value=_CharacterChatTokenizer(),
                    ):
                        succeeded = process_directory(
                            str(input_dir), str(dataset_path),
                            apply_chat_template=requested_chat,
                        )
                    self.assertFalse(succeeded)
                    self.assertFalse(
                        (input_dir / "word_level" / "org--model_float32_word.jsonl").exists()
                    )

    def test_conversion_refreshes_existing_output_for_added_samples_and_changed_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            dataset_path = root / "dataset.jsonl"
            source = input_dir / "org--model_float32.jsonl"
            output = input_dir / "word_level" / "org--model_float32_word.jsonl"
            dataset = [
                dict(index=index, question="Hi all", choices=["A"], ans_idx=0)
                for index in (0, 1)
            ]
            first = dict(index=0, ans_idx=0, input_attrs=[[1.0, 2.0, 3.0, 4.0, 5.0]])
            second = dict(index=1, ans_idx=0, input_attrs=[[2.0] * 5])
            save_jsonl(dataset, str(dataset_path))
            save_jsonl([first], str(source))
            tokenizer = _TokenizerAdapter(
                [(0, 0), (0, 2), (2, 3), (3, 6), (6, 7), (7, 8)]
            )
            with patch(
                "src.token_to_word_attribution.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ):
                self.assertTrue(process_directory(str(input_dir), str(dataset_path)))
                self.assertEqual(len(load_jsonl(str(output))), 1)
                save_jsonl([first, second], str(source))
                self.assertTrue(process_directory(str(input_dir), str(dataset_path)))
                self.assertEqual(
                    [item["index"] for item in load_jsonl(str(output))], [0, 1]
                )
                dataset[0]["question"] = "H iall"
                save_jsonl(dataset, str(dataset_path))
                self.assertTrue(process_directory(
                    str(input_dir), str(dataset_path), skip_existing=True
                ))
            self.assertEqual(
                load_jsonl(str(output))[0]["word_attrs_per_choice"], [[3.0, 12.0]]
            )

    def test_choice_token_overlapping_the_prompt_boundary_is_excluded_from_question(self):
        tokenizer = _TokenizerAdapter([(0, 1), (1, 3)])
        self.assertEqual(
            get_question_offset_mapping("Q\n", "Q\nA", tokenizer), [(0, 1)]
        )

    def test_jsonl_write_preserves_existing_output_when_serialization_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_file = root / "converted.jsonl"
            original = '{"status": "complete"}\n'
            output_file.write_text(original, encoding="utf-8")

            with self.assertRaises(TypeError):
                save_jsonl(
                    [{"index": 0}, {"index": 1, "invalid": object()}],
                    str(output_file),
                )

            self.assertEqual(output_file.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob(".converted.jsonl.*.tmp")), [])

    def test_cli_only_supports_sum_aggregation(self):
        required = ["--input-dir", "input", "--dataset-path", "dataset.jsonl"]

        args = parse_conversion_args(required)
        self.assertFalse(hasattr(args, "aggregation"))

        with self.assertRaises(SystemExit):
            parse_conversion_args(required + ["--aggregation", "mean"])

    def test_cli_returns_nonzero_when_no_attribution_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "empty"
            input_dir.mkdir()
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("", encoding="utf-8")

            exit_code = conversion_main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--dataset-path",
                    str(dataset_path),
                ]
            )

        self.assertEqual(1, exit_code)

    def test_cli_returns_nonzero_when_a_tokenizer_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "token_level"
            input_dir.mkdir()
            (input_dir / "org--missing-model_float32.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("", encoding="utf-8")

            with patch(
                "src.token_to_word_attribution.AutoTokenizer.from_pretrained",
                side_effect=RuntimeError("tokenizer unavailable"),
            ):
                exit_code = conversion_main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--dataset-path",
                        str(dataset_path),
                    ]
                )

        self.assertEqual(1, exit_code)

    def test_cli_returns_nonzero_when_an_attribution_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "token_level"
            input_dir.mkdir()
            (input_dir / "org--model_float32.jsonl").write_text(
                "not-json\n", encoding="utf-8"
            )
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("", encoding="utf-8")

            with patch(
                "src.token_to_word_attribution.AutoTokenizer.from_pretrained",
                return_value=_TokenizerAdapter([]),
            ):
                exit_code = conversion_main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--dataset-path",
                        str(dataset_path),
                    ]
                )

        self.assertEqual(1, exit_code)

    def test_cli_returns_nonzero_when_attributions_do_not_match_the_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "token_level"
            input_dir.mkdir()
            (input_dir / "org--model_float32.jsonl").write_text(
                json.dumps({"index": 7, "input_attrs": []}) + "\n",
                encoding="utf-8",
            )
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("", encoding="utf-8")

            with patch(
                "src.token_to_word_attribution.AutoTokenizer.from_pretrained",
                return_value=_TokenizerAdapter([]),
            ):
                exit_code = conversion_main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--dataset-path",
                        str(dataset_path),
                    ]
                )

            output_file = (
                input_dir / "word_level" / "org--model_float32_word.jsonl"
            )
            output_exists = output_file.exists()

        self.assertEqual(1, exit_code)
        self.assertFalse(output_exists)

    def test_token_to_word_conversion_preserves_total_attribution(self):
        question = "Hi all\n"
        tokenizer = _TokenizerAdapter(
            [
                (0, 0),  # special token
                (0, 2),  # Hi
                (2, 3),  # space
                (3, 6),  # all
                (6, 7),  # newline
                (7, 8),  # answer choice, excluded from the question
            ]
        )
        token_attrs = [1.0, 2.0, 3.0, 4.0, 5.0]

        char_attrs = token_to_character_attribution(
            question,
            question + "A",
            token_attrs,
            tokenizer,
        )
        words, word_attrs = character_to_word_attribution(question, char_attrs)

        self.assertEqual(words, ["Hi", "all"])
        self.assertAlmostEqual(sum(char_attrs), sum(token_attrs))
        self.assertAlmostEqual(sum(word_attrs), sum(token_attrs))
        self.assertEqual(word_attrs, [6.0, 9.0])

    def test_token_conversion_rejects_mismatched_attribution_length(self):
        question = "Hi\n"
        tokenizer = _TokenizerAdapter([(0, 2), (2, 3), (3, 4)])

        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            token_to_character_attribution(
                question,
                question + "A",
                [1.0],
                tokenizer,
            )

    def test_directory_conversion_joins_dataset_and_attributions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "token_level"
            output_dir = root / "word_level"
            cache_dir = root / "huggingface-cache"
            input_dir.mkdir()
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "index": 7,
                        "question": "Hi all",
                        "choices": ["A", "B"],
                        "ans_idx": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            attribution_file = input_dir / "org--model_float32.jsonl"
            attribution_file.write_text(
                json.dumps(
                    {
                        "index": 7,
                        "ans_idx": 1,
                        "input_attrs": [
                            [1.0, 2.0, 3.0, 4.0, 5.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tokenizer = _TokenizerAdapter(
                [(0, 0), (0, 2), (2, 3), (3, 6), (6, 7), (7, 8)]
            )

            with patch(
                "src.token_to_word_attribution.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer:
                process_directory(
                    str(input_dir),
                    str(dataset_path),
                    str(output_dir),
                    cache_dir=str(cache_dir),
                )

            load_tokenizer.assert_called_once_with(
                "org/model",
                cache_dir=str(cache_dir),
                local_files_only=False,
                trust_remote_code=False,
            )
            output_file = output_dir / "org--model_float32_word.jsonl"
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["index"], 7)
            self.assertEqual(payload["ans_idx"], 1)
            self.assertEqual(
                payload["word_attrs_per_choice"],
                [[6.0, 9.0], [3.0, 2.0]],
            )


if __name__ == "__main__":
    unittest.main()
