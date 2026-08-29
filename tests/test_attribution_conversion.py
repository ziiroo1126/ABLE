import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.token_to_word_attribution import (
    character_to_word_attribution,
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


class AttributionConversionTests(unittest.TestCase):
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
