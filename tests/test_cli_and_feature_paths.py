import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.calculate_able import (
    DTYPES,
    main as calculate_main,
    parse_args as parse_calculate_args,
)
from src.process_able_features import (
    model_name_from_word_file,
    parse_args as parse_feature_args,
)
from src.token_to_word_attribution import parse_cli_args as parse_conversion_args


class CalculateAbleCliTest(unittest.TestCase):
    def test_public_defaults_allow_hub_download_without_remote_code(self):
        args = parse_calculate_args(["example_models"])

        self.assertFalse(args.local_files_only)
        self.assertFalse(args.trust_remote_code)
        self.assertEqual(DTYPES[args.dtype], torch.bfloat16)

    def test_smoke_test_arguments(self):
        args = parse_calculate_args(
            ["example_models", "--max-samples", "5", "--dtype", "float32"]
        )

        self.assertEqual(args.max_samples, 5)
        self.assertEqual(DTYPES[args.dtype], torch.float32)

    @patch("src.calculate_able.RunnerABLE")
    def test_returns_nonzero_when_a_model_fails(self, runner_class):
        runner_class.return_value.run.return_value = False

        exit_code = calculate_main(["example_models", "--max-samples", "1"])

        self.assertEqual(exit_code, 1)


class FeaturePathTest(unittest.TestCase):
    def test_default_family_file_exists_in_public_layout(self):
        args = parse_feature_args([])
        self.assertEqual(args.family_file, "./data/models/model-family.csv")

    def test_word_filename_supports_all_cli_dtypes(self):
        expected = "meta-llama--Llama-2-7b-hf"
        for dtype in ("bfloat16", "float16", "float32"):
            with self.subTest(dtype=dtype):
                filename = f"{expected}_{dtype}_word.jsonl"
                self.assertEqual(model_name_from_word_file(filename), expected)


class AttributionConversionCliTest(unittest.TestCase):
    def test_tokenizer_loading_options_are_explicit(self):
        required = ["--input-dir", "input", "--dataset-path", "dataset.jsonl"]

        defaults = parse_conversion_args(required)
        configured = parse_conversion_args(
            required
            + [
                "--cache-dir",
                "/tmp/able-hf-cache",
                "--local-files-only",
                "--trust-remote-code",
            ]
        )

        self.assertIsNone(defaults.cache_dir)
        self.assertFalse(defaults.trust_remote_code)
        self.assertFalse(defaults.local_files_only)
        self.assertEqual(configured.cache_dir, Path("/tmp/able-hf-cache"))
        self.assertTrue(configured.local_files_only)
        self.assertTrue(configured.trust_remote_code)


if __name__ == "__main__":
    unittest.main()
