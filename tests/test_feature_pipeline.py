import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import joblib
import numpy as np

from src.process_able_features import main as feature_main


def _write_word_attributions(path: Path, scale: float) -> None:
    record = {
        "index": 0,
        "ans_idx": 0,
        "word_attrs_per_choice": [
            [scale, 2 * scale, 3 * scale],
            [4 * scale, 5 * scale, 6 * scale],
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


class FeaturePipelineTests(unittest.TestCase):
    def test_jl_cli_builds_normalized_features_for_multiple_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "word_level"
            output_dir = root / "features"
            input_dir.mkdir()

            _write_word_attributions(
                input_dir / "org--model-a_float32_word.jsonl", 1.0
            )
            _write_word_attributions(
                input_dir / "org--model-b_bfloat16_word.jsonl", 2.0
            )
            family_file = root / "model-family.csv"
            family_file.write_text(
                "model_name,model_family\n"
                "org--model-a,family-a\n"
                "org--model-b,family-b\n",
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()):
                exit_code = feature_main(
                    [
                        "--method",
                        "jl",
                        "--dim",
                        "4",
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--family-file",
                        str(family_file),
                    ]
                )

            self.assertEqual(exit_code, 0)
            for model_name, family in (
                ("org--model-a", "family-a"),
                ("org--model-b", "family-b"),
            ):
                output_file = output_dir / model_name / f"{model_name}_able.json"
                payload = json.loads(output_file.read_text(encoding="utf-8"))
                self.assertEqual(payload["model_name"], model_name)
                self.assertEqual(payload["model_family"], family)
                self.assertEqual(len(payload["able"]), 4)
                self.assertAlmostEqual(np.linalg.norm(payload["able"]), 1.0)

            self.assertTrue((output_dir / "0_jl_projector.pkl").is_file())

    def test_duplicate_model_dtypes_fail_before_writing_outputs(self):
        for existing_output in (False, True):
            with self.subTest(existing_output=existing_output), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_dir = root / "word_level"
                output_dir = root / "features"
                input_dir.mkdir()
                for model_name, dtype, scale in (
                    ("org--model-a", "float16", 1.0),
                    ("org--model-a", "float32", 2.0),
                    ("org--model-b", "float32", 3.0),
                ):
                    _write_word_attributions(
                        input_dir / f"{model_name}_{dtype}_word.jsonl", scale
                    )

                if existing_output:
                    model_dir = output_dir / "org--model-a"
                    model_dir.mkdir(parents=True)
                    (model_dir / "org--model-a_able.json").write_text(
                        '{"able": [9.0]}\n', encoding="utf-8"
                    )
                    (output_dir / "0_jl_projector.pkl").write_bytes(b"existing projector")

                before = {
                    path.relative_to(output_dir): path.read_bytes()
                    for path in output_dir.rglob("*") if path.is_file()
                }
                stdout = StringIO()
                with redirect_stdout(stdout):
                    exit_code = feature_main(
                        [
                            "--input-dir", str(input_dir),
                            "--output-dir", str(output_dir),
                            "--family-file", str(root / "missing-family.csv"),
                        ]
                    )

                after = {
                    path.relative_to(output_dir): path.read_bytes()
                    for path in output_dir.rglob("*") if path.is_file()
                }
                self.assertEqual(exit_code, 1)
                self.assertIn("Multiple input files resolve to model org--model-a", stdout.getvalue())
                self.assertIn("separate input/output directories", stdout.getvalue())
                self.assertEqual(before, after)
                self.assertEqual(output_dir.exists(), existing_output)

    def test_jl_normalization_modes_match_projection_geometry(self):
        feature_matrix = np.array(
            [[1, 2, 3, 4, 5, 6], [6, -4, 0, 3, -2, 1], [0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "word_level"
            input_dir.mkdir()
            for idx, row in enumerate(feature_matrix):
                record = {
                    "index": 0,
                    "ans_idx": 0,
                    "word_attrs_per_choice": [row[:3].tolist(), row[3:].tolist()],
                }
                (input_dir / f"org--model-{idx}_float32_word.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )

            for mode in ("pre", "post", "both", "none"):
                with self.subTest(mode=mode):
                    output_dir = root / mode
                    with redirect_stdout(StringIO()):
                        exit_code = feature_main(
                            [
                                "--method", "jl", "--dim", "3",
                                "--norm-mode", mode,
                                "--input-dir", str(input_dir),
                                "--output-dir", str(output_dir),
                                "--family-file", str(root / "missing-family.csv"),
                            ]
                        )

                    self.assertEqual(exit_code, 0)
                    projector = joblib.load(output_dir / "0_jl_projector.pkl")
                    raw_projection = feature_matrix @ projector.components_.T
                    input_lengths = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
                    projected_lengths = np.linalg.norm(raw_projection, axis=1, keepdims=True)
                    input_lengths[input_lengths == 0] = 1
                    projected_lengths[projected_lengths == 0] = 1
                    expected = {
                        "none": raw_projection,
                        "pre": raw_projection / input_lengths,
                        "post": raw_projection / projected_lengths,
                        "both": raw_projection / projected_lengths,
                    }[mode]
                    actual = []
                    for idx in range(len(feature_matrix)):
                        model_name = f"org--model-{idx}"
                        output_file = output_dir / model_name / f"{model_name}_able.json"
                        actual.append(json.loads(output_file.read_text(encoding="utf-8"))["able"])

                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_correct_only_keeps_only_the_labeled_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "word_level"
            output_dir = root / "features"
            input_dir.mkdir()
            _write_word_attributions(
                input_dir / "org--model-a_float16_word.jsonl", 1.0
            )

            with redirect_stdout(StringIO()):
                exit_code = feature_main(
                    [
                        "--method",
                        "none",
                        "--norm-mode",
                        "none",
                        "--correct-only",
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--family-file",
                        str(root / "missing-family.csv"),
                    ]
                )

            output_file = (
                output_dir / "org--model-a" / "org--model-a_able.json"
            )
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["able"], [1.0, 2.0, 3.0])
            self.assertEqual(payload["model_family"], "unknown")

    def test_empty_input_returns_a_clear_nonzero_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "empty"
            output_dir = root / "features"
            input_dir.mkdir()
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = feature_main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--family-file",
                        str(root / "missing-family.csv"),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("No word-attribution files found", stdout.getvalue())
            self.assertFalse(output_dir.exists())

    def test_inconsistent_feature_dimensions_return_a_clear_nonzero_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "word_level"
            output_dir = root / "features"
            input_dir.mkdir()
            _write_word_attributions(
                input_dir / "org--model-a_float32_word.jsonl", 1.0
            )
            short_record = {
                "index": 0,
                "ans_idx": 0,
                "word_attrs_per_choice": [[1.0, 2.0], [3.0, 4.0]],
            }
            (input_dir / "org--model-b_float32_word.jsonl").write_text(
                json.dumps(short_record) + "\n", encoding="utf-8"
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = feature_main(
                    [
                        "--method",
                        "none",
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--family-file",
                        str(root / "missing-family.csv"),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Inconsistent feature dimensions", stdout.getvalue())
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
