import csv
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

from src.io.model_data import ModelDataLoader


class ModelMetadataTests(unittest.TestCase):
    def test_model_list_requires_a_yaml_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "mapping.yaml"
            model_file.write_text("model: org/name\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected a list"):
                ModelDataLoader(str(model_file))

    def test_missing_model_list_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_file = Path(tmp) / "missing.yaml"

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                ModelDataLoader(str(missing_file))

    def test_empty_model_list_fails_under_optimized_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "empty.yaml"
            model_file.write_text("[]\n", encoding="utf-8")
            code = (
                "from src.io.model_data import ModelDataLoader; "
                f"ModelDataLoader({str(model_file)!r})"
            )

            result = subprocess.run(
                [sys.executable, "-O", "-c", code],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError: No models found", result.stderr)

    def test_every_paper_model_has_exactly_one_family_mapping(self):
        model_ids = yaml.safe_load(
            Path("data/models/model_list.yaml").read_text(encoding="utf-8")
        )
        expected_names = {model_id.replace("/", "--") for model_id in model_ids}

        with Path("data/models/model-family.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        name_counts = Counter(row["model_name"] for row in rows)
        duplicate_names = sorted(
            name for name, count in name_counts.items() if count != 1
        )
        actual_names = set(name_counts)

        self.assertEqual([], duplicate_names, "duplicate family mappings")
        self.assertEqual(
            expected_names,
            actual_names,
            "family map must exactly cover the final paper model list; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}",
        )


if __name__ == "__main__":
    unittest.main()
