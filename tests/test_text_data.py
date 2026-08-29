import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.io.text_data import TextDataLoader


class TextDataLoaderTest(unittest.TestCase):
    def test_reads_records_after_multibyte_utf8_text(self):
        records = [
            {"index": 0, "question": "ASCII", "choices": ["A", "B"], "ans_idx": 0},
            {"index": 1, "question": "café 中文", "choices": ["A", "B"], "ans_idx": 1},
            {"index": 2, "question": "final", "choices": ["A", "B"], "ans_idx": 0},
        ]

        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "probe.jsonl"
            dataset_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )

            with patch(
                "src.io.text_data.cfg.get_texts_path",
                return_value=dataset_path,
            ):
                loader = TextDataLoader("probe")
                loaded = loader.get_texts([0, 1, 2])

        self.assertEqual([row[0] for row in loaded], [0, 1, 2])
        self.assertEqual(loaded[1][1], "café 中文")
        self.assertEqual(loaded[2][1], "final")


if __name__ == "__main__":
    unittest.main()
