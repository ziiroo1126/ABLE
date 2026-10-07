import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.io.able_data import ABLEDataManager


class AbleDataManagerTests(unittest.TestCase):
    def test_failed_serialization_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.io.able_data.cfg.get_ABLE_dir", return_value=Path(tmp)):
                manager = ABLEDataManager("dataset", "org/model")
            manager.save_results([{"index": 0, "able": [1.0]}])
            original = Path(manager.able_path).read_bytes()

            with self.assertRaises(TypeError):
                manager.save_results([{"index": 1, "able": {object()}}])

            self.assertEqual(Path(manager.able_path).read_bytes(), original)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_incompatible_new_records_are_rejected_without_changing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.io.able_data.cfg.get_ABLE_dir", return_value=Path(tmp)):
                manager = ABLEDataManager(
                    "dataset", "org/model",
                    expected_metadata={"use_chat_template": True, "attribution_version": 2},
                )
            record = {"index": 0, "use_chat_template": True, "attribution_version": 2}
            manager.save_results([record])
            original = Path(manager.able_path).read_bytes()

            with self.assertRaisesRegex(ValueError, "incompatible computation metadata"):
                manager.save_results([{**record, "index": 1, "use_chat_template": False}])

            self.assertEqual(Path(manager.able_path).read_bytes(), original)

    def test_repeated_indices_are_replaced_and_results_remain_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch(
                "src.io.able_data.cfg.get_ABLE_dir", return_value=output_dir
            ):
                manager = ABLEDataManager(
                    "dataset", "org/model", dtype_str="float32"
                )

            manager.save_results(
                [
                    {"index": 2, "able": [2.0]},
                    {"index": 0, "able": [0.0]},
                ]
            )
            manager.save_results(
                [
                    {"index": 1, "able": [1.0]},
                    {"index": 2, "able": [20.0]},
                ]
            )

            self.assertEqual(manager.load_computed_idx(), [0, 1, 2])
            self.assertEqual(
                manager.load_existing_results(),
                [
                    {"index": 0, "able": [0.0]},
                    {"index": 1, "able": [1.0]},
                    {"index": 2, "able": [20.0]},
                ],
            )


if __name__ == "__main__":
    unittest.main()
