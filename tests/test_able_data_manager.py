import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.io.able_data import ABLEDataManager


class AbleDataManagerTests(unittest.TestCase):
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
