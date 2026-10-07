import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.io.able_data import ABLEDataManager
from src.prompting import ATTRIBUTION_VERSION
from src.runner import RunnerABLE


class RunnerCacheTests(unittest.TestCase):
    def _run_in_workspace(self, root, runs, existing=None, fail=False):
        dataset = root / "dataset.jsonl"
        dataset.write_text("".join(
            json.dumps({"index": i, "question": "Q", "choices": ["A", "B"], "ans_idx": 0}) + "\n"
            for i in range(2)
        ), encoding="utf-8")
        models = root / "models.yaml"
        models.write_text("- org/model\n", encoding="utf-8")
        calls = []

        class Calculator:
            def __init__(self, **kwargs):
                self.chat = kwargs["apply_chat_template"]

            def __call__(self, texts):
                calls.append((self.chat, [row[0] for row in texts]))
                if fail:
                    raise RuntimeError("Calculation failed")
                return [
                    {"index": row[0], "use_chat_template": self.chat,
                     "attribution_version": ATTRIBUTION_VERSION, "input_attrs": [[1.0], [2.0]]}
                    for row in texts
                ]

        with (
            patch("src.runner.get_models_path", return_value=models),
            patch("src.io.text_data.cfg.get_texts_path", return_value=dataset),
            patch("src.io.able_data.cfg.get_ABLE_dir", return_value=root),
            patch("src.runner.get_log_ABLE_dir", return_value=root / "logs"),
            patch("src.runner.ABLECalculator", Calculator),
        ):
            manager = ABLEDataManager("dataset", "org/model", "bfloat16")
            if existing is not None:
                manager.save_results(existing)
            statuses = []
            for chat, indices in runs:
                runner = RunnerABLE(
                    textdata_name="dataset", model_list_name="models",
                    apply_chat_template=chat, textindex_list=indices,
                )
                statuses.append(runner.run())
            return calls, statuses, manager.load_existing_results()

    def test_mode_change_recomputes_and_does_not_merge_incompatible_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, statuses, records = self._run_in_workspace(
                Path(tmp), [(False, [0, 1]), (True, [0]), (True, [0, 1]), (True, [0, 1])]
            )
        self.assertEqual(statuses, [True, True, True, True])
        self.assertEqual(calls, [(False, [0, 1]), (True, [0]), (True, [1])])
        self.assertEqual([record["index"] for record in records], [0, 1])
        self.assertTrue(all(record["use_chat_template"] for record in records))

    def test_previous_attribution_versions_are_recomputed(self):
        for version in (None, ATTRIBUTION_VERSION - 1):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                existing = {"index": 0, "use_chat_template": True}
                if version is not None:
                    existing["attribution_version"] = version
                calls, statuses, records = self._run_in_workspace(
                    Path(tmp), [(True, [0])], existing=[existing]
                )
                self.assertEqual(calls, [(True, [0])])
                self.assertEqual(statuses, [True])
                self.assertEqual(records[0]["attribution_version"], ATTRIBUTION_VERSION)

    def test_failed_recomputation_keeps_existing_output(self):
        existing = [{"index": 0, "use_chat_template": False, "attribution_version": ATTRIBUTION_VERSION}]
        with tempfile.TemporaryDirectory() as tmp:
            calls, statuses, records = self._run_in_workspace(
                Path(tmp), [(True, [0])], existing=existing, fail=True
            )
        self.assertEqual(calls, [(True, [0])])
        self.assertEqual(statuses, [False])
        self.assertEqual(records, existing)


if __name__ == "__main__":
    unittest.main()
