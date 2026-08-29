import os
import json
from typing import List, Dict
import config._config as cfg


class ABLEDataManager:
    def __init__(
            self,
            dataset_name,
            model_name,
            dtype_str = 'float16'
    ):
        able_dir = cfg.get_ABLE_dir(dataset_name)
        self.able_path = f"{able_dir}/{model_name.replace('/', '--')}_{dtype_str}.jsonl"

    def load_computed_idx(self):
        if os.path.exists(self.able_path):
            with open(self.able_path, 'r', encoding='utf-8') as f:
                computed_idx = [json.loads(line)["index"] for line in f]
            return computed_idx
        else:
            return []

    def load_existing_results(self):
        if os.path.exists(self.able_path):
            with open(self.able_path, 'r', encoding='utf-8') as f:
                existing_results = [json.loads(line) for line in f]
            return existing_results
        else:
            return []

    def save_results(self, ables: List[Dict]):
        existing_results = self.load_existing_results()
        results_by_index = {
            result["index"]: result for result in existing_results
        }
        results_by_index.update(
            {result["index"]: result for result in ables}
        )
        with open(self.able_path, 'w', encoding='utf-8') as f:
            sorted_results = sorted(
                results_by_index.values(),
                key=lambda x: x["index"]
            )
            for result in sorted_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
