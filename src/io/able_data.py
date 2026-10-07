import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import config._config as cfg


class ABLEDataManager:
    def __init__(
            self,
            dataset_name,
            model_name,
            dtype_str = 'float16',
            expected_metadata: Optional[Dict] = None,
    ):
        able_dir = cfg.get_ABLE_dir(dataset_name)
        self.able_path = f"{able_dir}/{model_name.replace('/', '--')}_{dtype_str}.jsonl"
        self.expected_metadata = dict(expected_metadata or {})

    def _is_compatible(self, result: Dict) -> bool:
        return all(
            key in result and result[key] == value
            for key, value in self.expected_metadata.items()
        )

    def load_computed_idx(self):
        return [result["index"] for result in self.load_existing_results()]

    def load_existing_results(self):
        if os.path.exists(self.able_path):
            with open(self.able_path, 'r', encoding='utf-8') as f:
                existing_results = [json.loads(line) for line in f]
            return [result for result in existing_results if self._is_compatible(result)]
        else:
            return []

    def save_results(self, ables: List[Dict]):
        if any(not self._is_compatible(result) for result in ables):
            raise ValueError("Cannot save attributions with incompatible computation metadata")
        existing_results = self.load_existing_results()
        results_by_index = {
            result["index"]: result for result in existing_results
        }
        results_by_index.update(
            {result["index"]: result for result in ables}
        )
        # Keep the previous run intact until the replacement is fully serialized.
        path = Path(self.able_path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.tmp', delete=False,
            ) as f:
                temporary_path = Path(f.name)
                for result in sorted(results_by_index.values(), key=lambda x: x["index"]):
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
