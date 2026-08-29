import yaml
from pathlib import Path
from loguru import logger
from typing import List, Optional


class ModelDataLoader:
    def __init__(self, models_path: str, num_models: Optional[int] = None):
        self.models_path = models_path
        model_list = self._load_model_names()
        if not model_list:
            raise ValueError(f"No models found in the model list: {models_path}")
        if num_models and num_models <= len(model_list):
            self.model_list = model_list[:num_models]
            logger.success(
                f"Loaded {len(self.model_list)} models from file: {models_path}"
            )
        else:
            self.model_list = model_list
            if num_models and num_models > len(model_list):
                logger.warning(
                    f"Loaded all {len(self.model_list)} models from file: {models_path}"
                )
            else:
                logger.success(
                    f"Loaded {len(self.model_list)} models from file: {models_path}"
                )

    def _load_model_names(self) -> List[str]:
        yaml_path = Path(self.models_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"The model list YAML file does not exist: {yaml_path}"
            )
        with open(yaml_path, "r", encoding="utf-8") as f:
            model_list = yaml.safe_load(f)
        if not isinstance(model_list, list):
            raise ValueError(
                "The YAML file format is incorrect: expected a list, "
                f"got {type(model_list).__name__}"
            )
        return model_list
