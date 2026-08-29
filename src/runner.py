import torch
from datetime import datetime
from loguru import logger
from typing import Optional, List
from .io.text_data import TextDataLoader
from .io.able_data import ABLEDataManager
from .io.model_data import ModelDataLoader
from .logging.logger import LoggerABLE
from .calculator.able import ABLECalculator
from config._config import get_log_ABLE_dir, get_models_path


class RunnerABLE:
    def __init__(
        self,
        textdata_name: str,
        apply_chat_template: bool,
        model_list_name: str,
        dtype: torch.dtype = torch.bfloat16,
        num_models: Optional[int] = None,
        log_name: Optional[str] = None,
        textindex_list: Optional[list] = None,
        models_dir: Optional[str] = None,
        trust_remote_code: Optional[bool] = True,
        local_files_only: Optional[bool] = True,
        run_batch: Optional[bool] = False,
    ):
        self.apply_chat_template = apply_chat_template
        self.run_batch = run_batch

        self.textdata_name = textdata_name
        self.model_list_name = model_list_name
        self.modelname_list = ModelDataLoader(
            get_models_path(model_list_name), num_models
        ).model_list
        self.dtype = dtype
        self.dtype_str = str(dtype).split(".")[-1]
        self.textindex_list = textindex_list
        self.models_dir = models_dir
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.log_name = log_name

        # Load texts
        loader = TextDataLoader(textdata_name)
        if textindex_list is None:
            self.textindex_list = list(range(len(loader.index_map)))
        self.texts = loader.get_texts(self.textindex_list)

        # Set up logger
        log_dir = get_log_ABLE_dir()
        if self.log_name is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_name = f"{textdata_name}_{now}.log"
        self.logger = LoggerABLE(log_dir, self.log_name)

    def _get_uncomputed_texts(self, model_name: str):
        data_manager = ABLEDataManager(self.textdata_name, model_name, self.dtype_str)
        computed_idx = data_manager.load_computed_idx()
        uncomputed_idx = sorted(list[int](set(self.textindex_list) - set(computed_idx)))
        if not uncomputed_idx:
            return None
        else:
            uncomputed_texts = [
                idx_text for idx_text in self.texts if idx_text[0] in uncomputed_idx
            ]
            return uncomputed_texts

    def _compute_able(self, model_name: str) -> bool:
        data_manager = ABLEDataManager(self.textdata_name, model_name, self.dtype_str)
        texts_to_process = self._get_uncomputed_texts(model_name)
        if texts_to_process is None:
            self.logger.log_finish(model_name, "FINISH (already computed)")
            return True
        else:
            try:
                caculator = ABLECalculator(
                    model_name=model_name,
                    dtype=self.dtype,
                    cache_dir=self.models_dir,
                    local_files_only=self.local_files_only,
                    trust_remote_code=self.trust_remote_code,
                    apply_chat_template=self.apply_chat_template,
                    run_batch=self.run_batch,
                )
                results = caculator(texts_to_process)
                data_manager.save_results(results)
                self.logger.log_finish(model_name, "FINISH")
                return True
            except Exception as e:
                self.logger.log_error(model_name, "ERROR", str(e))
                return False

    def _write_header(self):
        self.logger.write(f"------------------------")
        self.logger.write(f"Textdata: {self.textdata_name}")
        self.logger.write(f"Model list name: {self.model_list_name}")
        self.logger.write(f"Number of texts: {len(self.texts)}")
        self.logger.write(f"Number of models: {len(self.modelname_list)}")
        self.logger.write(f"Dtype: {self.dtype}")
        self.logger.write(f"Apply chat template: {self.apply_chat_template}")
        self.logger.write(f"Run batch: {self.run_batch}")
        self.logger.write(f"Log name: {self.log_name}")
        self.logger.write(f"------------------------")

    def run(self) -> bool:
        self._write_header()
        succeeded = True
        for model_name in self.modelname_list:
            logger.info(f"Computing ABLE for model: {model_name}")
            succeeded = self._compute_able(model_name) and succeeded
        return succeeded
