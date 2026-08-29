import os
import fcntl
from loguru import logger
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def file_lock(file_path):
    with open(file_path, 'a+', encoding='utf-8') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

class LoggerABLE:
    def __init__(self, log_dir: str, log_name: str):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_name)
        self._ensure_files_exist()
        self.write(f"Starting to log to file: {self.log_path}")

    def _ensure_files_exist(self):
        os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with file_lock(self.log_path) as f:
                pass

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_finish(self, model_name: str, message: str):
        timestamp = self._now()
        log_entry = f"[{timestamp}] {message} | {model_name}"
        with file_lock(self.log_path) as f:
            f.write(log_entry + "\n")
        logger.success(f"{message} | {model_name}")

    def log_error(self, model_name: str, error_status: str, error_message: str):
        timestamp = self._now()
        log_entry = f"[{timestamp}] {error_status} | {model_name} | {repr(error_message)}"
        with file_lock(self.log_path) as f:
            f.write(log_entry + "\n")
        logger.error(f"{error_status} | {model_name} | {repr(error_message)}")

    def write(self, message: str):
        with file_lock(self.log_path) as f:
            f.write(message + "\n")
        logger.info(message)
