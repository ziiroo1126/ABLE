import argparse
from pathlib import Path
from typing import Sequence

import torch

from . import RunnerABLE


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token-level ABLE attributions for one or more models."
    )
    parser.add_argument(
        "model_list_name",
        type=str,
        help="Model-list basename under data/models (for example, example_models)",
    )
    parser.add_argument("--log-name", "--log_name", dest="log_name", default=None)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ABLE_dataset_public_1000",
        help=(
            "Probe dataset basename under data/selected_data. The public default "
            "excludes GPQA; use ABLE_dataset_1200 after authorized reconstruction."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only the first N probe samples (useful for smoke tests)",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=None,
        help="Process only the first N models from the selected model list",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
        help="Model computation dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face model cache directory",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only models and tokenizers already present in the local cache",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repositories to execute custom modeling code",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply a tokenizer chat template when one is available",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Process answer options sequentially instead of as a batch",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be a positive integer")
    if args.num_models is not None and args.num_models <= 0:
        raise SystemExit("--num-models must be a positive integer")

    textindex_list = (
        list(range(args.max_samples)) if args.max_samples is not None else None
    )

    runner = RunnerABLE(
        textdata_name=args.dataset_name,
        model_list_name=args.model_list_name,
        num_models=args.num_models,
        log_name=args.log_name,
        textindex_list=textindex_list,
        models_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dtype=DTYPES[args.dtype],
        apply_chat_template=args.apply_chat_template,
        run_batch=not args.no_batch,
    )
    return 0 if runner.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
