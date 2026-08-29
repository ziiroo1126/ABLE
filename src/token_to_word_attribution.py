"""
Token-level Attribution to Word-level Attribution Converter

This script converts token-level attribution vectors to character-level,
then aggregates them to word-level attributions.

Processing Pipeline:
1. Parse model name from filename and load corresponding tokenizer
2. Use tokenizer's offset_mapping to map token attributions to characters
3. Aggregate character attributions to word level (space attribution appended to previous word)
"""

import json
import os
import re
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence

from loguru import logger
from transformers import AutoTokenizer
from rich.progress import track


# ============================================================================
# I/O Utility Functions
# ============================================================================


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Atomically replace a JSONL file after every record is serialized."""
    target_path = Path(file_path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for item in data:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_model_name_from_filename(filename: str) -> str:
    """Parse HuggingFace model name from filename.

    Example: "meta-llama--Meta-Llama-3-8B_bfloat16.jsonl" -> "meta-llama/Meta-Llama-3-8B"
    """
    name = filename.replace(".jsonl", "")
    for suffix in ["_bfloat16", "_float16", "_float32", "_fp16", "_bf16", "_fp32"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("--", "/")


# ============================================================================
# Attribution Conversion Core Functions
# ============================================================================


def get_question_offset_mapping(
    question_prompt_str: str,
    full_text: str,
    tokenizer,
) -> List[Tuple[int, int]]:
    """Get token offset mapping for the question part.

    Args:
        question_prompt_str: Question text (including trailing newline)
        full_text: Complete text (question + choice)
        tokenizer: HuggingFace tokenizer

    Returns:
        Character range for each token in question part [(start, end), ...]
    """
    q_char_len = len(question_prompt_str)

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()

    # Find first token belonging to choice part (start >= q_char_len)
    for i, (start, end) in enumerate(offsets):
        if start >= q_char_len:
            return offsets[:i]

    return offsets


def token_to_character_attribution(
    question_prompt_str: str,
    full_text: str,
    token_attrs: List[float],
    tokenizer,
) -> List[float]:
    """Convert token-level attribution to character-level attribution.

    Attribution conservation: each token's attribution is evenly distributed
    to all characters it covers.
    Special tokens (e.g., BOS) have their attribution accumulated to the first valid character.

    Args:
        question_prompt_str: Question text
        full_text: Complete text (question + choice)
        token_attrs: List of token-level attribution values
        tokenizer: HuggingFace tokenizer

    Returns:
        List of character-level attribution values, length equals question_prompt_str characters
    """
    offset_mapping = get_question_offset_mapping(
        question_prompt_str=question_prompt_str,
        full_text=full_text,
        tokenizer=tokenizer,
    )

    # Check length match
    num_tokens = min(len(token_attrs), len(offset_mapping))
    if len(token_attrs) != len(offset_mapping):
        logger.warning(
            f"Length mismatch: token_attrs ({len(token_attrs)}) vs "
            f"offset_mapping ({len(offset_mapping)}). Using first {num_tokens} tokens."
        )
        raise ValueError("Length mismatch")

    # Initialize character attribution array
    char_attrs = [0.0] * len(question_prompt_str)

    # Accumulate special token attribution (e.g., BOS token)
    pending_special_attr = 0.0
    first_valid_found = False

    # Distribute each token's attribution evenly to its covered characters
    for token_idx in range(num_tokens):
        start, end = offset_mapping[token_idx]
        attr_value = token_attrs[token_idx]

        # Special token (start == end): accumulate attribution for later
        if start == end:
            pending_special_attr += attr_value
            continue

        # Boundary check
        start = max(0, start)
        end = min(end, len(question_prompt_str))
        num_chars = end - start

        if num_chars <= 0:
            continue

        # First valid token: add accumulated special token attribution
        if not first_valid_found:
            attr_value += pending_special_attr
            pending_special_attr = 0.0
            first_valid_found = True

        # Attribution conservation: even distribution
        attr_per_char = attr_value / num_chars
        for char_idx in range(start, end):
            char_attrs[char_idx] += attr_per_char

    # If all tokens are special tokens, assign attribution to first character
    if pending_special_attr != 0 and len(char_attrs) > 0:
        char_attrs[0] += pending_special_attr

    return char_attrs


def character_to_word_attribution(
    text: str, char_attrs: List[float]
) -> Tuple[List[str], List[float]]:
    """Aggregate character attribution to word level.

    Space attribution is appended to the previous word (i.e., word + trailing space),
    leading space attribution is appended to the first word,
    trailing space attribution is appended to the last word.

    Args:
        text: Text string
        char_attrs: List of character-level attribution values
    Returns:
        (list of words, list of word attributions)
    """
    words = []
    word_attrs = []

    # Find all word positions
    word_matches = list(re.finditer(r"\S+", text))

    for i, match in enumerate(word_matches):
        word = match.group()
        word_start = match.start()

        # Determine trailing space end position (space attribution appended to previous word)
        if i == len(word_matches) - 1:
            # Last word: include all trailing characters (spaces, newlines, etc.)
            trailing_end = len(text)
        else:
            # Non-last word: include up to next word's start
            trailing_end = word_matches[i + 1].start()

        # Current word + its trailing spaces
        word_char_attrs = char_attrs[word_start:trailing_end]

        word_attr = sum(word_char_attrs) if word_char_attrs else 0.0

        words.append(word)
        word_attrs.append(word_attr)

    # Handle leading spaces (text-start spaces attributed to first word)
    if word_matches and word_matches[0].start() > 0:
        leading_attr = sum(char_attrs[:word_matches[0].start()])
        if leading_attr != 0 and word_attrs:
            word_attrs[0] += leading_attr

    return words, word_attrs


# ============================================================================
# Batch Processing Function
# ============================================================================


def process_directory(
    input_dir: str,
    dataset_path: str,
    output_dir: str = None,
    apply_chat_template: bool = False,
    skip_existing: bool = True,
    target_models: List[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
) -> bool:
    """Batch process all attribution files in a directory.

    Args:
        input_dir: Input directory containing token-level attribution JSONL files
        dataset_path: Dataset path containing question and choices
        output_dir: Output directory, default is input_dir/word_level
        apply_chat_template: Whether to use chat template for question formatting
        skip_existing: Whether to skip existing output files
        target_models: List of models to process (HuggingFace format). If None, process all models.
        local_files_only: Whether to restrict tokenizer loading to the local cache.
        trust_remote_code: Whether to allow custom code from model repositories.
        cache_dir: Optional Hugging Face cache directory for tokenizer files.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Set output directory
    if output_dir is None:
        output_path = input_path / "word_level"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all JSONL files
    jsonl_files = sorted([f for f in input_path.glob("*.jsonl") if f.is_file()])
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {input_dir}")
        return False

    logger.info(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    # Load dataset
    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_jsonl(dataset_path)
    dataset_by_index = {item["index"]: item for item in dataset}

    # Process each file
    succeeded = True
    for jsonl_file in jsonl_files:
        filename = jsonl_file.name
        output_file = output_path / f"{jsonl_file.stem}_word.jsonl"

        # Skip existing files
        if skip_existing and output_file.exists():
            logger.info(f"Skipping {filename} (output exists)")
            continue

        # Parse model name and load tokenizer
        model_name = parse_model_name_from_filename(filename)

        # Filter models
        if target_models is not None and model_name not in target_models:
            logger.info(f"Skipping {filename} (model {model_name} not in target list)")
            continue

        logger.info(f"Processing {filename} -> model: {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            succeeded = False
            continue

        is_chat_model = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        )

        # Load attribution data
        try:
            attributions = load_jsonl(str(jsonl_file))
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            succeeded = False
            continue

        # Process each sample
        results = []
        for attr_item in track(attributions, description=f"Converting {filename}"):
            index = attr_item["index"]

            if index not in dataset_by_index:
                continue

            data_item = dataset_by_index[index]
            question = data_item["question"]
            choices = data_item["choices"]

            # Build question prompt
            if apply_chat_template and is_chat_model:
                user_messages = [{"role": "user", "content": question}]
                question_prompt_str = tokenizer.apply_chat_template(
                    user_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                question_prompt_str = (
                    question if question.endswith("\n") else f"{question}\n"
                )

            input_attrs = attr_item.get("input_attrs", [])

            result_item = {
                "index": index,
                "ans_idx": attr_item.get("ans_idx"),
                "word_attrs_per_choice": [],
            }

            # Process each choice
            for choice_idx, token_attrs in enumerate(input_attrs):
                choice = choices[choice_idx] if choice_idx < len(choices) else ""

                # Build full text
                if apply_chat_template and is_chat_model:
                    full_messages = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": choice},
                    ]
                    full_text = tokenizer.apply_chat_template(
                        full_messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    full_text = question_prompt_str + choice

                # Token -> Character
                char_attrs = token_to_character_attribution(
                    question_prompt_str=question_prompt_str,
                    full_text=full_text,
                    token_attrs=token_attrs,
                    tokenizer=tokenizer,
                )

                # Character -> Word
                _words, word_attrs = character_to_word_attribution(
                    text=question_prompt_str,
                    char_attrs=char_attrs,
                )

                # Verify attribution conservation
                char_sum = sum(char_attrs)
                word_sum = sum(word_attrs)
                if abs(char_sum) > 1e-10:
                    rel_diff = abs(char_sum - word_sum) / abs(char_sum)
                    assert rel_diff < 0.01, (
                        f"Attribution not conserved: "
                        f"char_sum={char_sum:.6f}, word_sum={word_sum:.6f}"
                    )

                result_item["word_attrs_per_choice"].append(word_attrs)

            results.append(result_item)

        if not attributions or len(results) != len(attributions):
            logger.error(
                f"Refusing to save incomplete output for {filename}: "
                f"converted {len(results)} of {len(attributions)} records"
            )
            succeeded = False
            continue

        # Save results
        save_jsonl(results, str(output_file))
        logger.info(f"Saved {len(results)} samples to {output_file}")

    if succeeded:
        logger.info(f"All files processed. Output directory: {output_path}")
    else:
        logger.error(f"Conversion completed with errors. Output directory: {output_path}")
    return succeeded


# ============================================================================
# Command Line Interface
# ============================================================================


def parse_cli_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Convert token-level attributions to word-level attributions."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing token-level attribution JSONL files"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset JSONL file containing questions and choices"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: input_dir/word_level)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face tokenizer cache directory"
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template for chat models"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process files even if output exists"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizers only from the local Hugging Face cache"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from model repositories (disabled by default)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli_args(argv)

    succeeded = process_directory(
        input_dir=args.input_dir,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        apply_chat_template=args.apply_chat_template,
        skip_existing=not args.no_skip_existing,
        target_models=None,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
