"""
Process Word-level ABLE Attributions

This script processes word-level ABLE attributions and applies dimensionality reduction.
It supports PCA, Johnson-Lindenstrauss random projection, or no reduction.

Features:
- Multiple dimensionality reduction methods (PCA, JL, none)
- Configurable normalization modes
- Support for correct-only or all-options modes
- Saves individual model files and reduction model (PCA/JL projector)
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Sequence
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection
import joblib


# ==========================================
# 1. Command Line Arguments
# ==========================================
def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Process word-level ABLE attributions and reduce dimensions."
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["pca", "jl", "none"],
        default="jl",
        help="Dimensionality reduction method: 'pca', 'jl' (Johnson-Lindenstrauss), or 'none' (default: jl)"
    )
    parser.add_argument(
        "--dim", "-d",
        type=int,
        default=256,
        help="Target dimension for JL projection (ignored for PCA which uses variance ratio) (default: 256)"
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Variance ratio to retain for PCA (default: 0.95)"
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        choices=["pre", "post", "both", "none"],
        default="post",
        help="Normalization mode: 'pre' (L2 norm before reduction only), "
             "'post' (L2 norm after reduction only), "
             "'both' (L2 norm before and after reduction), "
             "'none' (no normalization). Default: 'post'"
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Only use attributions for correct answer options"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./able/word_level",
        help="Input directory containing word-level attribution files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--family-file",
        type=str,
        default="./data/models/model-family.csv",
        help="Path to model family CSV file"
    )

    return parser.parse_args(argv)


# ==========================================
# 2. Helper Functions
# ==========================================
def load_model_families(csv_path: str) -> Dict[str, str]:
    """Load model family mapping table."""
    if not Path(csv_path).exists():
        print(f"Warning: Family file not found at {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    family_map = dict(zip(df["model_name"], df["model_family"]))
    return family_map


def get_model_family(model_name: str, family_map: Dict[str, str]) -> str:
    """Get model family, return 'unknown' if not found."""
    if model_name in family_map:
        return family_map[model_name]
    return "unknown"


def safe_model_name(model_id: str) -> str:
    """Convert model name to filesystem-safe format."""
    return model_id.replace("/", "--")


def model_name_from_word_file(file_path: Path | str) -> str:
    """Extract the filesystem-safe model name from a word-attribution file."""
    name = Path(file_path).name
    if not name.endswith("_word.jsonl"):
        raise ValueError(f"Unexpected word-attribution filename: {name}")
    name = name[: -len("_word.jsonl")]
    for dtype_suffix in ("_bfloat16", "_float16", "_float32", "_bf16", "_fp16", "_fp32"):
        if name.endswith(dtype_suffix):
            name = name[: -len(dtype_suffix)]
            break
    return name


def get_output_dir_name(args) -> str:
    """Generate output directory name based on arguments."""
    feature_mode = "cor_option" if args.correct_only else "all_options"

    if args.method == "pca":
        dim_suffix = f"var{int(args.pca_variance * 100)}"
        norm_suffix = f"_{args.norm_mode}norm"
    elif args.method == "jl":
        dim_suffix = str(args.dim)
        norm_suffix = "_nonorm" if args.norm_mode == "none" else "_norm"
    else:
        dim_suffix = "full"
        norm_suffix = "_nonorm" if args.norm_mode == "none" else "_norm"

    return f"./able/able_word_{feature_mode}_{args.method}_{dim_suffix}{norm_suffix}"


# ==========================================
# 3. Feature Processing Functions
# ==========================================
def process_word_attrs(
    file_path: Path,
    family_map: Dict[str, str],
    correct_only: bool = True,
) -> Dict[str, Any]:
    """
    Process word_level JSONL files.
    Concatenate attribution vectors for each sample (normalization done in main).
    """
    model_name = model_name_from_word_file(file_path)

    flattened_attrs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                word_attrs_per_choice = record.get("word_attrs_per_choice", [])

                if correct_only:
                    ans_idx = record.get("ans_idx")
                    if ans_idx is not None and ans_idx < len(word_attrs_per_choice):
                        sample_attrs = np.array(
                            word_attrs_per_choice[ans_idx], dtype=np.float32
                        )
                        flattened_attrs.extend(sample_attrs.tolist())
                else:
                    if isinstance(word_attrs_per_choice, list):
                        for choice_attrs in word_attrs_per_choice:
                            if isinstance(choice_attrs, list):
                                sample_attrs = np.array(choice_attrs, dtype=np.float32)
                                flattened_attrs.extend(sample_attrs.tolist())

    return {
        "model_name": model_name,
        "model_family": get_model_family(model_name, family_map),
        "able_feature": np.array(flattened_attrs, dtype=np.float32).tolist(),
    }


# ==========================================
# 4. Main Processing Logic
# ==========================================
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Print configuration
    print("=" * 60)
    print("ABLE Feature Processing")
    print("=" * 60)
    print(f"Method: {args.method}")
    if args.method == "jl":
        print(f"Target dimension: {args.dim}")
    elif args.method == "pca":
        print(f"PCA variance ratio: {args.pca_variance}")
    print(f"Normalization mode: {args.norm_mode}")
    print(f"Correct only: {args.correct_only}")
    print("=" * 60)

    input_dir = Path(args.input_dir)
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(get_output_dir_name(args))

    print(f"\nLoading model families from {args.family_file}...")
    family_map = load_model_families(args.family_file)

    print(f"Scanning jsonl files in {input_dir}...")
    files = sorted(input_dir.glob("*_word.jsonl"))
    if not files:
        print(f"Error: No word-attribution files found in {input_dir}")
        return 1

    all_results = []

    mode_desc = "correct option only" if args.correct_only else "all options"
    print(f"Processing {len(files)} files (mode: {mode_desc})...")

    for file_path in tqdm(files):
        result = process_word_attrs(
            file_path, family_map, correct_only=args.correct_only
        )
        all_results.append(result)

    # Get feature dimension info
    feature_dims = [len(result["able_feature"]) for result in all_results]
    if len(set(feature_dims)) != 1:
        print(
            "Error: Inconsistent feature dimensions across models: "
            f"{feature_dims}"
        )
        return 1
    original_dim = feature_dims[0]

    # Build feature matrix (n_models, original_dim)
    feature_matrix = np.array(
        [r["able_feature"] for r in all_results], dtype=np.float32
    )
    n_models = feature_matrix.shape[0]

    print(f"\nOriginal feature matrix shape: {feature_matrix.shape}")

    pca = None
    projector = None

    if args.method == "pca":
        do_prenorm = args.norm_mode in ("pre", "both")
        do_postnorm = args.norm_mode in ("post", "both")

        if do_prenorm:
            print(f"\nApplying L2 normalization before PCA...")
            feature_matrix = normalize(feature_matrix, norm='l2', axis=1)
        else:
            print(f"\nSkipping pre-normalization for PCA...")

        print(f"Applying PCA to retain {args.pca_variance*100:.0f}% variance...")

        pca = PCA(n_components=args.pca_variance, svd_solver="full", random_state=0)
        projected_matrix = pca.fit_transform(feature_matrix)

        final_dim = projected_matrix.shape[1]
        print(f"PCA selected {final_dim} components to explain {args.pca_variance*100:.0f}% variance")

        if do_postnorm:
            print(f"Applying L2 normalization after PCA...")
            projected_matrix = normalize(projected_matrix, norm='l2', axis=1)
        else:
            print(f"Skipping post-normalization for PCA...")

    elif args.method == "jl":
        use_norm = args.norm_mode != "none"

        if use_norm:
            effective_mode = "norm"
            print(f"\n[JL] Normalization mode '{args.norm_mode}' -> using post-normalization (pre/post/both are equivalent for JL)")
        else:
            effective_mode = "none"
            print(f"\n[JL] No normalization will be applied (preserving original scale)")

        print(f"Applying Johnson-Lindenstrauss random projection to reduce dimensions to {args.dim}...")

        projector = GaussianRandomProjection(n_components=args.dim, random_state=0)
        projected_matrix = projector.fit_transform(feature_matrix)
        final_dim = args.dim

        if use_norm:
            print(f"Applying L2 normalization after JL projection...")
            projected_matrix = normalize(projected_matrix, norm='l2', axis=1)
        else:
            print(f"Skipping normalization for JL (none mode)...")

    else:  # none (no dimensionality reduction)
        print(f"\nNo dimensionality reduction applied.")
        do_norm = args.norm_mode in ("pre", "post", "both")
        if do_norm:
            print(f"Applying L2 normalization...")
            projected_matrix = normalize(feature_matrix, norm='l2', axis=1)
        else:
            projected_matrix = feature_matrix
        final_dim = original_dim

    # Update results and save to individual files
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reduced dimensions from {original_dim} to {final_dim} for {n_models} models.")
    print(f"\nSaving each model to {output_dir}...")

    saved_files = []
    for idx, result in enumerate(tqdm(all_results, desc="Saving models")):
        model_name = result["model_name"]
        model_family = result["model_family"]
        able = projected_matrix[idx].tolist()

        safe_name = safe_model_name(model_name)
        model_dir = output_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        output_path = model_dir / f"{safe_name}_able.json"
        output_data = {
            "model_name": model_name,
            "model_family": model_family,
            "able": able,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        saved_files.append(output_path)

    # Save dimensionality reduction model to output directory
    if args.method == "pca" and pca is not None:
        pca_model_path = output_dir / "0_pca_model.pkl"
        joblib.dump(pca, pca_model_path)
        print(f"\nSaved PCA model to {pca_model_path}")
    elif args.method == "jl" and projector is not None:
        jl_model_path = output_dir / "0_jl_projector.pkl"
        joblib.dump(projector, jl_model_path)
        print(f"\nSaved JL projector to {jl_model_path}")

    # Verify and print dimensions
    print("\nVerification:")
    print(f"{'Model':<50} | {'feat dim':<8}")
    print("-" * 65)
    for output_path in saved_files[:5]:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m_name = data["model_name"]
        feat_dim = len(data["able"])
        print(f"{m_name:<50} | {feat_dim:<8}")

    print(f"\nDone! Saved {len(saved_files)} models to {output_dir}")
    print(f"File format: <output_dir>/<safe_model_name>/<safe_model_name>_able.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
