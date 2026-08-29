# Probe data

`ABLE_dataset_public_1000.jsonl` contains the five openly redistributable
200-example subsets used by ABLE: ARC, HellaSwag, MMLU, WinoGrande, and
CommonsenseQA. Its `index` field is contiguous from 0 to 999. `full_index`
records each example's position in the paper's original 1,200-example corpus.

The paper also used 200 examples from GPQA. The official GPQA access conditions
ask users not to reveal its examples in plain text or images online, so those
questions and answers are not included here. `gpqa_selection_manifest.json`
contains only source indices and ordering metadata; it contains no GPQA text.

## Reconstruct the paper corpus

1. Request and accept access at <https://huggingface.co/datasets/Idavidrein/gpqa>.
2. Authenticate locally with `hf auth login`.
3. Install the loader with `pip install datasets`.
4. From the repository root, run:

```bash
python scripts/build_full_probe_dataset.py
```

This creates the ignored local file `ABLE_dataset_1200.jsonl`. The script pins
the recorded GPQA revision and verifies a canonical SHA-256 digest before
writing the file. You can instead supply an authorized local official source:

```bash
python scripts/build_full_probe_dataset.py --gpqa-source /path/to/gpqa_main.csv
```

Do not commit the reconstructed file or the downloaded GPQA source.

## Sources, transformations, and licenses

ABLE selected 200 examples from each source with seed 12345, converted them to
a shared multiple-choice prompt format, and interleaved the resulting records.
The public file therefore contains transformed excerpts rather than original
dataset dumps. The underlying examples remain subject to their source licenses.

| `resource` | Source configuration and split | Source license |
|---|---|---|
| `ai2_arc` | [ARC-Challenge, test](https://huggingface.co/datasets/allenai/ai2_arc) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `hellaswag` | [HellaSwag, validation](https://huggingface.co/datasets/Rowan/hellaswag) | [MIT](licenses/HellaSwag-LICENSE.txt) |
| `mmlu` | [MMLU all, test](https://huggingface.co/datasets/cais/mmlu) | [MIT](licenses/MMLU-LICENSE.txt) |
| `winogrande` | [WinoGrande XL, validation](https://huggingface.co/datasets/allenai/winogrande) | [CC BY](https://github.com/allenai/winogrande#license) |
| `commonsense_qa` | [CommonsenseQA, validation](https://huggingface.co/datasets/tau/commonsense_qa) | [MIT](https://huggingface.co/datasets/tau/commonsense_qa#licensing-information) |
| `gpqa` | [GPQA main, train](https://huggingface.co/datasets/Idavidrein/gpqa) | CC BY 4.0 plus the official access conditions; text is not redistributed |

To satisfy ARC's share-alike requirement, the transformed public compilation
`ABLE_dataset_public_1000.jsonl` is released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This does not
replace the notices or licenses of the individual source datasets. Please cite
the corresponding benchmark papers when using these examples.

The CommonsenseQA license identification follows its Hugging Face dataset
card; the upstream GitHub repository does not currently contain a standalone
license file. Users with redistribution requirements should independently
confirm that upstream designation.
