import gc
import math
import torch
from loguru import logger
from rich.progress import track
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple, Optional

try:
    from transformers import activations

    if not hasattr(activations, "PytorchGELUTanh") and hasattr(activations, "GELUTanh"):
        activations.PytorchGELUTanh = activations.GELUTanh
except ImportError:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import LayerGradientXActivation


class ABLECalculator:
    def __init__(
        self,
        model_name: str,
        apply_chat_template: bool,
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        local_files_only: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        run_batch: Optional[bool] = False,
    ):
        self.apply_chat_template = apply_chat_template
        self.run_batch = run_batch

        self.model_name = model_name
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self._over_length_indices = []

        if self.model_name in ["mosaicml/mpt-30b-instruct", "mosaicml/mpt-7b-8k", "mosaicml/mpt-7b", "mosaicml/mpt-7b-instruct"]:
            self.trust_remote_code = False

        self._load_model()
        self.is_chat_model = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.pad_id = self.tokenizer.pad_token_id

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            cache_dir=self.cache_dir,
            dtype=self.dtype,
            output_hidden_states=False,
            output_attentions=False,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        self._set_device()
        self._set_max_length()

    def _set_device(self):
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            module_names = list(self.model.hf_device_map.keys())
            first_module = module_names[0]
            self.input_device = self.model.hf_device_map[first_module]
            last_module = module_names[-1]
            self.output_device = self.model.hf_device_map[last_module]
        else:
            self.input_device = next(iter(self.model.parameters())).device
            self.output_device = self.input_device

    def _set_max_length(self):
        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_length = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "n_positions"):
            self.max_length = self.model.config.n_positions
        else:
            self.max_length = getattr(self.model.config, "max_sequence_length", 2048)

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        choice_len_tensor: torch.Tensor,
        question_len: int,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        logits = out.logits[:, :-1, :]
        output_device = logits.device
        labels = input_ids[:, 1:].to(output_device)
        choice_len_tensor = choice_len_tensor.to(output_device)

        picked = (
            F.log_softmax(logits, dim=-1)
            .gather(-1, labels.unsqueeze(-1))
            .squeeze(-1)
        )
        _, seq_len = picked.shape

        indices = torch.arange(seq_len, device=output_device).unsqueeze(0)
        start_idx = question_len - 1
        end_idxs = (start_idx + choice_len_tensor).unsqueeze(1)

        mask = (indices >= start_idx) & (indices < end_idxs)
        masked = picked * mask
        return masked.sum(dim=1)

    def _calculate_able(
        self, tokenized_data: List[Tuple[int, List[torch.Tensor], int, int, List[int]]]
    ) -> List[Dict]:
        results = []
        gxact = LayerGradientXActivation(
            forward_func=None, layer=self.model.get_input_embeddings()
        )

        for index, tokenized, question_len, ans_idx, choice_lens in track(
            tokenized_data, total=len(tokenized_data), description="Processing"
        ):
            current_textdata = {
                "index": index,
                "num_tokens": [],
                "input_attrs": [],
                "calculate_vals": [],
                "able": None,
                "ans_idx": ans_idx,
                "use_chat_template": self.apply_chat_template,
            }

            def forward_wrapper(inputs, c_lens, mask):
                return self._model_forward(
                    input_ids=inputs,
                    choice_len_tensor=c_lens,
                    question_len=question_len,
                    attention_mask=mask,
                )

            gxact.forward_func = forward_wrapper

            if self.run_batch:
                tensors_to_pad = [t.squeeze(0) for t in tokenized]
                current_textdata["num_tokens"] = [t.shape[0] for t in tensors_to_pad]

                input_batch = pad_sequence(
                    tensors_to_pad, batch_first=True, padding_value=self.pad_id
                ).to(self.input_device)
                c_lens_tensor = torch.tensor(choice_lens, device=self.input_device)
                sequence_lengths = torch.tensor(
                    current_textdata["num_tokens"], device=self.input_device
                ).unsqueeze(1)
                token_positions = torch.arange(
                    input_batch.shape[1], device=self.input_device
                ).unsqueeze(0)
                attention_mask = (token_positions < sequence_lengths).long()

                with torch.no_grad():
                    calculate_vals_batch = forward_wrapper(
                        input_batch, c_lens_tensor, attention_mask
                    )
                    current_textdata["calculate_vals"] = (
                        calculate_vals_batch.cpu().tolist()
                    )

                self.model.zero_grad()
                attr_batch = gxact.attribute(
                    inputs=input_batch,
                    additional_forward_args=(c_lens_tensor, attention_mask),
                    attribute_to_layer_input=False,
                )
                token_attr_batch = attr_batch.sum(dim=-1)

                for i in range(len(tokenized)):
                    current_textdata["input_attrs"].append(
                        token_attr_batch[i, :question_len].detach().cpu()
                    )

                del (
                    input_batch,
                    c_lens_tensor,
                    sequence_lengths,
                    token_positions,
                    attention_mask,
                    attr_batch,
                    token_attr_batch,
                )

            else:
                for i in range(len(tokenized)):
                    tokens = tokenized[i].to(self.input_device)
                    c_len_tensor = torch.tensor(
                        [choice_lens[i]], device=self.input_device
                    )
                    current_textdata["num_tokens"].append(tokens.shape[1])
                    attention_mask = torch.ones_like(tokens, device=self.input_device)

                    with torch.no_grad():
                        calculate_vals = forward_wrapper(
                            tokens, c_len_tensor, attention_mask
                        )
                        current_textdata["calculate_vals"].append(calculate_vals.item())

                    self.model.zero_grad()
                    attr = gxact.attribute(
                        inputs=tokens,
                        additional_forward_args=(c_len_tensor, attention_mask),
                        attribute_to_layer_input=False,
                    )
                    token_attr = attr.sum(dim=-1).squeeze(0)
                    current_textdata["input_attrs"].append(
                        token_attr[:question_len].detach().cpu()
                    )
                    del tokens, c_len_tensor, attention_mask, attr, token_attr

            current_textdata["input_attrs"] = [
                t.cpu().float().tolist() for t in current_textdata["input_attrs"]
            ]
            results.append(current_textdata)
        return results

    def _tokenize(
        self, texts: List[Tuple[int, str, List[str], int]]
    ) -> List[Tuple[int, List[torch.Tensor], int, int, List[int]]]:
        tokenized_data = []
        for text in texts:
            index, question, choices, ans_idx = text
            tokenized: List[torch.Tensor] = []
            choice_lens: List[int] = []

            if self.apply_chat_template and self.is_chat_model:
                user_messages = [{"role": "user", "content": question}]
                question_prompt_str = self.tokenizer.apply_chat_template(
                    user_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                question_prompt_str = (
                    question if question.endswith(("\n")) else f"{question}\n"
                )
            q_char_len = len(question_prompt_str)
            final_split_idx = -1

            for c in choices:
                if self.apply_chat_template and self.is_chat_model:
                    user_messages = [{"role": "user", "content": question}]
                    full_messages = user_messages + [
                        {"role": "assistant", "content": c}
                    ]
                    full_prompt_str = self.tokenizer.apply_chat_template(
                        full_messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    full_prompt_str = question_prompt_str + c

                try:
                    enc = self.tokenizer(
                        full_prompt_str,
                        return_tensors="pt",
                        return_offsets_mapping=True,
                    )
                    full_ids = enc["input_ids"]
                    offsets = enc["offset_mapping"][0]

                    split_idx = full_ids.shape[1]
                    for i, (start, end) in enumerate(offsets):
                        if start >= q_char_len:
                            split_idx = i
                            break
                except Exception as e:
                    logger.warning(
                        f"Offset mapping failed for index {index}: {e}. Falling back to length count."
                    )
                    full_ids = self.tokenizer(full_prompt_str, return_tensors="pt")[
                        "input_ids"
                    ]
                    if self.apply_chat_template and self.is_chat_model:
                        q_ids = self.tokenizer(
                            question_prompt_str, return_tensors="pt"
                        )["input_ids"]
                        split_idx = q_ids.shape[1]
                    else:
                        q_ids = self.tokenizer(
                            question_prompt_str, return_tensors="pt"
                        )["input_ids"]
                        split_idx = q_ids.shape[1]

                if full_ids.shape[1] > self.max_length:
                    self._over_length_indices.append(index)
                    raise ValueError(
                        f"Sample [{index}] is too long for model: {self.model_name}"
                    )

                if final_split_idx == -1:
                    final_split_idx = split_idx

                if self.apply_chat_template and self.is_chat_model:
                    pure_prompt_str = question_prompt_str + c
                    pure_ids = self.tokenizer(pure_prompt_str, return_tensors="pt")[
                        "input_ids"
                    ]
                    curr_choice_len = pure_ids.shape[1] - split_idx
                else:
                    curr_choice_len = full_ids.shape[1] - split_idx

                if curr_choice_len <= 0:
                    curr_choice_len = 1

                choice_lens.append(curr_choice_len)
                tokenized.append(full_ids)

            tokenized_data.append(
                (index, tokenized, final_split_idx, ans_idx, choice_lens)
            )
        return tokenized_data

    def _clear_cache(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def __call__(self, texts: List[Tuple[int, str, List[str], int]]) -> List[Dict]:
        tokenized_data = self._tokenize(texts)
        ables = self._calculate_able(tokenized_data)
        self._clear_cache()
        return ables
