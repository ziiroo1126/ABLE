"""Shared prompt construction and token boundaries for ABLE attributions."""

ATTRIBUTION_VERSION = 2


def build_question_prompt(
    question: str, tokenizer, apply_chat_template: bool
) -> tuple[str, bool]:
    use_chat_template = (
        apply_chat_template and getattr(tokenizer, "chat_template", None) is not None
    )
    if use_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = question if question.endswith("\n") else f"{question}\n"
    return prompt, use_chat_template


def tokenize_prompt_text(
    text: str,
    tokenizer,
    use_chat_template: bool = False,
    return_offsets_mapping: bool = False,
):
    kwargs = {"return_tensors": "pt"}
    if return_offsets_mapping:
        kwargs["return_offsets_mapping"] = True
    if use_chat_template:
        # The rendered template already contains its own special tokens.
        kwargs["add_special_tokens"] = False
    return tokenizer(text, **kwargs)


def choice_start_from_offsets(offsets, prompt_length: int) -> int:
    # A token can include both trailing prompt whitespace and the answer.
    for index, (_, end) in enumerate(offsets):
        if end > prompt_length:
            return index
    return len(offsets)
