from vllm import LLM


def init_llm(model_path: str, seed: int) -> LLM:
    if model_path.startswith("mistralai/"):
        llm = LLM(
            model=model_path,
            seed=seed,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )
    else:
        llm = LLM(
            model=model_path,
        )
    return llm


def get_model_name(model_path: str) -> str:
    return model_path.rsplit("/", maxsplit=1)[-1] if "/" in model_path else model_path.rsplit(".", maxsplit=1)[-1]
