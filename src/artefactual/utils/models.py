from vllm import LLM


def init_llm(model_path: str, seed: int, tensor_parallel_size: int) -> LLM:
    if model_path.startswith("mistralai/"):
        llm = LLM(
            model=model_path,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )
    else:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
        )
    return llm


def get_model_name(model_path: str) -> str:
    if "/" in model_path:
        model_name = model_path.rsplit("/", maxsplit=1)[-1]
    else:
        model_name = model_path.rsplit(".", maxsplit=1)[-1]
    return model_name
