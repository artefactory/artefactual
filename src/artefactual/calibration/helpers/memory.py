import contextlib
import gc
import logging

import ray
import torch
from vllm import LLM
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


def clear_gpu_memory(llm: LLM) -> None:
    """
    Clears GPU memory by destroying the model parallel and distributed environment,
    deleting the LLM object, and clearing the CUDA cache.

    This function is intended to be called after using a vLLM model to free up
    GPU resources. It handles the destruction of various components created by
    vLLM and PyTorch for distributed processing.

    Args:
        llm (LLM): The vLLM LLM object to be deleted.
    """
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    logging.info("Successfully deleted the llm pipeline and freed the GPU memory.")
