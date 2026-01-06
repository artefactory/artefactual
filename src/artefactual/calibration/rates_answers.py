import argparse
import json
import logging
import pathlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError
from vllm import SamplingParams

from artefactual.calibration import init_llm

logger = logging.getLogger(__name__)


class RatingConfig(BaseModel):
    """Configuration for answer rating."""

    judge_model_path: str = "mistralai/Ministral-8B-Instruct-2410"
    input_file: str | Path
    temperature: float = 0.0
    max_tokens: int = 10
    seed: int = 42


class ResultItem(BaseModel):
    """Model for a single result item in the input JSON."""

    query_id: str
    query: str
    # Support both keys to be robust
    expected_answers: list[str] | str | None = None
    expected_answer: list[str] | str | None = None
    generated_answers: list[dict[str, Any]]

    def get_expected_answers(self) -> list[str]:
        """Retrieve expected answers as a list, handling aliases and types."""
        ans = self.expected_answers or self.expected_answer or []
        if isinstance(ans, str):
            return [ans]
        return ans


def _prepare_judgment_messages(query: str, expected_answers: list[str], generated_answer: str) -> list[dict]:
    """Prepare messages for the judge LLM."""
    prompt = (
        "You are a precise judge evaluating the correctness of an answer to a question.\n"
        "Compare the Generated Answer to the Expected Answer(s).\n"
        "If the Generated Answer conveys the same meaning as any of the Expected Answers, mark it as True.\n"
        "Otherwise, mark it as False.\n"
        "If you cannot determine, mark it as None.\n"
        "Output a single word: 'True', 'False', or 'None'. Do not provide any explanation.\n\n"
        f"Question: {query}\n"
        f"Expected Answer(s): {expected_answers}\n"
        f"Generated Answer: {generated_answer}\n"
        "Judgment:"
    )
    return [{"role": "user", "content": prompt}]


def _parse_judgment(text: str) -> bool | None:
    """Parse the judgment text from the LLM."""
    text = text.strip().lower().rstrip(".")
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _load_results(input_file: str | Path) -> list[dict]:
    """Load results from the input JSON file."""
    # FIXME: input_file should already be a Path; avoid converting inside this function.
    input_path = pathlib.Path(input_file)
    try:
        with input_path.open(encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        logger.exception("Input file not found: %s", input_path)
        return []
    results = raw_data.get("results", [])
    if not results:
        logger.warning("No results found in input file.")
    return results


def _extract_answer_text(gen_ans_dict: dict[str, Any]) -> str | None:
    """Extract answer text from a generated answer dictionary."""
    for k, v in gen_ans_dict.items():
        if k.isdigit():
            return v
    for k, v in gen_ans_dict.items():
        if k != "epr_score":
            return v
    return None


def _prepare_rating_data(results_list: list[dict]) -> tuple[list, list]:
    """Prepare messages and metadata for the judge LLM."""
    messages_list, metadata_list = [], []
    for item_data in results_list:
        try:
            result = ResultItem(**item_data)
        except ValidationError as e:
            logger.warning(f"Skipping invalid item due to validation error: {e}")
            continue

        expected_answers = result.get_expected_answers()
        for gen_ans_dict in result.generated_answers:
            ans_text = _extract_answer_text(gen_ans_dict)
            if ans_text is None:
                continue

            messages = _prepare_judgment_messages(result.query, expected_answers, ans_text)
            messages_list.append(messages)
            metadata_list.append({
                "query_id": result.query_id,
                "uncertainty_score": gen_ans_dict.get("epr_score"),
                "generated_answer": ans_text,
            })
    return messages_list, metadata_list


def _generate_judgments(messages_list: list, config: RatingConfig) -> list:
    """Generate judgments using the judge LLM."""
    logger.info(f"Initializing judge model: {config.judge_model_path}")
    llm = init_llm(
        model_path=config.judge_model_path,
        seed=config.seed,
    )
    sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens)
    logger.info(f"Generating judgments for {len(messages_list)} answers...")
    return llm.chat(messages=messages_list, sampling_params=sampling_params, use_tqdm=True)


def _create_results_dataframe(outputs: list, metadata_list: list) -> pd.DataFrame:
    """Create a DataFrame from the LLM outputs and metadata."""
    final_data = []
    for i, output in enumerate(outputs):
        judgment_text = output.outputs[0].text
        judgment = _parse_judgment(judgment_text)
        final_data.append({
            "query_id": metadata_list[i]["query_id"],
            "uncertainty_score": metadata_list[i]["uncertainty_score"],
            "judgment": judgment,
        })

    df = pd.DataFrame(final_data)
    if not df.empty:
        df.set_index("query_id", inplace=True)
    return df


def rate_answers(config: RatingConfig) -> pd.DataFrame:
    """
    Rate answers generated by a model using a judge LLM.

    Args:
        config (RatingConfig): Configuration for the rating process.

    Returns:
        pd.DataFrame: DataFrame containing uncertainty scores and judgments, indexed by query_id.
    """
    results_list = _load_results(config.input_file)
    if not results_list:
        return pd.DataFrame()

    messages_list, metadata_list = _prepare_rating_data(results_list)
    if not messages_list:
        logger.warning("No generated answers found to rate.")
        return pd.DataFrame()

    outputs = _generate_judgments(messages_list, config)
    return _create_results_dataframe(outputs, metadata_list)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Rate answers using a judge LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Path to the judge model.",
    )
    parser.add_argument("--output_file", type=str, help="Path to save the output DataFrame (CSV).")

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_file)
    if input_path.exists():
        config = RatingConfig(input_file=args.input_file, judge_model_path=args.model_path)
        df = rate_answers(config)
        logger.info("Rated answers head:\n%s", df.head(10))
        logger.info(f"Total rated: {len(df)}")
        if args.output_file:
            df.to_csv(args.output_file)
            logger.info(f"Saved results to {args.output_file}")
    else:
        logger.error(f"Input file {args.input_file} not found.")
        sys.exit(1)
