import logging
import sys
from pathlib import Path

from artefactual.calibration.helpers.models import get_model_name
from artefactual.calibration.outputs_entropy import (
    GenerationConfig,
    generate_entropy_dataset,
)
from artefactual.calibration.rates_answers import RatingConfig, rate_answers
from artefactual.calibration.train_calibration import train_calibration

# Add the src directory to sys.path to allow importing the artefactual package
# This assumes the script is located in examples/ and the package is in src/
current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    # Define paths
    workspace_root = current_dir.parent
    input_data_path = workspace_root / "sample_qa_data.json"
    output_dir = workspace_root / "outputs"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    model_path = "mistralai/Ministral-8B-Instruct-2410"
    judge_model_path = "mistralai/Ministral-8B-Instruct-2410"

    # --- Step 1: Generate Entropy Dataset ---
    logger.info("=== Step 1: Generating Entropy Dataset ===")
    gen_config = GenerationConfig(
        model_path=model_path,
        n_queries=5,  # Limit to 5 queries for demonstration purposes
        log_to_file=False,
        iterations=1,
        number_logprobs=15,
    )

    generate_entropy_dataset(input_path=input_data_path, output_path=output_dir, config=gen_config)

    # Determine the output file name from Step 1
    # Logic matches outputs_entropy.py: {input_dataset_name}_{model_name}_entropy.json
    model_name = get_model_name(model_path)
    input_dataset_name = input_data_path.stem
    entropy_output_file = output_dir / f"{input_dataset_name}_{model_name}_entropy.json"

    if not entropy_output_file.exists():
        logger.error(f"Expected entropy output file not found: {entropy_output_file}")
        msg = f"Expected entropy output file not found: {entropy_output_file}"
        raise FileNotFoundError(msg)

    logger.info(f"Entropy dataset generated at: {entropy_output_file}")

    # --- Step 2: Rate Answers ---
    logger.info("=== Step 2: Rating Answers ===")
    rating_config = RatingConfig(input_file=entropy_output_file, judge_model_path=judge_model_path, temperature=0.0)

    df_judgments = rate_answers(rating_config)

    if df_judgments.empty:
        logger.error("No judgments were generated. Exiting.")
        sys.exit(1)

    judgment_output_file = output_dir / "df_judgment.csv"
    df_judgments.to_csv(judgment_output_file)
    logger.info(f"Judgments saved to {judgment_output_file}")

    # --- Step 3: Train Calibration ---
    logger.info("=== Step 3: Training Calibration ===")
    calibration_weights_file = output_dir / f"calibration_weights_{model_name}.json"

    train_calibration(input_file=judgment_output_file, output_file=calibration_weights_file)
    logger.info(f"Calibration weights saved to {calibration_weights_file}")
    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    run_pipeline()
