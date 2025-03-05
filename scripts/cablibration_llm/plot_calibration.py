# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beartype",
#     "etils[eapp,edc,epath]",
#     "polars",
#     "scikit-learn",
#     "toolz",
#     "wandb",
# ]
# ///
"""
Script that takes a responses and a score file.
It read samples from both file, join them on the id then plot calibration curve using the score key.
If there is no score key, it uses the logprobs key to compute a naive score.
If here is no logprobs key, it returns an error
"""

import dataclasses

import polars as pl
import wandb
from absl import app, logging
from beartype import beartype
from etils import eapp, edc, epath
from sklearn.calibration import calibration_curve


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    responses_file: epath.Path
    ratings_file: epath.Path
    scores_file: epath.Path
    name: str | None = None
    n_bins: int = 11
    random_seed: int = 42
    x_label: str = "Mean predicted value"
    y_label: str = "Fraction of positives"
    threshold: int = 3


@beartype
def read_file(path: epath.Path) -> pl.DataFrame:
    with path.open("r") as src:
        df = pl.read_ndjson(src)
    return df.with_columns([pl.col("id").cast(pl.Int64, strict=True)])


def main(cfg: AppConfig):
    # TODO: Add identifier for judge model
    # TODO: Add identifier for scoring method
    logging.info("\n%s", cfg)
    df_ratings = read_file(cfg.ratings_file)
    df_scores = read_file(cfg.scores_file)
    df_responses = read_file(cfg.responses_file)

    df = df_ratings.join(df_scores, on="id")
    df = df.join(df_responses, on="id")
    df = df.select(["question", "answer", "response", "rating", "score", "explanation", "logprobs", "id"])

    logging.info("Got %d rows", len(df))

    y_true = (df.select(pl.col("rating")).to_numpy().flatten() > cfg.threshold).astype(int)
    y_probas = df.select(pl.col("score")).to_numpy().flatten()
    frac_pos, pred_values = calibration_curve(y_true, y_probas, n_bins=cfg.n_bins, pos_label=True)
    data_line = list(zip(frac_pos, pred_values, strict=False))

    table_line = wandb.Table(data=data_line, columns=[cfg.x_label, cfg.y_label])
    # hist, edges = np.histogram(y_probas, bins=cfg.n_bins, density=False)
    data_hist = [[h] for h in y_probas]
    table_data = wandb.Table(data=data_hist, columns=["Score"])
    columns = df.columns
    data = wandb.Table(data=[tuple(dico.values()) for dico in df.to_dicts()], columns=columns)
    with wandb.init(config=dataclasses.asdict(cfg), project="artefactual", name=cfg.name) as run:
        run.log({
            "calibration_plot": wandb.plot.line(table_line, cfg.x_label, cfg.y_label, title="Calibration plot"),
            "histogram": wandb.plot.histogram(table_data, "Score", title="Score distribution"),
            "Dataset": data,
        })


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))
