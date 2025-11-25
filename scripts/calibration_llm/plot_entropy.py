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
Script that take as input a list of file
parse each of them to dataframe
augment them with the original question
extract the temperature from the filename

"""

import dataclasses

from absl import app, logging
from artefactual.data.readers import read_file
from beartype import beartype
from etils import eapp, edc, epath


@beartype
@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    responses: list[epath.Path]
    scores: list[epath.Path]
    name: str | None = None
    threshold: int = 3


def main(cfg: AppConfig):
    logging.info("\n%s", cfg)
    for file_scores, file_responses in zip([cfg.scores, cfg.responses], strict=False):
        df_scores = read_file(file_scores)
        df_responses = read_file(file_responses)

        df = df_responses.join(df_scores, on="id")
        df = df.select(["question", "answer", "response", "score", "logprobs", "id"])

    # TODO: implement entropy calculations and plotting with respect to temperature
    # Note: Uncomment this when ready to log to W&B
    # import wandb
    # with wandb.init(config=dataclasses.asdict(cfg), project="artefactual", name=cfg.name) as run:
    #     run.log({})


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))
