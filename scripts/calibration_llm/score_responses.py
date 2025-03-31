# /// script
# requires-python = ">=3.11"
#
# dependencies = [
#     "einops",
#     "etils[eapp,edc,epath]",
#     "jaxtyping",
#     "numpy",
#     "orjson",
#     "plum-dispatch",
#     "polars",
#     "scikit-learn",
#     "toolz",
# ]
# ///

import dataclasses
from functools import partial

import numpy as np
import polars as pl
import tlz
from absl import app, logging
from etils import eapp, edc, epath

# Import from artefactual library
from artefactual.data import join_samples, read_file
from artefactual.scoring import ScoringMethod, process_logprobs, score_fn


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    responses_file: epath.Path
    max_length: int
    ratings_file: epath.Path | None = None
    batch_size: int = 2**20
    output_dir: epath.Path | None = None
    method: ScoringMethod = ScoringMethod.NAIVE
    threshold: int = 4

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.responses_file.parent


DEFAULT_GET_ID = tlz.curried.get("id")


def main(cfg: AppConfig):
    logging.info("\n%s", cfg)

    output_file: epath.Path = cfg.output_dir / f"scores_{cfg.method}_{cfg.responses_file.name}.json".replace("/", "_")
    if cfg.ratings_file:
        output_file = f"{output_file.name}_{cfg.ratings_file.name}.json"
    if output_file.exists():
        msg = f"File {output_file} already exists"
        raise FileExistsError(msg)
    samples = read_file(cfg.responses_file)

    labels = None
    if cfg.ratings_file:
        rating_samples = tlz.pipe(cfg.ratings_file, read_file, tlz.curried.filter(lambda d: d["rating"] is not None))
        samples = join_samples(rating_samples, samples)
        ratings = tlz.pipe(samples, tlz.curried.pluck("rating"), tlz.curried.map(tlz.compose_left(float, int)))
        labels = tlz.pipe(
            ratings,
            tlz.curried.map(lambda rating: rating >= cfg.threshold),
            list,
            partial(np.array, dtype=int),
        )

    ids, logprobs = zip(*tlz.pluck(["id", "logprobs"], samples), strict=False)
    ids = np.array(list(map(int, ids)), dtype=int)

    logprobs = process_logprobs(logprobs, max_len=cfg.max_length)

    if labels is not None:
        ids, scores, labels = score_fn(cfg.method, logprobs, ids, labels)
        df = pl.DataFrame({"id": ids, "score": scores, "label": labels})
    else:
        scores = score_fn(cfg.method, logprobs)
        df = pl.DataFrame({"id": ids, "score": scores})

    with output_file.open("w") as dst:
        df.write_ndjson(dst)
    logging.info("Wrote scores into %s", output_file)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))
