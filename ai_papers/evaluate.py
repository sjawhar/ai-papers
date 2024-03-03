from __future__ import annotations

import pathlib
from typing import Any

import click
import pandas as pd
import ruamel.yaml
import sklearn.metrics


@click.command()
@click.argument(
    "DATA_FILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)
@click.argument(
    "PREDS_FILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)
@click.argument(
    "METRICS_FILE",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
    default="params.yaml",
)
def main(
    data_file: pathlib.Path,
    preds_file: pathlib.Path,
    metrics_file: pathlib.Path,
    params_file: pathlib.Path,
):
    yaml = ruamel.yaml.YAML(typ="safe")
    click.echo(f"Loading parameters from {params_file}")
    with params_file.open() as file:
        params: dict[str, Any] = yaml.load(file)["evaluate"]

    click.echo(f"Loading data from {data_file}")
    y_true = pd.read_json(data_file, lines=True)["label"]
    click.echo(f"Loading predictions from {preds_file}")
    y_pred = pd.read_json(preds_file, lines=True)["label"]
    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "precision": sklearn.metrics.precision_score(y_true, y_pred),
        "recall": sklearn.metrics.recall_score(y_true, y_pred),
        "f1": sklearn.metrics.f1_score(y_true, y_pred),
    }
    click.echo(f"Writing metrics to {metrics_file}")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    yaml.default_flow_style = False
    with metrics_file.open("w") as file:
        yaml.dump(
            pd.Series(metrics).round(params["precision"]).sort_index().to_dict(),
            file,
        )


if __name__ == "__main__":
    main()
