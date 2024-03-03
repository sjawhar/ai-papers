from __future__ import annotations

import pathlib
import random
from typing import Any

import click
import jsonlines
import ruamel.yaml


@click.command()
@click.argument(
    "INPUT_FILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)
@click.argument(
    "OUTPUT_FILE",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
    default="params.yaml",
)
def main(
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    params_file: pathlib.Path,
):
    click.echo(f"Loading parameters from {params_file}")
    yaml = ruamel.yaml.YAML(typ="safe")
    with params_file.open() as file:
        params: dict[str, Any] = yaml.load(file)
    answers = params["answers"]
    params = params["make_examples"]
    random.seed(params["random_seed"])

    examples: list[dict[str, Any]] = list(jsonlines.open(input_file, "r").iter())

    num_examples = params["num_examples"]
    num_positive = max(1, num_examples // 2)
    num_negative = num_examples - num_positive
    click.echo(
        f"Preparing {num_positive} positive and {num_negative} negatiive examples from {input_file}"
    )
    selected_examples = {
        True: (num_positive, []),
        False: (num_negative, []),
    }
    for example in sorted(
        examples,
        key=lambda x: (len(x["title"]) + len(x["abstract"])),
    ):
        is_positive = example.pop("label")
        num_expected, selected = selected_examples[is_positive]
        if len(selected) < num_expected:
            example["answer"] = answers["positive" if is_positive else "negative"]
            selected.append(example)
        if all(
            len(selected) >= num_expected
            for num_expected, selected in selected_examples.values()
        ):
            break

    examples = [
        example for _, selected in selected_examples.values() for example in selected
    ]
    random.shuffle(examples)

    click.echo(f"Writing examples to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_file, mode="w", compact=True, sort_keys=True) as writer:
        writer.write_all(examples)

    click.echo("Done!")


if __name__ == "__main__":
    main()
