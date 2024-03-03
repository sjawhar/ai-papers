import pathlib

import click
import jsonlines


@click.command()
@click.argument(
    "INPUT_FILE",
    type=click.Path(exists=True, path_type=pathlib.Path, readable=True, dir_okay=False),
)
@click.argument(
    "OUTPUT_FILE",
    type=click.Path(path_type=pathlib.Path, writable=True, dir_okay=False),
)
def main(input_file: pathlib.Path, output_file: pathlib.Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with (
        jsonlines.open(input_file) as reader,
        jsonlines.open(
            output_file,
            "w",
            compact=True,
            sort_keys=True,
        ) as writer,
    ):
        for item in reader:
            title, _, abstract = item["text"].partition(".")
            item_clean = {
                "title": title.strip(),
                "abstract": abstract.strip(),
            }
            if label := item.get("label"):
                item_clean["label"] = str(label).lower() == "true"
            writer.write(item_clean)


if __name__ == "__main__":
    main()
