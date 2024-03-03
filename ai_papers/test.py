from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import click
import jinja2
import jsonlines
import torch.cuda
import transformers

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutput

_QUESTION = "Is this paper AI-relevant?"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument(
    "DATA_FILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)
@click.argument(
    "PROMPT_FILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)
@click.argument(
    "MODEL_DIR",
    type=click.Path(
        exists=True, file_okay=False, readable=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "OUTPUT_FILE",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
)
def main(
    data_file: pathlib.Path,
    prompt_file: pathlib.Path,
    model_dir: pathlib.Path,
    output_file: pathlib.Path,
):
    model: transformers.GPT2LMHeadModel = transformers.GPT2LMHeadModel.from_pretrained(
        model_dir
    ).to(_DEVICE)
    tokenizer: transformers.GPT2Tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        model_dir
    )
    prompt_template = jinja2.Template(prompt_file.read_text())
    responses = []
    with jsonlines.open(data_file) as reader:
        for prompt_data in reader:
            title, _, abstract = prompt_data["text"].partition(".")
            prompt = prompt_template.render(
                title=title, abstract=abstract, question=_QUESTION
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=False).to(
                _DEVICE
            )
            output_ids = model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=1
            )
            output = tokenizer.decode(
                output_ids[0, -1].to("cpu"), skip_special_tokens=True
            )
            output = {"yes": True, "no": False}.get(
                output.strip().lower(), float("nan")
            )
            responses.append({"label": output})

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(responses)


if __name__ == "__main__":
    main()
