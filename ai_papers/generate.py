from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import click
import jinja2
import jsonlines
import torch.cuda
import tqdm
import transformers

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutput

_QUESTION = "Is this paper AI-relevant?"


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")

    click.echo(f"Loading model from {model_dir}")
    model: transformers.GPT2LMHeadModel = transformers.GPT2LMHeadModel.from_pretrained(
        model_dir
    ).to(device)
    tokenizer: transformers.GPT2Tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        model_dir
    )

    click.echo(f"Using prompt template {prompt_file}")
    prompt_template = jinja2.Template(prompt_file.read_text())
    responses = []
    with jsonlines.open(data_file) as reader:
        for prompt_data in tqdm.tqdm(reader):
            title, _, abstract = str(prompt_data["text"]).partition(".")
            prompt = prompt_template.render(
                title=title.strip(), abstract=abstract.strip(), question=_QUESTION
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs: CausalLMOutput = model(
                **{k: v.to(model.device) for k, v in inputs.items()}
            )
            logits = outputs.logits[0, -1].to("cpu")
            output = tokenizer.decode(logits.argmax(), skip_special_tokens=True)
            label = {"yes": True, "no": False}.get(output.strip().lower(), float("nan"))
            responses.append({"label": label, "output": output})

    click.echo(f"Writing responses to {output_file}")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(output_file, mode="w", compact=True, sort_keys=True) as writer:
        writer.write_all(responses)

    click.echo("Done!")


if __name__ == "__main__":
    main()
