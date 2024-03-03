from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import click
import jinja2
import jsonlines
import ruamel.yaml
import torch.cuda
import tqdm
import transformers

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutput


def _generate_in_chunks(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    stop_token: str | None = None,
):
    input_ids: torch.Tensor = tokenizer.encode(prompt, return_tensors="pt").to(
        model.device
    )
    chunk_size = model.config.max_position_embeddings - max_new_tokens
    chunks = [
        input_ids[:, idx_chunk : idx_chunk + chunk_size]
        for idx_chunk in range(0, input_ids.size(1), chunk_size)
    ]

    past_key_values = None
    for idx_chunk, chunk in enumerate(chunks):
        with torch.no_grad():
            if idx_chunk < len(chunks) - 1:
                output_chunk: CausalLMOutput = model(
                    chunk, past_key_values=past_key_values
                )
                past_key_values = output_chunk.past_key_values
                continue

            output_ids = model.generate(
                chunk,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    return _get_response(tokenizer, output_ids, input_ids, stop_token)


def _get_response(
    tokenizer: transformers.PreTrainedTokenizer,
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
    stop_token: str | None,
):
    output = tokenizer.decode(output_ids[0], clean_up_tokenization_spaces=True)
    response = output[
        len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)) :
    ]
    if stop_token is not None:
        try:
            response = response[: response.find(stop_token)]
        except ValueError:
            click.echo(f"Stop token '{stop_token}' not found in response")
            pass

    return response


def _generate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    stop_token: str | None = None,
):
    input_ids: torch.Tensor = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids=input_ids.to(model.device),
        max_length=input_ids.size(1) + max_new_tokens,
        do_sample=False,
    )
    return _get_response(tokenizer, output_ids, input_ids, stop_token)


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
@click.option(
    "--examples-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)
@click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
    default="params.yaml",
)
def main(
    data_file: pathlib.Path,
    examples_file: pathlib.Path | None,
    prompt_file: pathlib.Path,
    model_dir: pathlib.Path,
    output_file: pathlib.Path,
    params_file: pathlib.Path,
):
    click.echo(f"Loading parameters from {params_file}")
    yaml = ruamel.yaml.YAML(typ="safe")
    with params_file.open() as file:
        params: dict[str, Any] = yaml.load(file)
    answers = params["answers"]
    params = params["generate"]
    template_args = {"instruction": params["instruction"]}
    if examples_file:
        click.echo(f"Using examples from {examples_file}")
        template_args["examples"] = list(jsonlines.open(examples_file, mode="r").iter())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")

    click.echo(f"Loading model from {model_dir}")
    tokenizer: transformers.GPT2Tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        model_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    model: transformers.GPT2LMHeadModel = (
        transformers.GPT2LMHeadModel.from_pretrained(
            model_dir,
            pad_token_id=tokenizer.eos_token_id,
        )
        .eval()
        .to(device)
    )

    click.echo(f"Using prompt template {prompt_file}")
    prompt_template = jinja2.Template(prompt_file.read_text().strip())
    responses = []

    # Add one to allow for space
    max_new_tokens = 1 + max(len(tokenizer.encode(x)) for x in answers)
    answer_positive = answers["positive"].strip().lower()
    with jsonlines.open(data_file) as reader, torch.no_grad():
        for prompt_data in tqdm.tqdm(reader):
            prompt = prompt_template.render(
                title=prompt_data["title"],
                abstract=prompt_data["abstract"],
                **template_args,
            )
            response = _generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                stop_token=params["stop_token"],
            )
            label = True if response.strip().lower() == answer_positive else False
            responses.append({"label": label, "response": response})

    click.echo(f"Writing responses to {output_file}")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(output_file, mode="w", compact=True, sort_keys=True) as writer:
        writer.write_all(responses)

    click.echo("Done!")


if __name__ == "__main__":
    main()
