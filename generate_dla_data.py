import random
import argparse
import pickle
import os
import gzip
from pathlib import Path
import re

from transformer_lens import HookedTransformer
import pandas as pd
import numpy as np
import torch
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm

from neel_plotly import *

from utils import (
    get_model,
    preload_models,
    load_language_data,
    get_common_tokens,
    get_weird_tokens,
)


SEED = 42


def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_probe_data(save_path):
    with open(save_path.joinpath("checkpoint_probe_df.pkl"), "rb") as f:
        return pickle.load(f)


def get_all_non_letter_tokens(model: HookedTransformer):
    all_tokens = [i for i in range(model.cfg.d_vocab)]
    letter_tokens = []
    for token in all_tokens:
        str_token = model.to_single_str_token(token)
        if not bool(re.search(r"[a-zA-Z]", str_token)):
            letter_tokens.append(token)
    return torch.LongTensor(letter_tokens)


def process_data(model_name: str, save_path: Path, data_path: Path):
    model = get_model(model_name, 0)
    num_checkpoints = preload_models(model_name)
    lang_data = load_language_data(data_path)
    all_ignore, _ = get_weird_tokens(model, plot_norms=False)

    non_letter_tokens = get_all_non_letter_tokens(model)
    ignore_and_non_letter = torch.LongTensor(
        list(set(all_ignore.tolist()).union(set(non_letter_tokens.tolist())))
    )
    german_counts, german_tokens = get_common_tokens(
        lang_data["de"], model, ignore_and_non_letter, k=200
    )
    english_counts, english_tokens = get_common_tokens(
        lang_data["en"], model, ignore_and_non_letter, k=200
    )

    probe_df = load_probe_data(save_path)

    neurons = probe_df[
        (probe_df["F1"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "F1"]].copy()
    neurons = neurons.sort_values(by="F1", ascending=False)
    good_neurons = neurons["NeuronLabel"].unique()
    print(len(good_neurons))

    dla_data = []
    for checkpoint in tqdm(range(num_checkpoints)):
        model = get_model(model_name, checkpoint)
        for neuron_label in good_neurons:
            layer, neuron = neuron_label[1:].split("N")
            layer, neuron = int(layer), int(neuron)
            dla = model.W_out[layer, neuron, :] @ model.W_U
            german_dla = dla[german_tokens].mean().item()
            english_dla = dla[english_tokens].mean().item()
            dla_data.append(
                [
                    checkpoint,
                    german_dla,
                    english_dla,
                    german_dla - english_dla,
                    neuron_label,
                ]
            )
    dla_df = pd.DataFrame(
        dla_data,
        columns=["Checkpoint", "English DLA", "German DLA", "DLA diff", "Neuron"],
    )

    with open("dla_df_all.pkl", "wb") as f:
        pickle.dump(dla_df, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="pythia-70m",
        help="Name of model from TransformerLens",
    )
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--data_dir", default="data/europarl")

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)

    process_data(args.model, Path(save_path), Path(args.data_dir))
