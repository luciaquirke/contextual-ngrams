import random
import argparse
import pickle
import os
import gzip
from pathlib import Path
import json
import re

import pandas as pd
import numpy as np
import torch
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm

from neel_plotly import *

from utils import (
    get_model,
    load_language_data,
    get_common_tokens,
    generate_random_prompts,
    get_weird_tokens,
)


SEED = 42
NEURON = 669
LAYER = 3


def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_data(
    model_name: str, output_dir: Path, data_dir: Path
) -> tuple[pd.DataFrame, list, list]:
    set_seeds()
    model = get_model(model_name, 0)
    with open(output_dir.joinpath("checkpoint_probe_df.pkl"), "rb") as f:
        probe_df = pickle.load(f)
    print("Loaded probe_df")
    checkpoints = []
    top_probe = []
    for checkpoint in probe_df["Checkpoint"].unique():
        checkpoint_df = probe_df[probe_df["Checkpoint"] == checkpoint]
        top_probe.append(checkpoint_df["MCC"].max())
        checkpoints.append(checkpoint)

    lang_data = load_language_data(data_dir)
    german_data = lang_data["de"][:200]
    english_data = lang_data["en"][:200]
    return checkpoints, german_data, english_data


def eval_loss(model, data):
    """Mean of mean of token losses for each prompt"""
    losses = []
    for prompt in data:
        loss = model(prompt, return_type="loss")
        losses.append(loss.item())
    return losses


def evaluate_checkpoints(model_name, checkpoints, german_data, english_data):
    data = []
    for checkpoint in tqdm(checkpoints):
        model = get_model(model_name, checkpoint)
        german_loss = eval_loss(model, german_data)
        english_loss = eval_loss(model, english_data)
        assert len(german_loss) == len(english_loss)
        for i in range(len(german_loss)):
            data.append([checkpoint, german_loss[i], english_loss[i]])
    df = pd.DataFrame(data, columns=["Checkpoint", "German Loss", "English Loss"])
    return df


def process_data(model_name: str, output_dir: Path, data_dir: Path):
    checkpoints, german_data, english_data = load_data(model_name, output_dir, data_dir)

    df = evaluate_checkpoints(model_name, checkpoints, german_data, english_data)
    with open(output_dir.joinpath("general_language_loss.csv"), "w") as f:
        df.to_csv(f, index=False)


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
