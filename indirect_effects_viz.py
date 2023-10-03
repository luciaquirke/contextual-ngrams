import random
import argparse
import pickle
import os
import gzip
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go

from neel_plotly import *

from utils import get_model


SEED = 42


def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def process_data(model_name: str, output_dir: Path, image_dir: Path) -> None:
    set_seeds()
    with open(
            output_dir.joinpath("checkpoint_probe_df.pkl"), "rb"
        ) as f:
        probe_df = pickle.load(f)

    checkpoints = []
    top_probe = []
    for checkpoint in probe_df["Checkpoint"].unique():
        checkpoint_df = probe_df[probe_df["Checkpoint"] == checkpoint]
        top_probe.append(checkpoint_df["MCC"].max())
        checkpoints.append(checkpoint)

    with open(
            output_dir.joinpath("context_effect_split.csv"), "r"
        ) as f:
        context_effect_df = pd.read_csv(f)

    melt_df = context_effect_df.melt(id_vars=["Checkpoint"], var_name="Type", value_name="LossIncrease", value_vars=["Direct Effect", "Indirect Effect"])
    fig = px.line(melt_df, x="Checkpoint", y="LossIncrease", color="Type", title="Direct and indirect ablation effect of L3N669 on German text")
    fig.update_layout(xaxis_title="Checkpoint", yaxis_title="Loss Increase", font=dict(size=24))

    fig.write_image(image_dir.joinpath("direct_indirect_effect.png"), width=2000)



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

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    save_image_path = os.path.join(save_path, "images")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_image_path, exist_ok=True)
    
    process_data(args.model, Path(save_path), Path(save_image_path))

