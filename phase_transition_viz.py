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
            output_dir.joinpath("indirect_loss_trigram_losses_checkpoints.csv"), "r"
        ) as f:
        context_effect_df = pd.read_csv(f)

    context_effect_df["Ablation increase"] = context_effect_df["Ablated Loss"] - context_effect_df["Original Loss"]
    context_effect_df["Ablation increase (fraction)"] = context_effect_df["Ablation increase"] / context_effect_df["Original Loss"]
    print(f"Number of trigrams: {len(context_effect_df['Trigram'].unique())}")

    with open(
            output_dir.joinpath("general_language_loss.csv"), "r"
        ) as f:
        loss_df = pd.read_csv(f)

    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    grouped_trigram_loss = context_effect_df.groupby('Checkpoint')['Original Loss'].describe(percentiles=percentiles).reset_index()
    grouped_german_loss = loss_df.groupby('Checkpoint')['German Loss'].describe(percentiles=percentiles).reset_index()
    grouped_english_loss = loss_df.groupby('Checkpoint')['English Loss'].describe(percentiles=percentiles).reset_index()


    # Plot
    fig = make_subplots()
    shade_color_1 = 'rgba(255,127,14,0.2)'
    line_color_1 = 'rgb(255,127,14)'
    fig.add_trace(go.Scatter(x=grouped_trigram_loss['Checkpoint'], y=grouped_trigram_loss['25%'], fill=None, mode='lines', line_color=shade_color_1, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped_trigram_loss['Checkpoint'], y=grouped_trigram_loss['75%'], fill='tonexty', fillcolor=shade_color_1, line_color=shade_color_1, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped_trigram_loss['Checkpoint'], y=grouped_trigram_loss['50%'], mode='lines', line=dict(color=line_color_1, width=2), name="Trigram Loss"))
    fig.update_layout(title="German trigram (N=235) losses over checkpoints", xaxis_title="Checkpoint", yaxis_title="Loss")

    # Plot
    shade_color_2 = 'rgba(0,128,255,0.2)'
    line_color_2 = 'rgb(0,128,255)'
    fig.add_trace(go.Scatter(x=grouped_german_loss['Checkpoint'], y=grouped_german_loss['25%'], fill=None, mode='lines', line_color=shade_color_2, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped_german_loss['Checkpoint'], y=grouped_german_loss['75%'], fill='tonexty', fillcolor=shade_color_2, line_color=shade_color_2, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped_german_loss['Checkpoint'], y=grouped_german_loss['50%'], mode='lines', line=dict(color=line_color_2, width=2), name="German Loss"))
    fig.update_layout(xaxis_title="Checkpoint", yaxis_title="Loss")

    # Plot
    shade_color_2 = 'rgba(214,39,40,0.2)'
    line_color_2 = 'rgb(214,39,40)'
    fig.add_trace(go.Scatter(x=grouped_english_loss['Checkpoint'], y=grouped_english_loss['25%'], fill=None, mode='lines', line_color=shade_color_2, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped_english_loss['Checkpoint'], y=grouped_english_loss['75%'], fill='tonexty', fillcolor=shade_color_2, line_color=shade_color_2, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped_english_loss['Checkpoint'], y=grouped_english_loss['50%'], mode='lines', line=dict(color=line_color_2, width=2), name="English Loss"))
    fig.update_layout(xaxis_title="Checkpoint", yaxis_title="Loss")

    fig.update_layout(title_text="Model loss on German text, English text, and contextual trigrams")
    fig.update_layout(
        #yaxis=dict(type='log'),
        #yaxis2=dict(type='linear')
        yaxis=dict(range=[1, 13]),
        font=dict(size=24)
    )

    fig.write_image(image_dir.joinpath("phase_transition.png"), width=2000)



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

