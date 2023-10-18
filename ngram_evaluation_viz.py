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


    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    grouped = context_effect_df.groupby('Checkpoint')['Original Loss'].describe(percentiles=percentiles).reset_index()
    # Plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    shade_color_1 = 'rgba(255,0,0,0.2)'
    line_color_1 = 'rgb(255,0,0)'
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color=shade_color_1, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor=shade_color_1, line_color=shade_color_1, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_1, width=2), name="Trigram Loss"))
    fig.update_layout(title="German trigram (N=235) losses over checkpoints", xaxis_title="Checkpoint", yaxis_title="Loss")

    grouped = context_effect_df.groupby('Checkpoint')['Ablation increase (fraction)'].describe(percentiles=percentiles).reset_index()
    # Plot
    shade_color_2 = 'rgba(0,128,255,0.2)'
    line_color_2 = 'rgb(0,128,255)'
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color=shade_color_2, showlegend=False), secondary_y=True)
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor=shade_color_2, line_color=shade_color_2, name="25th-75th percentile", showlegend=False), secondary_y=True)
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_2, width=2), name="Ablation loss increase"), secondary_y=True)
    fig.update_layout(title="German trigram (N=235) loss increases from ablating L3N669", xaxis_title="Checkpoint", yaxis_title="Loss increase")

    line_color_3 = 'rgb(255,128,0)'
    context_neuron_probe_df = probe_df[probe_df["NeuronLabel"] == "L3N669"].copy()
    fig.add_trace(go.Scatter(x=context_neuron_probe_df['Checkpoint'], y=context_neuron_probe_df['F1'], name='F1 score',line=dict(color=line_color_3, width=2)), secondary_y=True)
    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="", secondary_y=True)

    fig.update_layout(title_text="Trigram evaluation, ablation loss, and German neuron F1 score over training")
    fig.update_layout(
        #yaxis=dict(type='log'),
        #yaxis2=dict(type='linear')
        yaxis=dict(range=[1, 13]),
        yaxis2=dict(range=[-0.15, 1.15]),
        font=dict(size=24)
    )

    fig.write_image(image_dir.joinpath("figure_1.png"), width=2000)

    # Calculate percentiles at each x-coordinate
    percentiles = [0.05, 0.5, 0.95]
    grouped = context_effect_df.groupby('Checkpoint')['Original Loss'].describe(percentiles=percentiles).reset_index()
    # Plot
    fig = make_subplots()
    shade_color_1 = 'rgba(255,0,0,0.2)'
    line_color_1 = 'rgb(255,0,0)'
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['5%'], fill=None, mode='lines', line_color=shade_color_1, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['95%'], fill='tonexty', fillcolor=shade_color_1, line_color=shade_color_1, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_1, width=2), name="Trigram Loss"))
    fig.update_layout(title="German trigram (N=235) losses over checkpoints", xaxis_title="Checkpoint", yaxis_title="Loss")

    grouped = context_effect_df.groupby('Checkpoint')['Ablated Loss'].describe(percentiles=percentiles).reset_index()
    # Plot
    shade_color_2 = 'rgba(0,128,255,0.2)'
    line_color_2 = 'rgb(0,128,255)'
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['5%'], fill=None, mode='lines', line_color=shade_color_2, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['95%'], fill='tonexty', fillcolor=shade_color_2, line_color=shade_color_2, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_2, width=2), name="Ablated Trigram Loss"))
    fig.update_layout(xaxis_title="Checkpoint", yaxis_title="Loss")

    fig.update_layout(title_text="Contextual trigram losses with and without ablating L3N699")
    fig.update_layout(
        #yaxis=dict(type='log'),
        #yaxis2=dict(type='linear')
        yaxis=dict(range=[1, 13]),
        font=dict(size=24)
    )

    fig.write_image(image_dir.joinpath("trigram_losses.png"), width=2000)


    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    grouped = context_effect_df.groupby('Checkpoint')['Original Loss'].describe(percentiles=percentiles).reset_index()
    # Plot
    fig = make_subplots()
    shade_color_1 = 'rgba(255,127,14,0.2)'
    line_color_1 = 'rgb(255,127,14)'
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color=shade_color_1, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor=shade_color_1, line_color=shade_color_1, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_1, width=2), name="Trigram Loss"))
    fig.update_layout(title="German trigram (N=235) losses over checkpoints", xaxis_title="Checkpoint", yaxis_title="Loss")

    grouped = context_effect_df.groupby('Checkpoint')["Indirect Effect Loss"].describe(percentiles=percentiles).reset_index()
    # Plot
    shade_color_2 = 'rgba(0,128,255,0.2)'
    line_color_2 = 'rgb(0,128,255)'
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color=shade_color_2, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor=shade_color_2, line_color=shade_color_2, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_2, width=2), name="Indirect Ablation Loss"))
    fig.update_layout(xaxis_title="Checkpoint", yaxis_title="Loss")

    grouped = context_effect_df.groupby('Checkpoint')['Direct Effect Loss'].describe(percentiles=percentiles).reset_index()
    # Plot
    shade_color_2 = 'rgba(214,39,40,0.2)'
    line_color_2 = 'rgb(214,39,40)'
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color=shade_color_2, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor=shade_color_2, line_color=shade_color_2, name="25th-75th percentile", showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_2, width=2), name="Direct Ablation Loss"))
    fig.update_layout(xaxis_title="Checkpoint", yaxis_title="Loss")

    fig.update_layout(title_text="Indirect and direct contextual trigram losses from ablating L3N699")
    fig.update_layout(
        #yaxis=dict(type='log'),
        #yaxis2=dict(type='linear')
        yaxis=dict(range=[1, 13]),
        font=dict(size=24)
    )

    fig.write_image(image_dir.joinpath("trigram_losses_direct_indirect.png"), width=2000)


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

