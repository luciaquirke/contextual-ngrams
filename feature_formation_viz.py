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
    model = get_model(model_name, 0)
    with gzip.open(
            output_dir.joinpath("checkpoint_probe_df.pkl.gz"), "rb"
        ) as f:
        probe_df = pickle.load(f)

    checkpoints = []
    top_probe = []
    for checkpoint in probe_df["Checkpoint"].unique():
        checkpoint_df = probe_df[probe_df["Checkpoint"] == checkpoint]
        top_probe.append(checkpoint_df["MCC"].max())
        checkpoints.append(checkpoint)
    fig = px.line(
        x=checkpoints,
        y=top_probe,
        title="Top Probe MCC by Checkpoint",
        width=800,
        height=400,
    )
    fig.write_image(image_dir.joinpath("top_mcc_by_checkpoint.png"), width=2000)

    accurate_mcc_neurons = probe_df[
        (probe_df["MCC"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "MCC"]].copy()
    accurate_mcc_neurons = accurate_mcc_neurons.sort_values(by="MCC", ascending=False)
    print(
        len(accurate_mcc_neurons["NeuronLabel"].unique()),
        "neurons with an MCC > 0.85 for German text recognition at any point during training.",
    )
    good_mcc_neurons = accurate_mcc_neurons["NeuronLabel"].unique()[:50]
    accurate_f1_neurons = probe_df[
        (probe_df["F1"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "F1"]].copy()
    accurate_f1_neurons = accurate_f1_neurons.sort_values(by="F1", ascending=False)
    print(
        len(accurate_f1_neurons["NeuronLabel"].unique()),
        "neurons with an F1 > 0.85 for German text recognition at any point during training.",
    )
    good_f1_neurons = accurate_f1_neurons["NeuronLabel"].unique()[:50]

    # Melt the DataFrame
    probe_df_melt = probe_df[probe_df["NeuronLabel"].isin(good_f1_neurons)].melt(id_vars=['Checkpoint'], var_name='NeuronLabel', value_vars="F1", value_name='F1 score')
    probe_df_melt['F1 score'] = pd.to_numeric(probe_df_melt['F1 score'], errors='coerce')

    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    
    grouped = probe_df_melt.groupby('Checkpoint')['F1 score'].describe(percentiles=percentiles).reset_index()
    L3N669_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color='rgba(31,119,180,0.2)', showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor='rgba(31,119,180,0.2)', line_color='rgba(31,119,180,0.2)', showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color='rgb(31,119,180)', width=2), name="Median F1 Score"))
    fig.add_trace(go.Scatter(x=L3N669_df['Checkpoint'], y=L3N669_df['F1'], mode='lines', line=dict(color='#FF7F0E', width=2), name="L3N669 F1 Score"))
    fig.update_layout(title="F1 Scores of German Context Neurons", xaxis_title="Checkpoint", yaxis_title="F1 score")

    fig.write_image(image_dir.joinpath("top_f1s_with_quartiles.png"), width=2000)

    layer_vals = np.random.randint(0, model.cfg.n_layers, good_mcc_neurons.size)
    neuron_vals = np.random.randint(0, model.cfg.d_mlp, good_mcc_neurons.size)
    random_neurons = probe_df[
        (probe_df["Layer"].isin(layer_vals)) & (probe_df["Neuron"].isin(neuron_vals))
    ]
    random_neurons = random_neurons["NeuronLabel"].unique()

    fig = px.line(
        probe_df[probe_df["NeuronLabel"].isin(good_mcc_neurons)],
        x="Checkpoint",
        y="MCC",
        color="NeuronLabel",
        title="Neurons with max MCC >= 0.85",
    )
    fig.write_image(image_dir.joinpath("high_mcc_neurons.png"), width=2000)

    accurate_f1_neurons = probe_df[
        (probe_df["F1"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "F1"]].copy()
    accurate_f1_neurons = accurate_f1_neurons.sort_values(by="F1", ascending=False)
    print(
        len(accurate_f1_neurons["NeuronLabel"].unique()),
        "neurons with an F1 > 0.85 for German text recognition at any point during training.",
    )
    good_f1_neurons = accurate_f1_neurons["NeuronLabel"].unique()[:50]
    probe_df.sort_values(by=["Checkpoint", "NeuronLabel"], inplace=True)
    fig = px.line(
        probe_df[probe_df["NeuronLabel"].isin(good_f1_neurons)], 
        x="Checkpoint", 
        y="F1", 
        color="NeuronLabel", 
        title="Neurons with max F1 >= 0.85", 
        width=800
    )
    fig.write_image(image_dir.joinpath("high_f1_neurons.png"), width=2000)

    context_neuron_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    fig = px.line(
        context_neuron_df,
        x="Checkpoint",
        y=["MeanGermanActivation", "MeanNonGermanActivation"],
    )
    fig.write_image(image_dir.joinpath("mean_activations.png"), width=2000)

    with gzip.open(
        output_dir.joinpath("checkpoint_layer_ablation_df.pkl.gz"), "rb"
    ) as f:
        layer_ablation_df = pickle.load(f)
    
    fig = px.line(
        layer_ablation_df.groupby(["Checkpoint", "Layer"]).mean().reset_index(),
        x="Checkpoint",
        y="LossDifference",
        color="Layer",
        title="Loss difference for zero-ablating MLP layers on German data",
        width=900,
    )
    fig.write_image(image_dir.joinpath("layer_ablation_losses.png"))


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

