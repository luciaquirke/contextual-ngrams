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
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color_2, width=2), name="Ablation Loss increase"), secondary_y=True)
    fig.update_layout(title="German trigram (N=235) loss increases from ablating L3N669", xaxis_title="Checkpoint", yaxis_title="Loss increase")

    line_color_3 = 'rgb(255,128,0)'
    context_neuron_probe_df = probe_df[probe_df["NeuronLabel"] == "L3N669"].copy()
    fig.add_trace(go.Scatter(x=context_neuron_probe_df['Checkpoint'], y=context_neuron_probe_df['F1'], name='F1 score',line=dict(color=line_color_3, width=2)), secondary_y=True)
    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="", secondary_y=True)

    fig.update_layout(title_text="Trigram evaluation, loss increase from ablating L3N699, and L3N669 F1 score over training checkpoints")
    fig.update_layout(
        #yaxis=dict(type='log'),
        #yaxis2=dict(type='linear')
        yaxis=dict(range=[1, 13]),
        yaxis2=dict(range=[-0.15, 1.15]),
        font=dict(size=24)
    )

    fig.write_image(image_dir.joinpath("figure_1.png"), width=2000)

    # fig = px.line(
    #     x=checkpoints,
    #     y=top_probe,
    #     title="Top Probe MCC by Checkpoint",
    #     width=800,
    #     height=400,
    # )
    # fig.write_image(image_dir.joinpath("top_mcc_by_checkpoint.png"), width=2000)

    # accurate_mcc_neurons = probe_df[
    #     (probe_df["MCC"] > 0.85)
    #     & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    # ][["NeuronLabel", "MCC"]].copy()
    # accurate_mcc_neurons = accurate_mcc_neurons.sort_values(by="MCC", ascending=False)
    # print(
    #     len(accurate_mcc_neurons["NeuronLabel"].unique()),
    #     "neurons with an MCC > 0.85 for German text recognition at any point during training.",
    # )
    # good_mcc_neurons = accurate_mcc_neurons["NeuronLabel"].unique()[:50]
    # accurate_f1_neurons = probe_df[
    #     (probe_df["F1"] > 0.85)
    #     & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    # ][["NeuronLabel", "F1"]].copy()
    # accurate_f1_neurons = accurate_f1_neurons.sort_values(by="F1", ascending=False)
    # print(
    #     len(accurate_f1_neurons["NeuronLabel"].unique()),
    #     "neurons with an F1 > 0.85 for German text recognition at any point during training.",
    # )
    # good_f1_neurons = accurate_f1_neurons["NeuronLabel"].unique()

    # # Melt the DataFrame
    # probe_df_melt = probe_df[probe_df["NeuronLabel"].isin(good_f1_neurons)].melt(id_vars=['Checkpoint'], var_name='NeuronLabel', value_vars="F1", value_name='F1 score')
    # probe_df_melt['F1 score'] = pd.to_numeric(probe_df_melt['F1 score'], errors='coerce')

    # # Calculate percentiles at each x-coordinate
    # percentiles = [0.25, 0.5, 0.75]
    
    # grouped = probe_df_melt.groupby('Checkpoint')['F1 score'].describe(percentiles=percentiles).reset_index()
    # L3N669_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    # # Plot
    # fig = go.Figure()

    # fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color='rgba(31,119,180,0.2)', showlegend=False))
    # fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor='rgba(31,119,180,0.2)', line_color='rgba(31,119,180,0.2)', showlegend=False))
    # fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color='rgb(31,119,180)', width=2), name="Median of Other<br>Context Neurons"))
    # fig.add_trace(go.Scatter(x=L3N669_df['Checkpoint'], y=L3N669_df['F1'], mode='lines', line=dict(color='#FF7F0E', width=2), name="L3N669"))
    # fig.update_layout(title="F1 Scores of German Context Neurons", xaxis_title="Checkpoint", yaxis_title="F1 score", font=dict(size=24))

    # fig.write_image(image_dir.joinpath("top_f1s_with_quartiles.png"), width=2000)

    # layer_vals = np.random.randint(0, model.cfg.n_layers, good_mcc_neurons.size)
    # neuron_vals = np.random.randint(0, model.cfg.d_mlp, good_mcc_neurons.size)
    # random_neurons = probe_df[
    #     (probe_df["Layer"].isin(layer_vals)) & (probe_df["Neuron"].isin(neuron_vals))
    # ]
    # random_neurons = random_neurons["NeuronLabel"].unique()

    # fig = px.line(
    #     probe_df[probe_df["NeuronLabel"].isin(good_mcc_neurons)],
    #     x="Checkpoint",
    #     y="MCC",
    #     color="NeuronLabel",
    #     title="Neurons with max MCC >= 0.85",
    # )
    # fig.write_image(image_dir.joinpath("high_mcc_neurons.png"), width=2000)

    # accurate_f1_neurons = probe_df[
    #     (probe_df["F1"] > 0.85)
    #     & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    # ][["NeuronLabel", "F1"]].copy()
    # accurate_f1_neurons = accurate_f1_neurons.sort_values(by="F1", ascending=False)
    # print(
    #     len(accurate_f1_neurons["NeuronLabel"].unique()),
    #     "neurons with an F1 > 0.85 for German text recognition at any point during training.",
    # )
    # good_f1_neurons = accurate_f1_neurons["NeuronLabel"].unique()[:50]
    # probe_df.sort_values(by=["Checkpoint", "NeuronLabel"], inplace=True)
    # fig = px.line(
    #     probe_df[probe_df["NeuronLabel"].isin(good_f1_neurons)], 
    #     x="Checkpoint", 
    #     y="F1", 
    #     color="NeuronLabel", 
    #     title="Neurons with max F1 >= 0.85", 
    #     width=800
    # )
    # fig.write_image(image_dir.joinpath("high_f1_neurons.png"), width=2000)

    # context_neuron_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    # fig = px.line(
    #     context_neuron_df,
    #     x="Checkpoint",
    #     y=["MeanGermanActivation", "MeanNonGermanActivation"],
    # )
    # fig.write_image(image_dir.joinpath("mean_activations.png"), width=2000)

    # # with gzip.open(
    # #     output_dir.joinpath("checkpoint_layer_ablation_df.pkl.gz"), "rb"
    # # ) as f:
    # #     layer_ablation_df = pickle.load(f)
    
    # # fig = px.line(
    # #     layer_ablation_df.groupby(["Checkpoint", "Layer"]).mean().reset_index(),
    # #     x="Checkpoint",
    # #     y="LossDifference",
    # #     color="Layer",
    # #     title="Loss difference for zero-ablating MLP layers on German data",
    # #     width=2000,
    # # )
    # # fig.write_image(image_dir.joinpath("layer_ablation_losses.png"), width=2000)


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
