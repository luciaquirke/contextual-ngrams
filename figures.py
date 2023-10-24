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
FIGURE_WIDTH = 1400

COLOR_1 = "rgb(99,110,250)" # Blue
COLOR_2 = "rgb(239,85,59)" # Red
COLOR_3 = "rgb(0,204,150)" # Green
COLOR_1_shaded = "rgba(99,110,250, 0.2)" # Blue
COLOR_2_shaded = "rgba(239,85,59, 0.2)" # Red
COLOR_3_shaded = "rgba(0,204,150, 0.2)" # Green

def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def process_data(model_name: str, output_dir: Path, image_dir: Path) -> None:
    set_seeds()
    with open(output_dir.joinpath("checkpoint_probe_df.pkl"), "rb") as f:
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

    num_trigrams = len(context_effect_df["Trigram"].unique())
    print(f"Number of trigrams: {num_trigrams}")

    ablation_df = pd.read_csv("data/checkpoint_ablation_data_all_neurons.csv")
    ablation_df["AblationIncrease"] = (
        ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    )

    with open(output_dir.joinpath("context_effect_split.csv"), "r") as f:
        split_effect_df = pd.read_csv(f)

    with open(output_dir.joinpath("dla_df_all.pkl"), "rb") as f:
        dla_all_df = pickle.load(f)

    with open(output_dir.joinpath("general_language_loss.csv"), "r") as f:
        loss_df = pd.read_csv(f)

    figure_1(probe_df, context_effect_df, num_trigrams, image_dir)
    figure_2(context_effect_df, num_trigrams, image_dir)
    figure_3(probe_df, image_dir)
    figure_4(ablation_df, image_dir)
    figure_5(split_effect_df, image_dir)
    figure_6(context_effect_df, num_trigrams, image_dir)
    figure_7(dla_all_df, image_dir)
    figure_8(context_effect_df, loss_df, image_dir)

def figure_1(probe_df, context_effect_df, num_trigrams: int, image_dir: Path) -> None:
    """Trigram evaluation, ablation loss, and German neuron F1 score over training"""
    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    grouped = (
        context_effect_df.groupby("Checkpoint")["Original Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    # Plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_1_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["75%"],
            fill="tonexty",
            fillcolor=COLOR_1_shaded,
            line_color=COLOR_1_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_1, width=2),
            name="Trigram loss",
        )
    )
    fig.update_layout(
        title=f"German trigram (N={num_trigrams}) losses over checkpoints",
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    grouped = (
        context_effect_df.groupby("Checkpoint")["Ablated Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_2_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["75%"],
            fill="tonexty",
            fillcolor=COLOR_2_shaded,
            line_color=COLOR_2_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_2, width=2),
            name="Ablation loss",
        )
    )
    fig.update_layout(
        title=f"German trigram (N={num_trigrams}) loss increases from ablating the German neuron",
        xaxis_title="Checkpoint",
        yaxis_title="Loss increase",
    )

    context_neuron_probe_df = probe_df[probe_df["NeuronLabel"] == "L3N669"].copy()
    fig.add_trace(
        go.Scatter(
            x=context_neuron_probe_df["Checkpoint"],
            y=context_neuron_probe_df["F1"],
            name="F1 score",
            line=dict(color=COLOR_3, width=2),
        ),
        secondary_y=True,
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="", secondary_y=True)

    fig.update_layout(
        title_text="Trigram evaluation, ablation loss, and German neuron F1 score over training"
    )
    fig.update_layout(
        # yaxis=dict(type='log'),
        # yaxis2=dict(type='linear')
        yaxis=dict(range=[0, 11.8]),
        yaxis2=dict(range=[0, 1.18]),
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    fig.write_image(image_dir.joinpath("figure_1.png"), width=FIGURE_WIDTH)

def figure_2(context_effect_df, num_trigrams: int, image_dir: Path) -> None:
    """Contextual trigram losses with and without ablating the German neuron"""
    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    grouped = (
        context_effect_df.groupby("Checkpoint")["Original Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    # Plot
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_1_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["75%"],
            fill="tonexty",
            fillcolor=COLOR_1_shaded,
            line_color=COLOR_1_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_1, width=2),
            name="Original loss",
        )
    )
    fig.update_layout(
        title=f"German trigram (N={num_trigrams}) losses over checkpoints",
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    grouped = (
        context_effect_df.groupby("Checkpoint")["Ablated Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    # Plot
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_2_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["75%"],
            fill="tonexty",
            fillcolor=COLOR_2_shaded,
            line_color=COLOR_2_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_2, width=2),
            name="Ablated loss",
        )
    )
    fig.update_layout(
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    fig.update_layout(
        title_text="Contextual trigram losses with and without ablating the German neuron"
    )
    fig.update_layout(
        # yaxis=dict(type='log'),
        # yaxis2=dict(type='linear')
        yaxis=dict(range=[0, 13]),
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    fig.write_image(image_dir.joinpath("figure_2.png"), width=FIGURE_WIDTH)

def figure_3(probe_df, image_dir: Path) -> None:
    """F1 scores of German neurons"""
    accurate_f1_neurons = probe_df[
        (probe_df["F1"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "F1"]].copy()
    accurate_f1_neurons = accurate_f1_neurons.sort_values(by="F1", ascending=False)
    num_neurons = len(accurate_f1_neurons["NeuronLabel"].unique())
    print(
        num_neurons,
        "neurons with an F1 > 0.85 for German text recognition at any point during training.",
    )
    good_f1_neurons = accurate_f1_neurons["NeuronLabel"].unique()

    # Melt the DataFrame
    probe_df_melt = probe_df[probe_df["NeuronLabel"].isin(good_f1_neurons)].melt(id_vars=['Checkpoint'], var_name='NeuronLabel', value_vars="F1", value_name='F1 score')
    probe_df_melt['F1 score'] = pd.to_numeric(probe_df_melt['F1 score'], errors='coerce')

    # Calculate percentiles at each x-coordinate
    percentiles = [0.05, 0.5, 0.95]
    
    grouped = probe_df_melt.groupby('Checkpoint')['F1 score'].describe(percentiles=percentiles).reset_index()
    L3N669_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    # Plot
    fig = go.Figure()

    line_color = COLOR_1
    shade_color = COLOR_1_shaded

    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['5%'], fill=None, mode='lines', line_color=shade_color, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['95%'], fill='tonexty', fillcolor=shade_color, line_color=shade_color, showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color=line_color, width=2), name="Median of other<br>German neurons"))
    fig.add_trace(go.Scatter(x=L3N669_df['Checkpoint'], y=L3N669_df['F1'], mode='lines', line=dict(color=COLOR_2, width=2), name="L3N669"))
    fig.update_layout(
        title=f"F1 scores of German neurons (N={num_neurons})", 
        xaxis_title="Checkpoint", 
        yaxis_title="F1 score", 
        font=dict(size=24, family="Times New Roman, Times, serif"))

    # FIGURE 3
    fig.write_image(image_dir.joinpath("figure_3.png"), width=FIGURE_WIDTH)

def figure_4(ablation_df, image_dir: Path):
    """Loss increase on German text when ablating context neurons"""
    ablation_df.sort_values(by=["Checkpoint", "Label"], inplace=True)

    L3N669_df = ablation_df[ablation_df["Label"] == "L3N669"]
    L3N669_df["Loss Increase"] = L3N669_df["AblationIncrease"]
    non_L3N669_df = ablation_df[(ablation_df["Label"] != "L3N669")]
    non_L3N669_df["Loss Increase"] = non_L3N669_df["AblationIncrease"]
    non_L3N669_df["Label"] = "Other neurons with MCC > 0.85"

    percentiles = [0, 0.5, 1]
    grouped = (
        non_L3N669_df.groupby("Checkpoint")["Loss Increase"]
        .describe(percentiles=percentiles)
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["0%"],
            fill=None,
            mode="lines",
            line_color=COLOR_1_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["100%"],
            fill="tonexty",
            fillcolor=COLOR_1_shaded,
            line_color=COLOR_1_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_1, width=2),
            name="Median of other<br>German neurons",
        )
    )
    fig.add_trace(
        go.Line(
            x=L3N669_df["Checkpoint"],
            y=L3N669_df["Loss Increase"],
            mode="lines",
            line=dict(color=COLOR_2, width=2),
            name="L3N669",
        )
    )

    fig.update_layout(
        title="Loss increase on German text when ablating context neurons",
        xaxis_title="Checkpoint",
        yaxis_title="Loss increase",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    fig.write_image(
        image_dir.joinpath("figure_4.png"), width=FIGURE_WIDTH
    )

def figure_5(split_effect_df, image_dir: Path):
    """Direct and indirect ablation effect of L3N669 on German text"""

    # Direct ablation loss = loss when running the model with indirect effect
    split_effect_df["Direct ablation loss"] = split_effect_df["Indirect Effect"]
    split_effect_df["Indirect ablation loss"] = split_effect_df["Direct Effect"]
    
    melt_df = split_effect_df.melt(
        id_vars=["Checkpoint"],
        var_name="Type",
        value_name="LossIncrease",
        value_vars=["Direct ablation loss", "Indirect ablation loss"],
    )
    fig = px.line(
        melt_df,
        x="Checkpoint",
        y="LossIncrease",
        color="Type",
        title="Direct and indirect ablation effect of L3N669 on German text",
    )
    fig.update_layout(
        xaxis_title="Checkpoint",
        yaxis_title="Loss increase",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )
    fig.update_layout(legend={'title_text':''})

    fig.write_image(image_dir.joinpath("figure_5.png"), width=FIGURE_WIDTH)

def figure_6(context_effect_df, num_trigrams: int, image_dir: Path) -> None:
    """Direct and indirect trigram ablation losses from ablating the German neuron"""
    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    grouped = (
        context_effect_df.groupby("Checkpoint")["Original Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    # Plot
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_1_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["75%"],
            fill="tonexty",
            fillcolor=COLOR_1_shaded,
            line_color=COLOR_1_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_1, width=2),
            name="Original loss",
        )
    )
    fig.update_layout(
        title=f"German trigram (N={num_trigrams}) losses over checkpoints",
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    # Direct loss = indirect effect is inactive
    grouped = (
        context_effect_df.groupby("Checkpoint")["Direct Effect Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    # Plot
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_2_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["75%"],
            fill="tonexty",
            fillcolor=COLOR_2_shaded,
            line_color=COLOR_2_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_2, width=2),
            name="Indirect ablation loss",
        )
    )

    grouped = (
        context_effect_df.groupby("Checkpoint")["Indirect Effect Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    # Plot
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_3_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["75%"],
            fill="tonexty",
            fillcolor=COLOR_3_shaded,
            line_color=COLOR_3_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Checkpoint"],
            y=grouped["50%"],
            mode="lines",
            line=dict(color=COLOR_3, width=2),
            name="Direct ablation loss",
        )
    )
    fig.update_layout(
        title_text="Direct and indirect trigram ablation losses from ablating the German neuron",
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        yaxis=dict(type='log'),
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    fig.write_image(
        image_dir.joinpath("figure_6.png"), width=FIGURE_WIDTH
    )

def figure_7(dla_all_df, image_dir):
    """Average DLA of German neuron on frequent German and English tokens"""
    dla_df = dla_all_df[dla_all_df["Neuron"].isin(["L3N669"])]
    dla_df["DLA Difference"] = dla_df["DLA diff"]

    fig = go.Figure()
    fig.add_trace(
        go.Line(
            x=dla_df["Checkpoint"],
            y=dla_df["English DLA"],
            mode="lines",
            name="German DLA",
        )
    )
    fig.add_trace(
        go.Line(
            x=dla_df["Checkpoint"],
            y=dla_df["German DLA"],
            mode="lines",
            name="English DLA",
        )
    )
    fig.update_layout(
        font=dict(size=24, family="Times New Roman, Times, serif"),
        width=FIGURE_WIDTH,
        title="Average DLA of German neuron on frequent German and English tokens",
        xaxis_title="Checkpoint",
        yaxis_title="DLA",
    )
    fig.write_image(image_dir.joinpath("figure_7.png"))

def figure_8(context_effect_df, loss_df, image_dir: Path):
    """Model loss on German text, English text, and contextual trigrams"""
    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    grouped_trigram_loss = (
        context_effect_df.groupby("Checkpoint")["Original Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    grouped_german_loss = (
        loss_df.groupby("Checkpoint")["German Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )
    grouped_english_loss = (
        loss_df.groupby("Checkpoint")["English Loss"]
        .describe(percentiles=percentiles)
        .reset_index()
    )

    # Plot
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=grouped_trigram_loss["Checkpoint"],
            y=grouped_trigram_loss["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_1_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_trigram_loss["Checkpoint"],
            y=grouped_trigram_loss["75%"],
            fill="tonexty",
            fillcolor=COLOR_1_shaded,
            line_color=COLOR_1_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_trigram_loss["Checkpoint"],
            y=grouped_trigram_loss["50%"],
            mode="lines",
            line=dict(color=COLOR_1, width=2),
            name="Trigram loss",
        )
    )
    fig.update_layout(
        title="German trigram (N=235) losses over checkpoints",
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    # Plot
    fig.add_trace(
        go.Scatter(
            x=grouped_german_loss["Checkpoint"],
            y=grouped_german_loss["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_2_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_german_loss["Checkpoint"],
            y=grouped_german_loss["75%"],
            fill="tonexty",
            fillcolor=COLOR_2_shaded,
            line_color=COLOR_2_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_german_loss["Checkpoint"],
            y=grouped_german_loss["50%"],
            mode="lines",
            line=dict(color=COLOR_2, width=2),
            name="German loss",
        )
    )
    fig.update_layout(
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        font=dict(size=24, family="Times New Roman, Times, serif"),
    )

    # Plot
    fig.add_trace(
        go.Scatter(
            x=grouped_english_loss["Checkpoint"],
            y=grouped_english_loss["25%"],
            fill=None,
            mode="lines",
            line_color=COLOR_3_shaded,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_english_loss["Checkpoint"],
            y=grouped_english_loss["75%"],
            fill="tonexty",
            fillcolor=COLOR_3_shaded,
            line_color=COLOR_3_shaded,
            name="25th-75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_english_loss["Checkpoint"],
            y=grouped_english_loss["50%"],
            mode="lines",
            line=dict(color=COLOR_3, width=2),
            name="English loss",
        )
    )
    fig.update_layout(
        title_text="Model loss on German text, English text, and contextual trigrams",
        xaxis_title="Checkpoint",
        yaxis_title="Loss",
        font=dict(size=24, family="Times New Roman, Times, serif"),
        yaxis=dict(range=[0, 12]),
    )

    fig.write_image(image_dir.joinpath("figure_8.png"), width=FIGURE_WIDTH)

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