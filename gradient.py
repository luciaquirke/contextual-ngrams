import random
import argparse
import gzip
import pickle
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import plotly.express as px
import plotly.graph_objects as go

from utils import get_model, preload_models, load_language_data


def get_checkpoints_df(model_name: str, num_checkpoints: int, lang_data: dict, hook_name="post"):
    def get_backward_activations(prompts, model, layer, neurons):
        bwd_activations = []
        hook_name = f"blocks.{layer}.mlp.hook_{hook_name}"

        def bwd_hook(value, hook):
            bwd_activations.append(value[0, :, neurons].detach())

        for prompt in prompts:
            with model.hooks(bwd_hooks=[(hook_name, bwd_hook)]):
                x = model(prompt, return_type="loss")
                x.backward()
        return torch.concat(bwd_activations, dim=0).mean(0).tolist()
    
    model = get_model(model_name, 0)

    all_checkpoint_dfs = []
    checkpoints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22, 23, 24, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, num_checkpoints-1]
    with tqdm(total=len(checkpoints)*model.cfg.n_layers) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for layer in range(model.cfg.n_layers):
                neurons = [i for i in range(model.cfg.d_mlp)]
                bwd_activations_german = get_backward_activations(lang_data['de'][:40], model, layer, neurons)
                bwd_activations_english = get_backward_activations(lang_data['en'][:60], model, layer, neurons)
                check_point_df = pd.DataFrame({
                    "GermanGradients": bwd_activations_german, "EnglishGradients": bwd_activations_english, 
                    "Neuron": neurons, "Layer": layer, "Checkpoint": checkpoint, "Label": [f"L{layer}N{i}" for i in neurons]})
                all_checkpoint_dfs.append(check_point_df)
                pbar.update(1)    
    
    checkpoint_df = pd.concat(all_checkpoint_dfs)
    checkpoint_df["GradientDiff"] = checkpoint_df["EnglishGradients"] - checkpoint_df["GermanGradients"]
    checkpoint_df = checkpoint_df.sort_values(by="Checkpoint")
    checkpoint_df.to_csv(f"./data/gradients_{hook_name}.csv", index=False)
    return checkpoint_df


def load_checkpoints_df(hook_name="post"):
    return pd.read_csv(f"./data/gradients_{hook_name}.csv")


def get_good_mcc_neurons(save_path: Path):
    with gzip.open(
        save_path.joinpath("checkpoint_probe_df.pkl.gz"), "rb"
    ) as f:
        probe_df = pickle.load(f)

    neurons = probe_df[(probe_df["MCC"] > 0.85) & (probe_df["MeanGermanActivation"]>probe_df["MeanNonGermanActivation"])][["NeuronLabel", "MCC"]].copy()
    neurons = neurons.sort_values(by="MCC", ascending=False)
    good_neurons = neurons["NeuronLabel"].unique()[:50]

    return good_neurons


def save_figures(checkpoint_df: pd.DataFrame, model_name: str, good_neurons, output_path: Path):
    model = get_model(model_name, 0)

    fig = px.scatter(checkpoint_df.groupby("Label").mean().reset_index(), x="GermanGradients", y="NonGermanGradients", color="Checkpoint", 
            height=800, width=1000,
            hover_name="Label", title="German vs NonGerman Backward Activations")
    axis_range = [-0.00007, 0.00007]
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    fig.save_image(output_path.joinpath("german_vs_non_german_backward_activations.png"))

    x = checkpoint_df.groupby("Label").mean()["GradientDiff"].sort_values(ascending=False)
    fig = px.histogram(x, title="Mean (NonGerman - German) Gradient per Neuron", histnorm="probability", width=900)
    fig.save_image(output_path.joinpath("mean_neuron_gradient_diff.png"))

    fig = px.line(checkpoint_df[checkpoint_df["Label"]=="L0N341"], y=["GradientDiff", "NonGermanGradients", "GermanGradients"], x="Checkpoint", width=800, title="Gradient L3N669 between NonGerman and German", facet_col="Layer", facet_col_wrap=3)
    fig.save_image(output_path.joinpath("mean_neuron_gradient_diff_L3N669.png"))

    checkpoint_df.sort_values(by=["Checkpoint", "Label"], inplace=True)
    fig = px.line(checkpoint_df[checkpoint_df["Label"].isin(good_neurons)], y="GradientDiff", x="Checkpoint", color="Label", width=800, title="Gradient difference NonGerman-German (post)")
    fig.save_image(output_path.joinpath("top_ctx_neurons_gradient_diff_post.png"))

    baseline_neurons = [f"L{random.randint(0, model.cfg.n_layers)}N{random.randint(0, model.cfg.d_mlp)}" for _ in range(25)]

    fig = px.line(checkpoint_df[checkpoint_df["Label"].isin(baseline_neurons)], y="GermanGradients", x="Checkpoint", color="Label", width=800, title="German gradients random neurons")
    fig.save_image(output_path.joinpath("random_ctx_neurons_gradient_diff_post.png"))

    checkpoint_df[["Checkpoint", "GermanGradients", "NonGermanGradients"]].groupby(["Checkpoint"]).mean().reset_index().head()

    grouped_df = checkpoint_df[["Checkpoint", "GermanGradients", "NonGermanGradients", "Layer"]].groupby(["Checkpoint", "Layer"]).mean().reset_index()
    grouped_df = grouped_df.melt(id_vars=["Checkpoint", "Layer"], value_vars=["GermanGradients", "NonGermanGradients"], var_name="Language", value_name="Gradient")
    grouped_df["LayerLanguage"] = grouped_df.apply(lambda row: f"L{row['Layer']} {row['Language']}", axis=1)

    fig = px.line(grouped_df, x="Checkpoint", y="Gradient", color="LayerLanguage")
    fig.save_image(output_path.joinpath("checkpoint_gradients.png"))

    grouped_df = checkpoint_df.loc[checkpoint_df["Layer"]>2, ["Checkpoint", "GermanGradients", "NonGermanGradients", "Layer"]].groupby(["Checkpoint"]).mean().reset_index()
    fig = px.line(grouped_df, x="Checkpoint", y=["GermanGradients", "NonGermanGradients"], title="Mean gradients for Layers 3, 4, and 5")
    fig.save_image(output_path.joinpath("mean_gradient_by_layer.png"))


def analyze_gradients(
    model_name: str, 
    layer: int,
    neuron: int,
    save_path: Path,
    data_path: Path
):
    num_checkpoints = preload_models(model_name)
    lang_data = load_language_data(data_path)
    checkpoint_df = get_checkpoints_df(model_name, num_checkpoints, lang_data)
    good_neurons = get_good_mcc_neurons(save_path)
    save_figures(checkpoint_df, model_name, good_neurons, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="pythia-70m",
        help="Name of model from TransformerLens",
    )
    parser.add_argument("--data_dir", default="data/europarl")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--layer", default=3)
    parser.add_argument("--neuron", default=669)

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)
    # TODO use this for viz / clean up 
    image_path = os.path.join(save_path, "images")
    os.makedirs(image_path, exist_ok=True)
    
    analyze_gradients(args.model, args.layer, args.neuron, Path(save_path), Path(args.data_dir))

