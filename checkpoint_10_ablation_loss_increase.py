import gzip
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from utils import get_model, load_language_data, eval_loss


def get_good_mcc_neurons(probe_df: pd.DataFrame):
    neurons = probe_df[
        (probe_df["MCC"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "MCC"]].copy()
    neurons = neurons.sort_values(by="MCC", ascending=False)
    good_neurons = neurons["NeuronLabel"].unique()[:50]

    return good_neurons


def load_probe_data(save_path):
    with gzip.open(save_path.joinpath("checkpoint_probe_df.pkl.gz"), "rb") as f:
        return pickle.load(f)


def main(model_name: str, save_path: Path, output_path: Path):
    model = get_model(model_name, 0)
    layers = range(model.cfg.n_layers)

    probe_df = load_probe_data(save_path)
    good_neurons = get_good_mcc_neurons(probe_df)

    layer_ablations = {i: {"Neurons": [], "English": [], "German": []} for i in layers}
    checkpoint = 10

    for neuron_label in good_neurons:
        layer, neuron = neuron_label[1:].split("N")
        layer, neuron = int(layer), int(neuron)
        mean_activation_german = probe_df[
            (probe_df["Checkpoint"] == checkpoint)
            & (probe_df["Layer"] == layer)
            & (probe_df["Neuron"] == neuron)
        ]["MeanGermanActivation"].item()
        mean_activation_english = probe_df[
            (probe_df["Checkpoint"] == checkpoint)
            & (probe_df["Layer"] == layer)
            & (probe_df["Neuron"] == neuron)
        ]["MeanEnglishActivation"].item()
        layer_ablations[layer]["Neurons"].append(neuron)
        layer_ablations[layer]["English"].append(mean_activation_english)
        layer_ablations[layer]["German"].append(mean_activation_german)

    def get_layer_ablation_hook(neurons, activation, layer):
        neurons = torch.LongTensor(neurons)
        activations = torch.FloatTensor(activation).cuda()
        assert neurons.shape == activations.shape

        def layer_ablation_hook(value, hook):
            value[:, :, neurons] = activations

        hook_point = f"blocks.{layer}.mlp.hook_post"
        return [(hook_point, layer_ablation_hook)]

    ablate_german_hooks = []
    for layer in layers:
        ablate_german_hooks += get_layer_ablation_hook(
            layer_ablations[layer]["Neurons"], layer_ablations[layer]["English"], layer
        )
    ablate_english_hooks = []
    for layer in layers:
        ablate_english_hooks += get_layer_ablation_hook(
            layer_ablations[layer]["Neurons"], layer_ablations[layer]["German"], layer
        )

    model = get_model(model_name, checkpoint)
    german_loss = eval_loss(model, german_data[:200], mean=False)
    with model.hooks(ablate_german_hooks):
        german_loss_ablated = eval_loss(model, german_data[:200], mean=False)
    english_loss = eval_loss(model, english_data[:200], mean=False)
    with model.hooks(ablate_english_hooks):
        english_loss_ablated = eval_loss(model, english_data[:200], mean=False)

    # %%
    losses = [
        [ablated - orig for ablated, orig in zip(german_loss_ablated, german_loss)],
        [ablated - orig for ablated, orig in zip(english_loss_ablated, english_loss)],
    ]
    names = ["German", "English"]

    # %%
    # Calculate mean and 95% CI
    Z = 1.96  # Z-score for 95% confidence
    means = [np.mean(loss) for loss in losses]
    ci_95 = [Z * (np.std(loss) / np.sqrt(len(loss))) for loss in losses]

    # Create bar plot
    fig = go.Figure(
        data=[
            go.Bar(
                name="Loss",
                x=names,
                y=means,
                error_y=dict(type="data", array=ci_95, visible=True),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="Ablation Loss Increase by Language for Checkpoint 10",
        font=dict(size=24, family="Times New Roman, Times, serif"),
        xaxis_title="Language",
        axis_title="Loss increase",
        width=600,
    )

    fig.save_image(output_path.joinpath("ablation_loss_increase.png"))
