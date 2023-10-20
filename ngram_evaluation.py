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
    get_context_effect,
    get_common_tokens,
    generate_random_prompts,
    get_weird_tokens,
    get_device,
)

device = get_device()
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)


SEED = 42
NEURON = 669
LAYER = 3


def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_data(
    model_name: str, output_dir: Path, data_dir: Path
) -> tuple[pd.DataFrame, list[int], np.ndarray, list[str], tuple[torch.Tensor, torch.Tensor]]:
    set_seeds()
    model = get_model(model_name, 0)
    with open(output_dir.joinpath("checkpoint_probe_df.pkl"), "rb") as f:
        probe_df = pickle.load(f)
    print("Loaded probe_df")
    checkpoints: list[int] = []
    top_probe: list[int] = []
    for checkpoint in probe_df["Checkpoint"].unique():
        checkpoint_df = probe_df[probe_df["Checkpoint"] == checkpoint]
        top_probe.append(checkpoint_df["MCC"].max())
        checkpoints.append(checkpoint)
    print("Checkpoint MCC analyzed")
    print(top_probe[:10], len(top_probe))
    print(checkpoints[:10], len(checkpoints))
    accurate_f1_neurons = probe_df[
        (probe_df["F1"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "F1"]].copy()
    accurate_f1_neurons = accurate_f1_neurons.sort_values(by="F1", ascending=False)
    print(
        len(accurate_f1_neurons["NeuronLabel"].unique()),
        "neurons with an F1 > 0.85 for German text recognition at any point during training.",
    )
    good_f1_neurons: np.ndarray = accurate_f1_neurons["NeuronLabel"].unique()

    lang_data = load_language_data(data_dir)
    german_data = lang_data["de"]
    ignore_tokens, _ = get_weird_tokens(model, plot_norms=False)
    common_tokens = get_common_tokens(german_data, model, ignore_tokens, k=100)
    return probe_df, checkpoints, good_f1_neurons, german_data, common_tokens


def get_deactivate_context_hook(activation_value):
    def deactivate_neurons_hook(value, hook):
        value[:, :, NEURON] = activation_value
        return value

    return [(f"blocks.{LAYER}.mlp.hook_post", deactivate_neurons_hook)]


def get_all_non_letter_tokens(model: HookedTransformer):
    all_tokens = [i for i in range(model.cfg.d_vocab)]
    letter_tokens = []
    for token in all_tokens:
        str_token = model.to_single_str_token(token)
        if not bool(re.search(r"[a-zA-Z]", str_token)):
            letter_tokens.append(token)
    return torch.LongTensor(letter_tokens)


def get_trigrams(model_name, probe_df, checkpoints, german_data):
    downstream_components = (
        "blocks.4.hook_attn_out",
        "blocks.5.hook_attn_out",
        "blocks.4.hook_mlp_out",
        "blocks.5.hook_mlp_out",
    )
    checkpoint = len(checkpoints) - 1

    model = get_model(model_name, checkpoint)
    mean_english_activation = probe_df[
        (probe_df["Checkpoint"] == checkpoint) & (probe_df["NeuronLabel"] == "L3N669")
    ]["MeanNonGermanActivation"].item()
    deactivate_context_hook = get_deactivate_context_hook(mean_english_activation)

    def get_low_indirect_effect_loss_increase_interest_measure(
        original_loss,
        activated_loss,
        ablated_loss,
        indirect_effect_removal_loss,
        direct_effect_removal_loss,
    ):
        # High ablation increase - prompts where the context neuron matters
        overall_loss_increase = (ablated_loss - original_loss).flatten()
        # High loss increase when removing the indirect effect
        indirect_loss_recovery = (
            indirect_effect_removal_loss - original_loss
        ).flatten()

        position_wise_min = torch.min(indirect_loss_recovery, overall_loss_increase)
        position_wise_min[original_loss.flatten() > 2.5] = 0
        max_interest = torch.max(position_wise_min).item()
        return position_wise_min, max_interest

    non_letter_tokens = get_all_non_letter_tokens(model)
    interesting_trigrams = []
    for prompt in tqdm(german_data):
        (
            original_loss,
            activated_loss,
            ablated_loss,
            indirect_effect_removal_loss,
            direct_effect_removal_loss,
        ) = get_context_effect(
            prompt,
            model,
            context_ablation_hooks=deactivate_context_hook,
            context_activation_hooks=[],
            downstream_components=downstream_components,
        )
        str_prompt = model.to_str_tokens(prompt)[1:]
        (
            interest_measure,
            max_interest,
        ) = get_low_indirect_effect_loss_increase_interest_measure(
            original_loss,
            activated_loss,
            ablated_loss,
            indirect_effect_removal_loss,
            direct_effect_removal_loss,
        )

        interesting_positions = torch.argwhere(interest_measure > 3)
        if len(interesting_positions) > 0:
            for position in interesting_positions:
                if position > 2:
                    trigram = "".join(str_prompt[position - 2 : position + 1])
                    trigram_tokens = model.to_tokens(trigram)
                    if not torch.any(
                        torch.isin(trigram_tokens.long().cpu(), non_letter_tokens)
                    ):
                        interesting_trigrams.append(trigram)

    interesting_trigrams = sorted(list(set(interesting_trigrams)))
    del model
    return interesting_trigrams


def calculate_trigram_losses(
    model_name, checkpoints, interesting_trigrams, probe_df, common_tokens
):
    downstream_components = (
        "blocks.4.hook_attn_out",
        "blocks.5.hook_attn_out",
        "blocks.4.hook_mlp_out",
        "blocks.5.hook_mlp_out",
    )
    data = []

    for checkpoint in tqdm(checkpoints):
        model = get_model(model_name, checkpoint)
        mean_english_activation = probe_df[
            (probe_df["Checkpoint"] == checkpoint)
            & (probe_df["NeuronLabel"] == "L3N669")
        ]["MeanNonGermanActivation"].item()
        deactivate_context_hook = get_deactivate_context_hook(mean_english_activation)

        for trigram in interesting_trigrams:
            prompts = generate_random_prompts(trigram, model, common_tokens, 100, 20)
            (
                original_metric,
                activated_metric,
                ablated_metric,
                direct_effect_metric,
                indirect_effect_metric,
            ) = get_context_effect(
                prompts,
                model,
                context_ablation_hooks=deactivate_context_hook,
                context_activation_hooks=[],
                downstream_components=downstream_components,
                pos=-1,
            )
            original_loss = original_metric.mean(0).item()
            ablated_loss = ablated_metric.mean(0).item()
            direct_loss = direct_effect_metric.mean(0).item()
            indirect_loss = indirect_effect_metric.mean(0).item()
            data.append(
                [
                    checkpoint,
                    trigram,
                    np.mean(original_loss),
                    np.mean(ablated_loss),
                    np.mean(direct_loss),
                    np.mean(indirect_loss),
                ]
            )

    context_effect_df = pd.DataFrame(
        data,
        columns=[
            "Checkpoint",
            "Trigram",
            "Original Loss",
            "Ablated Loss",
            "Direct Effect Loss",
            "Indirect Effect Loss",
        ],
    )
    return context_effect_df


def filter_trigram_candidates(
    interesting_trigrams: list[str],
    model_name: str,
    checkpoints: list[int],
    german_data: list[str],
    probe_df,
    common_tokens,
):
    checkpoint = len(checkpoints) - 1
    model = get_model(model_name, checkpoint)
    downstream_components = (
        "blocks.4.hook_attn_out",
        "blocks.5.hook_attn_out",
        "blocks.4.hook_mlp_out",
        "blocks.5.hook_mlp_out",
    )

    mean_english_activation = probe_df[
        (probe_df["Checkpoint"] == checkpoint) & (probe_df["NeuronLabel"] == "L3N669")
    ]["MeanNonGermanActivation"].item()
    deactivate_context_hook = get_deactivate_context_hook(mean_english_activation)

    original_losses = []
    ablated_losses = []
    direct_losses = []
    indirect_losses = []

    for trigram in tqdm(interesting_trigrams):
        prompts = generate_random_prompts(trigram, model, common_tokens, 100, 20)
        (
            original_metric,
            activated_metric,
            ablated_metric,
            direct_effect_metric,
            indirect_effect_metric,
        ) = get_context_effect(
            prompts,
            model,
            context_ablation_hooks=deactivate_context_hook,
            context_activation_hooks=[],
            downstream_components=downstream_components,
            pos=-1,
        )
        original_losses.append(original_metric.mean(0).item())
        ablated_losses.append(ablated_metric.mean(0).item())
        direct_losses.append(direct_effect_metric.mean(0).item())
        indirect_losses.append(indirect_effect_metric.mean(0).item())

    loss_df = pd.DataFrame(
        {
            "Trigram": interesting_trigrams,
            "Original": original_losses,
            "Ablated": ablated_losses,
            "Direct Effect": direct_losses,
            "Indirect Effect": indirect_losses,
        }
    )
    loss_df["Ablation increase"] = loss_df["Ablated"] - loss_df["Original"]
    loss_df_filtered = loss_df[
        (loss_df["Original"] < 3) & (loss_df["Ablation increase"] > 2)
    ]
    trigrams = loss_df_filtered["Trigram"].tolist()
    return trigrams


def process_data(model_name: str, output_dir: Path, data_dir: Path):
    probe_df, checkpoints, good_f1_neurons, german_data, common_tokens = load_data(
        model_name, output_dir, data_dir
    )
    print("Loaded data")

    interesting_trigrams = get_trigrams(model_name, probe_df, checkpoints, german_data)
    print(f"Got trigram candidates (N={len(interesting_trigrams)})")

    interesting_trigrams = filter_trigram_candidates(
        interesting_trigrams,
        model_name,
        checkpoints,
        german_data,
        probe_df,
        common_tokens,
    )
    print(f"Got trigrams (N={len(interesting_trigrams)})")

    with open(output_dir.joinpath("low_indirect_loss_trigrams.json"), "w") as f:
        json.dump(interesting_trigrams, f)

    context_effect_df = calculate_trigram_losses(
        model_name, checkpoints, interesting_trigrams, probe_df, common_tokens
    )
    context_effect_df.to_csv(
        output_dir.joinpath("indirect_loss_trigram_losses_checkpoints.csv"), index=False
    )


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
