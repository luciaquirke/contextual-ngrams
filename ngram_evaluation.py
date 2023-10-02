import random
import argparse
import pickle
import os
import gzip
from pathlib import Path
import json

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

from utils import get_model, load_language_data, get_common_tokens, generate_random_prompts, get_weird_tokens


SEED = 42
NEURON = 669
LAYER = 3


def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_data(model_name: str, output_dir: Path, data_dir: Path) -> None:
    set_seeds()
    model = get_model(model_name, 0)
    with gzip.open(
            output_dir.joinpath("checkpoint_probe_df.pkl.gz"), "rb"
        ) as f:
        probe_df = pickle.load(f)
    print("Loaded probe_df")
    checkpoints = []
    top_probe = []
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
    good_f1_neurons = accurate_f1_neurons["NeuronLabel"].unique()

    lang_data = load_language_data(data_dir)
    german_data = lang_data["de"]
    ignore_tokens, _ = get_weird_tokens(model, plot_norms=False)
    common_tokens = get_common_tokens(german_data, model, ignore_tokens, k=100)
    return probe_df, checkpoints, good_f1_neurons, german_data, common_tokens

def get_deactivate_context_hook(activation_value):
    def deactivate_neurons_hook(value, hook):
        value[:, :, NEURON] = activation_value
        return value
    return [(f'blocks.{LAYER}.mlp.hook_post', deactivate_neurons_hook)]

def get_all_non_letter_tokens(model: HookedTransformer):
    all_tokens = [i for i in range(model.cfg.d_vocab)]
    letter_tokens = []
    for token in all_tokens:
        str_token = model.to_single_str_token(token)
        if not bool(re.search(r'[a-zA-Z]', str_token)):
            letter_tokens.append(token)
    return torch.LongTensor(letter_tokens)

def get_context_effect(prompt: str | list[str], model: HookedTransformer, context_ablation_hooks: list, context_activation_hooks: list,
                      downstream_components=[], pos=None):  
    
    return_type = "loss"
    original_metric = model(prompt, return_type=return_type, loss_per_token=True)
    # 1. Activated loss: activate context
    with model.hooks(fwd_hooks=context_activation_hooks):
        activated_metric, activated_cache = model.run_with_cache(prompt, return_type=return_type, loss_per_token=True)

    # 2. Total effect: deactivate context
    with model.hooks(fwd_hooks=context_ablation_hooks):
        ablated_metric, ablated_cache = model.run_with_cache(prompt, return_type=return_type, loss_per_token=True)

    # 3. Direct effect: activate context, deactivate later components
    def deactivate_components_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value
    deactivate_components_hooks = [(freeze_act_name, deactivate_components_hook) for freeze_act_name in downstream_components]
    with model.hooks(fwd_hooks=deactivate_components_hooks+context_activation_hooks):
        direct_effect_metric = model(prompt, return_type=return_type, loss_per_token=True)

    # 4. Indirect effect: deactivate context, activate later components
    def activate_components_hook(value, hook: HookPoint):
        value = activated_cache[hook.name]
        return value         
    activate_components_hooks = [(freeze_act_name, activate_components_hook) for freeze_act_name in downstream_components]
    with model.hooks(fwd_hooks=activate_components_hooks+context_ablation_hooks):
        indirect_effect_metric = model(prompt, return_type=return_type, loss_per_token=True)

    if pos is None:
        return original_metric, activated_metric, ablated_metric, direct_effect_metric, indirect_effect_metric
    else:
        return original_metric[:, pos], activated_metric[:, pos], ablated_metric[:, pos], direct_effect_metric[:, pos], indirect_effect_metric[:, pos]

def get_trigrams(model_name, probe_df, checkpoints, german_data):
    downstream_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out")
    checkpoint = len(checkpoints)-1

    model = get_model(model_name, checkpoint)
    mean_english_activation = probe_df[(probe_df["Checkpoint"]==checkpoint) & (probe_df["NeuronLabel"]=="L3N669")]["MeanNonGermanActivation"].item()
    deactivate_context_hook = get_deactivate_context_hook(mean_english_activation)

    def get_interest_measure(original_metric, activated_metric, ablated_metric, direct_effect_metric, indirect_effect_metric):
        overall_loss_increase = (ablated_metric - original_metric).flatten()
        indirect_loss_increase = (indirect_effect_metric - original_metric).flatten()
        indirect_over_direct_loss_increase = (indirect_effect_metric - direct_effect_metric).flatten()
        position_wise_min = torch.min(torch.min(indirect_loss_increase, overall_loss_increase), indirect_over_direct_loss_increase)
        position_wise_min[original_metric.flatten() > 5] = 0
        max_interest = torch.max(position_wise_min).item()
        return position_wise_min, max_interest

    non_letter_tokens = get_all_non_letter_tokens(model)
    interesting_trigrams = []
    for prompt in tqdm(german_data):
        original_metric, activated_metric, ablated_metric, direct_effect_metric, indirect_effect_metric = get_context_effect(prompt, model, 
                            context_ablation_hooks=deactivate_context_hook, context_activation_hooks=[], downstream_components=downstream_components)
        str_prompt = model.to_str_tokens(prompt)[1:]
        interest_measure, max_interest = get_interest_measure(original_metric, activated_metric, ablated_metric, direct_effect_metric, indirect_effect_metric)
        
        measure_names = ["Original", "Ablated", "Direct Effect", "Indirect Effect"]
        interesting_positions = torch.argwhere(interest_measure>2)
        if len(interesting_positions) > 0:
            for position in interesting_positions:
                start_index = max(position-5, 0)
                end_index = min(position+5, len(str_prompt))
                measures = [original_metric.flatten().tolist()[start_index:end_index], ablated_metric.flatten().tolist()[start_index:end_index], direct_effect_metric.flatten().tolist()[start_index:end_index], indirect_effect_metric.flatten().tolist()[start_index:end_index]]
                if position>2:
                    trigram = "".join(str_prompt[position-2:position+1])
                    trigram_tokens = model.to_tokens(trigram)
                    if not torch.any(torch.isin(trigram_tokens.long().cpu(), non_letter_tokens)):
                        interesting_trigrams.append(trigram)
    return list(set(interesting_trigrams))

def calculate_trigram_losses(model_name, checkpoints, interesting_trigrams, probe_df, common_tokens):

    downstream_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out")
    data = []

    for checkpoint in [0, 10, 20, 50]:#tqdm(checkpoints): TODO
        model = get_model(model_name, checkpoint)
        mean_english_activation = probe_df[(probe_df["Checkpoint"]==checkpoint) & (probe_df["NeuronLabel"]=="L3N669")]["MeanNonGermanActivation"].item()
        deactivate_context_hook = get_deactivate_context_hook(mean_english_activation)

        for trigram in interesting_trigrams:
            prompts = generate_random_prompts(trigram, model, common_tokens, 100, 20)
            original_metric, activated_metric, ablated_metric, direct_effect_metric, indirect_effect_metric = get_context_effect(prompts, model, 
                                    context_ablation_hooks=deactivate_context_hook, context_activation_hooks=[], downstream_components=downstream_components, pos=-1)
            original_loss = original_metric.mean(0).item()
            ablated_loss = ablated_metric.mean(0).item()
            direct_loss = direct_effect_metric.mean(0).item()
            indirect_loss = indirect_effect_metric.mean(0).item()
            data.append([checkpoint, trigram, np.mean(original_loss), np.mean(ablated_loss), np.mean(direct_loss), np.mean(indirect_loss)])

    context_effect_df = pd.DataFrame(data, columns=["Checkpoint", "Trigram", "Original Loss", "Ablated Loss", "Direct Effect Loss", "Indirect Effect Loss"])
    return context_effect_df

def process_data(model_name: str, output_dir: Path, data_dir: Path):
    probe_df, checkpoints, good_f1_neurons, german_data, common_tokens = load_data(model_name, output_dir, data_dir)
    print("Loaded data")
    interesting_trigrams = get_trigrams(model_name, probe_df, checkpoints, german_data)
    print(f"Got trigrams (N={len(interesting_trigrams)})")
    with open(output_dir.joinpath("high_indirect_loss_trigrams.json"), 'w') as f:
            json.dump(interesting_trigrams, f)

    context_effect_df = calculate_trigram_losses(model_name, checkpoints, interesting_trigrams, probe_df, common_tokens)
    context_effect_df.to_csv(output_dir.joinpath("indirect_loss_trigram_losses_checkpoints.csv"), index=False)

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

