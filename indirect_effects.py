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

from utils import get_model, load_language_data, get_common_tokens, generate_random_prompts, get_weird_tokens

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)


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
    with open(
            output_dir.joinpath("checkpoint_probe_df.pkl"), "rb"
        ) as f:
        probe_df = pickle.load(f)
    print("Loaded probe_df")
    checkpoints = []
    top_probe = []
    for checkpoint in probe_df["Checkpoint"].unique():
        checkpoint_df = probe_df[probe_df["Checkpoint"] == checkpoint]
        top_probe.append(checkpoint_df["MCC"].max())
        checkpoints.append(checkpoint)

    lang_data = load_language_data(data_dir)
    german_data = lang_data["de"]
    return probe_df, checkpoints, german_data

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

def process_data(model_name: str, output_dir: Path, data_dir: Path):
    probe_df, checkpoints, german_data = load_data(model_name, output_dir, data_dir)
    print("Loaded data")

    downstream_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out")
    data = []
    for checkpoint in tqdm(range(0, len(checkpoints), 5)):
        model = get_model(model_name, checkpoint)
        mean_english_activation = probe_df[(probe_df["Checkpoint"]==checkpoint) & (probe_df["NeuronLabel"]=="L3N669")]["MeanNonGermanActivation"].item()
        deactivate_context_hook = get_deactivate_context_hook(mean_english_activation)
        original_metrics = []
        activated_metrics = []
        ablated_metrics = []
        direct_effect_metrics = []
        indirect_effect_metrics = []
        for prompt in german_data[:200]:
            original_metric, activated_metric, ablated_metric, direct_effect_metric, indirect_effect_metric = get_context_effect(prompt, model, 
                            context_ablation_hooks=deactivate_context_hook, context_activation_hooks=[], downstream_components=downstream_components)
            original_metrics.extend(original_metric.flatten().tolist())
            activated_metrics.extend(activated_metric.flatten().tolist())
            ablated_metrics.extend(ablated_metric.flatten().tolist())
            direct_effect_metrics.extend(direct_effect_metric.flatten().tolist())
            indirect_effect_metrics.extend(indirect_effect_metric.flatten().tolist())
        data.append([checkpoint, np.mean(original_metrics), np.mean(ablated_metrics) - np.mean(original_metrics), np.mean(direct_effect_metrics) - np.mean(original_metrics), np.mean(indirect_effect_metrics) - np.mean(original_metrics)])

    context_effect_df = pd.DataFrame(data, columns=["Checkpoint", "Original Loss", "Total Effect", "Direct Effect", "Indirect Effect"])
    with open(output_dir.joinpath("context_effect_split.csv"), 'w') as f:
        context_effect_df.to_csv(f, index=False)

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

