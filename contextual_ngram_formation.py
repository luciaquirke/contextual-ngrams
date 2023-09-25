import random
import argparse
import pickle
import os
import gzip
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import torch
import einops
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import plotly.express as px
from nltk import ngrams

from neel_plotly import *
import utils
from utils import get_model, preload_models


SEED = 42


def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_language_data(europarl_data_dir: Path) -> dict:
    """
    Returns: dictionary keyed by language code, containing 200 lines of each language included in the Europarl dataset.
    """
    lang_data = {}
    for file in os.listdir(europarl_data_dir):
        if file.endswith(".txt"):
            lang = file.split("_")[0]
            lang_data[lang] = utils.load_txt_data(europarl_data_dir.joinpath(file))

    for lang in lang_data.keys():
        print(lang, len(lang_data[lang]))
    return lang_data


def eval_prompts(prompts, model, pos=-1):
    '''Mean loss at position in prompts'''
    loss = model(prompts, return_type="loss", loss_per_token=True)[:, pos].mean().item()
    return loss


def get_deactivate_neurons_fwd_hooks(model: HookedTransformer, prompts: list[str], layer: int, neuron: int):
    mean_activation_inactive = get_mean_activation(model, data, layer, neuron)
    def deactivate_neurons_hook(value, hook):
        value[:, :, neuron] = mean_activation_inactive
        return value

    return [(f'blocks.{layer}.mlp.hook_post', deactivate_neurons_hook)]


def get_mean_activation(
    model: HookedTransformer, prompts: list[str], layer: int, neuron: int
) -> torch.Tensor:
    act_label = f"blocks.{layer}.mlp.hook_post"
    acts = []
    for prompt in prompts:
        with model.hooks([(act_label, save_activation)]):
            model(prompt)
            act = model.hook_dict[act_label].ctx["activation"][:, 10:400, neuron]
        act = einops.rearrange(act, "batch pos -> (batch pos)")
        acts.extend(act.tolist())
    return np.mean(acts)


def get_ngram_losses(
    model: HookedTransformer,
    checkpoint: int,
    ngrams: list[str],
    common_tokens: list[str],
    fwd_hooks: list[tuple[str, callable]],
) -> pd.DataFrame:
    data = []
    for ngram in ngrams:
        prompts = utils.generate_random_prompts(
            ngram, model, common_tokens, 100, 20
        )
        loss = eval_prompts(prompts, model)
        with model.hooks(fwd_hooks):
            ablated_loss = eval_prompts(prompts, model)
        data.append([loss, ablated_loss, ablated_loss - loss, checkpoint, ngram])

    df = pd.DataFrame(
        data,
        columns=["OriginalLoss", "AblatedLoss", "LossIncrease", "Checkpoint", "Ngram"],
    )
    return df


def get_common_ngrams(
    model: HookedTransformer, prompts: list[str], n: int, top_k=100
) -> list[str]:
    """
    n: n-gram length
    top_k: number of n-grams to return

    Returns: List of most common n-grams in prompts sorted by frequency
    """
    all_ngrams = []
    for prompt in tqdm(prompts):
        str_tokens = model.to_str_tokens(prompt)
        all_ngrams.extend(ngrams(str_tokens, n))
    # Filter n-grams which contain punctuation
    all_ngrams = [
        x
        for x in all_ngrams
        if all(
            [
                y.strip() not in ["\n", "-", "(", ")", ".", ",", ";", "!", "?", ""]
                for y in x
            ]
        )
    ]
    return Counter(all_ngrams).most_common(top_k)


def eval_loss(model, data, mean=True):
    '''Mean of mean of token losses for each prompt'''
    losses = []
    for prompt in data:
        loss = model(prompt, return_type="loss")
        losses.append(loss.item())
    if mean:
        return np.mean(losses)
    return losses


def eval_checkpoint(model: HookedTransformer, probe_df: pd.DataFrame, german_data: list[str], checkpoint: int, layer: int, neuron: int):
    model = get_model(checkpoint)
    german_loss = eval_loss(model, german_data)
    f1, mcc = probe_df[(probe_df['Checkpoint']==checkpoint)&(probe_df['Layer']==layer)&(probe_df['Neuron']==neuron)][['F1', 'MCC']].values[0]
    return [checkpoint, german_loss, f1, mcc]


def get_common_tokens(model, prompts):
    all_ignore, _ = utils.get_weird_tokens(model, plot_norms=False)    

    common_tokens = utils.get_common_tokens(
        prompts, model, all_ignore, k=100
    )
    return common_tokens


def get_random_trigrams(model, prompts, k=20):
    top_trigrams = get_common_ngrams(model, prompts, 3, 200)
    random_trigram_indices = np.random.choice(
        range(len(top_trigrams)), k, replace=False
    )
    random_trigrams = ["".join(top_trigrams[i][0]) for i in random_trigram_indices]
    return random_trigrams


def build_dfs(
        model_name: HookedTransformer,
        lang_data: dict,
        probe_df: pd.DataFrame,
        num_checkpoints: int, 
        layer: int,
        neuron: int,
        save_path: Path 
    ):
    german_data = lang_data["de"]
    non_german_data = np.concatenate([lang_data[lang] for lang in lang_data.keys() if lang != "de"])
    np.random.shuffle(non_german_data)
    non_german_data = non_german_data[:200].tolist()

    temp_model = get_model(model_name, 0)
    n_layers = temp_model.cfg.n_layers

    common_tokens = get_common_tokens(temp_model, german_data)
    random_trigrams = get_random_trigrams(temp_model, german_data)

    ngram_loss_dfs = []
    context_neuron_data = []
    logit_attrs = []

    deactivate_neurons_fwd_hooks = get_deactivate_neurons_fwd_hooks(model, german_data, layer, neuron)

    for checkpoint in tqdm(range(num_checkpoints)):
        model = get_model(checkpoint)

        ngram_loss_dfs.append(
            get_ngram_losses(model, checkpoint, random_trigrams, common_tokens, deactivate_neurons_fwd_hooks)
        )

        data = eval_checkpoint(model, probe_df, german_data, checkpoint, layer, neuron)
        with model.hooks(deactivate_neurons_fwd_hooks):
            german_ablated_loss = eval_loss(model, german_data)
        non_german_loss = eval_loss(model, non_german_data)
        data.extend([german_ablated_loss, non_german_loss])
        context_neuron_data.append(data)

        logit_attribution, labels = utils.pos_batch_DLA(german_data, model)
        logit_attrs.append(logit_attribution.cpu().numpy())

    context_neuron_df = pd.DataFrame(context_neuron_data, columns=["Checkpoint", "GermanLoss", "F1", "MCC", 'german_ablation_loss', 'non_german_ablation_loss'])
    context_neuron_df.to_csv(save_path.joinpath("checkpoint_eval.csv"), index=False)
    
    logit_attrs_df = pd.DataFrame()
    for i, logit_attribution in enumerate(logit_attrs):
        temp_df = pd.DataFrame()
        temp_df['logit_attribution'] = logit_attribution
        temp_df['checkpoint'] = [i] * len(logit_attribution)
        temp_df['index'] = range(len(logit_attribution))
        logit_attrs_df = pd.concat([logit_attrs_df, temp_df])

    # Compress with gzip using high compression and save
    with gzip.open(
        save_path.joinpath("checkpoint_ablation_data.pkl.gz"), "wb", compresslevel=9
    ) as f_out:
        pickle.dump({
            "neuron": context_neuron_df,
            "ngram": ngram_loss_dfs,
            "logit_attr": logit_attrs_df,
        }, f_out)
    

def load_probe_data(save_path):
    with gzip.open(save_path.joinpath("_checkpoint_features.pkl.gz"), "rb") as f:
        feature_data = pickle.load(f)
    return feature_data['probe']


def analyze_contextual_ngrams(
        model_name: str, 
        layer: int,
        neuron: int,
        save_path: Path,
        data_path: Path
    ):
    set_seeds()
    num_checkpoints = preload_models(model_name)
    lang_data = load_language_data(data_path)
    probe_df = load_probe_data(save_path)

    build_dfs(model_name, lang_data, probe_df, num_checkpoints, layer, neuron, save_path)


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
    parser.add_argument("--output_dir", default="contextual_ngrams_formation")
    parser.add_argument("--layer", default=3)
    parser.add_argument("--neuron", default=669)

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)
    
    analyze_contextual_ngrams(args.model, args.layer, args.neuron, Path(save_path), Path(args.data_dir))
