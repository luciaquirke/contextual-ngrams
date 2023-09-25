# %%
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int, Bool
from torch import Tensor
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import plotly.express as px 
from collections import defaultdict
import matplotlib.pyplot as plt
import re
from IPython.display import display, HTML
from datasets import load_dataset
from collections import Counter
import pickle
import os
import haystack_utils
from transformer_lens import utils
from fancy_einsum import einsum
import einops
import json
import ipywidgets as widgets
from IPython.display import display
from datasets import load_dataset
import random
import math
import random
import neel.utils as nutils
from neel_plotly import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import probing_utils
import pickle
from sklearn.metrics import matthews_corrcoef
import gzip
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotting_utils

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.autograd.set_grad_enabled(False)
#torch.set_grad_enabled(False)

%reload_ext autoreload
%autoreload 2

# %%
def get_model(checkpoint: int) -> HookedTransformer:
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
        checkpoint_index=checkpoint,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device)
    return model

NUM_CHECKPOINTS = 143
LAYER, NEURON = 3, 669
model = get_model(142)
german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

# %%
def print_loss(model, prompt):
    '''Loss heatmap of tokens in prompt'''
    loss = model(prompt, return_type="loss", loss_per_token=True)[0]
    tokens = model.to_str_tokens(prompt)[1:]
    haystack_utils.print_strings_as_html(tokens, loss.tolist(), max_value=6)

def eval_loss(model, data, mean=True):
    '''Mean of mean of token losses for each prompt'''
    losses = []
    for prompt in data:
        loss = model(prompt, return_type="loss")
        losses.append(loss.item())
    if mean:
        return np.mean(losses)
    return losses

def eval_prompts(prompts, model, pos=-1):
    '''Mean loss at position in prompts'''
    loss = model(prompts, return_type="loss", loss_per_token=True)[:, pos].mean().item()
    return loss

def get_probe_performance(model, german_data, english_data, layer, neuron, plot=False):
    german_activations = haystack_utils.get_mlp_activations(german_data, layer, model, neurons=[neuron], mean=False)[:50000]
    english_activations = haystack_utils.get_mlp_activations(english_data, layer, model, neurons=[neuron], mean=False)[:50000]
    if plot:
        haystack_utils.two_histogram(german_activations.flatten(), english_activations.flatten(), "German", "English")
    return train_probe(german_activations, english_activations)

def train_probe(german_activations, english_activations):
    labels = np.concatenate([np.ones(len(german_activations)), np.zeros(len(english_activations))])
    activations = np.concatenate([german_activations.cpu().numpy(), english_activations.cpu().numpy()])
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    x_train, x_test, y_train, y_test = train_test_split(activations, labels, test_size=0.2, random_state=SEED)
    probe = probing_utils.get_probe(x_train, y_train, max_iter=2000)
    f1, mcc = probing_utils.get_probe_score(probe, x_test, y_test)
    return f1, mcc

def eval_checkpoint(checkpoint: int):
    model = get_model(checkpoint)
    german_loss = eval_loss(model, german_data)
    f1, mcc = get_probe_performance(model, german_data, english_data, LAYER, NEURON)
    return [checkpoint, german_loss, f1, mcc]


# %%
model = get_model(NUM_CHECKPOINTS-1)
english_activations = {}
for layer in range(3, 4):
    english_activations[layer] = haystack_utils.get_mlp_activations(english_data, layer, model, mean=False)

MEAN_ACTIVATION_INACTIVE = english_activations[LAYER][:, NEURON].mean()

def deactivate_neurons_hook(value, hook):
    value[:, :, NEURON] = MEAN_ACTIVATION_INACTIVE
    return value
deactivate_neurons_fwd_hooks=[(f'blocks.{LAYER}.mlp.hook_post', deactivate_neurons_hook)]

print(MEAN_ACTIVATION_INACTIVE)

# %%
# Will take about 50GB of disk space for Pythia 70M models
def preload_models(NUM_CHECKPOINTS: int):
    for i in tqdm(range(NUM_CHECKPOINTS)):
        get_model(i)
# preload_models(NUM_CHECKPOINTS)

# %%
pile = load_dataset("NeelNanda/pile-10k", split='train')

# %%
# Probe performance for each neuron
def get_layer_probe_performance(model, checkpoint, layer):
    english_activations = haystack_utils.get_mlp_activations(pile['text'][:31], layer, model, mean=False, disable_tqdm=True)[:10000]
    german_activations = haystack_utils.get_mlp_activations(german_data[:30], layer, model, mean=False, disable_tqdm=True)[:10000]
    neuron_labels = [f'C{checkpoint}L{layer}N{i}' for i in range(model.cfg.d_mlp)]
    mean_english_activations = english_activations.mean(0).cpu().numpy()
    mean_german_activations = german_activations.mean(0).cpu().numpy()
    f1s = []
    mccs= []
    for neuron in range(model.cfg.d_mlp):
        f1, mcc = train_probe(german_activations[:, neuron].unsqueeze(-1), english_activations[:, neuron].unsqueeze(-1))
        f1s.append(f1)
        mccs.append(mcc)
    df = pd.DataFrame({"Label": neuron_labels, "Neuron": [i for i in range(model.cfg.d_mlp)], "F1": f1s, "MCC": mccs, "MeanGermanActivation": mean_german_activations, "MeanEnglishActivation": mean_english_activations})
    df["Checkpoint"] = checkpoint
    df["Layer"] = layer
    return df

def run_probe_analysis(n_layers):
    dfs = []
    checkpoints = list(range(40)) + [40,50,60,70,80,90,100, 110, 120, 130, 140]
    with tqdm(total=len(checkpoints)*n_layers) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for layer in range(n_layers):
                tmp_df = get_layer_probe_performance(model, checkpoint, layer)
                dfs.append(tmp_df)
                with open("data/layer_probe_performance_pile.pkl", "wb") as f:
                    pickle.dump(dfs, f)
                pbar.update(1)

    # Open the pickle file
    with open('./data/layer_probe_performance_pile.pkl', 'rb') as f:
        data = pickle.load(f)

    # Compress with gzip using high compression and save
    with gzip.open('./data/layer_probe_performance_pile.pkl.gz', 'wb', compresslevel=9) as f_out:
        pickle.dump(dfs, f_out)

# run_probe_analysis(model.cfg.n_layers)

def load_probe_analysis():
    with gzip.open('./data/layer_probe_performance_pile.pkl.gz', 'rb') as f:
        data = pickle.load(f)
    return data

# %%
dfs = load_probe_analysis()
probe_df = pd.concat(dfs)
probe_df["NeuronLabel"] = probe_df.apply(lambda row: f"L{row['Layer']}N{row['Neuron']}", axis=1)
probe_df.head()

# %%
probe_df[probe_df["NeuronLabel"]=="L3N669"].head(50)

# %%
checkpoints = []
top_probe = []
for checkpoint in probe_df["Checkpoint"].unique():
    tmp_df = probe_df[probe_df["Checkpoint"] == checkpoint]
    top_probe.append(tmp_df["MCC"].max())
    checkpoints.append(checkpoint)
px.line(x=checkpoints, y=top_probe, title="Top Probe MCC by Checkpoint", width=800, height=400)

# %%
neurons = probe_df[(probe_df["F1"] > 0.85) & (probe_df["MeanGermanActivation"]>probe_df["MeanEnglishActivation"])][["NeuronLabel", "MCC"]].copy()
neurons = neurons.sort_values(by="MCC", ascending=False)
print(len(neurons["NeuronLabel"].unique()))
good_neurons = neurons["NeuronLabel"].unique()[:50]

# %% [markdown]
# ### Ablate good neurons at checkpoint 10 on both English and German text

# %%
layer_ablations = {i:{"Neurons": [], "English": [], "German": []} for i in [3,4,5]}
checkpoint = 10

for neuron_label in good_neurons:
    layer, neuron = neuron_label[1:].split("N")
    layer, neuron = int(layer), int(neuron)
    mean_activation_german = probe_df[(probe_df["Checkpoint"] == checkpoint) & (probe_df["Layer"] == layer) & (probe_df["Neuron"] == neuron)]["MeanGermanActivation"].item()
    mean_activation_english = probe_df[(probe_df["Checkpoint"] == checkpoint) & (probe_df["Layer"] == layer) & (probe_df["Neuron"] == neuron)]["MeanEnglishActivation"].item()
    layer_ablations[layer]["Neurons"].append(neuron)
    layer_ablations[layer]["English"].append(mean_activation_english)
    layer_ablations[layer]["German"].append(mean_activation_german)

def get_layer_ablation_hook(neurons, activation, layer):
    neurons = torch.LongTensor(neurons)
    activations = torch.FloatTensor(activation).cuda()
    assert neurons.shape == activations.shape
    def layer_ablation_hook(value, hook):
        value[:, :, neurons] = activations
    hook_point = f'blocks.{layer}.mlp.hook_post'
    return [(hook_point, layer_ablation_hook)]

ablate_german_hooks = []
for layer in [3, 4, 5]:
    ablate_german_hooks += get_layer_ablation_hook(layer_ablations[layer]["Neurons"], layer_ablations[layer]["English"], layer)
ablate_english_hooks = []
for layer in [3, 4, 5]:
    ablate_english_hooks += get_layer_ablation_hook(layer_ablations[layer]["Neurons"], layer_ablations[layer]["German"], layer)

model = get_model(checkpoint)
german_loss = eval_loss(model, german_data[:200], mean=False)
with model.hooks(ablate_german_hooks):
    german_loss_ablated = eval_loss(model, german_data[:200], mean=False)
english_loss = eval_loss(model, english_data[:200], mean=False)
with model.hooks(ablate_english_hooks):
    english_loss_ablated = eval_loss(model, english_data[:200], mean=False)

# %%
losses = [[ablated - orig for ablated, orig in zip(german_loss_ablated, german_loss)], [ablated - orig for ablated, orig in zip(english_loss_ablated, english_loss)]]
names = ["German", "English"]

# %%
# Calculate mean and 95% CI
Z = 1.96  # Z-score for 95% confidence
means = [np.mean(loss) for loss in losses]
ci_95 = [Z * (np.std(loss) / np.sqrt(len(loss))) for loss in losses]

# Create bar plot
fig = go.Figure(data=[
    go.Bar(name='Loss', x=names, y=means, error_y=dict(type='data', array=ci_95, visible=True))
])

# Update layout
fig.update_layout(title='Ablation loss increase by Language for checkpoint 10', xaxis_title='Language', yaxis_title='Loss increase', width=600)

fig.show()


# %%
def get_mean_english(df, neuron, layer, checkpoint):
    label = f"C{checkpoint}L{layer}N{neuron}"
    df = df[df["Label"]==label]["MeanEnglishActivation"].item()
    return df

def get_mean_german(df, neuron, layer, checkpoint):
    label = f"C{checkpoint}L{layer}N{neuron}"
    df = df[df["Label"]==label]["MeanGermanActivation"].item()
    return df

get_mean_english(probe_df, 669, 3, 140)

# %%
# Ablation loss for top neurons

def run_ablation_analysis():
    ablation_data = []
    checkpoints = list(range(0, NUM_CHECKPOINTS, 10))
    print(checkpoints)
    with tqdm(total=len(checkpoints)*len(good_neurons)) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for neuron_name in good_neurons:
                layer, neuron = neuron_name[1:].split("N")
                layer, neuron = int(layer), int(neuron)
                english_activations = get_mean_english(probe_df, neuron, layer, checkpoint)
                assert english_activations is not None
                def tmp_hook(value, hook):
                    value[:, :, neuron] = english_activations
                    return value
                tmp_hooks=[(f'blocks.{layer}.mlp.hook_post', tmp_hook)]
                original_loss = eval_loss(model, german_data)
                with model.hooks(tmp_hooks):
                    ablated_loss = eval_loss(model, german_data)
                ablation_data.append([neuron_name, checkpoint, original_loss, ablated_loss])
                pbar.update(1)

    ablation_df = pd.DataFrame(ablation_data, columns=["Label", "Checkpoint", "OriginalLoss", "AblatedLoss"])
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    ablation_df.to_csv("data/checkpoint_ablation_data.csv")

def load_ablation_analysis():
    ablation_df = pd.read_csv("data/checkpoint_ablation_data.csv")
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    return ablation_df

ablation_df = load_ablation_analysis()
ablation_df.head()

# %%
# Ablation loss for group of top neurons

def get_ablation_hook(neurons, layer, activations):
    def ablate_neurons_hook(value, hook):
        value[:, :, neurons] = activations
        return value
    return [(f'blocks.{layer}.mlp.hook_post', ablate_neurons_hook)]

def get_neuron_loss(checkpoint, neurons: list[str]):
    model = get_model(checkpoint)
    ablation_neurons = {l:[] for l in range(model.cfg.n_layers)}
    for neuron_name in neurons:
        layer, neuron = neuron_name[1:].split("N")
        layer, neuron = int(layer), int(neuron)
        ablation_neurons[layer].append(neuron)
    hooks = []
    for layer in range(model.cfg.n_layers):
        activations = []
        for neuron in ablation_neurons[layer]:
            label = f"C{checkpoint}L{layer}N{neuron}"
            activation = probe_df[probe_df["Label"]==label]["MeanEnglishActivation"].item()
            assert activation is not None
            activations.append(activation)
        activations = torch.tensor(activations).cuda()
        hooks.extend(get_ablation_hook(ablation_neurons[layer], layer, activations))
    original_loss = eval_loss(model, german_data)
    with model.hooks(hooks):
        ablated_loss = eval_loss(model, german_data)
    return original_loss, ablated_loss

# all_neuron_diffs = []
# for checkpoint in list(range(0, NUM_CHECKPOINTS, 10)):
#     original_loss, ablated_loss = get_neuron_loss(checkpoint, good_neurons)
#     diff = ablated_loss - original_loss
#     print(f"Checkpoint {checkpoint}: {original_loss} -> {ablated_loss}")
#     all_neuron_diffs.append(diff)

# all_neuron_df = pd.DataFrame({"Label": "Top 50", "Checkpoint": list(range(0, NUM_CHECKPOINTS, 10)), "AblationIncrease": all_neuron_diffs})
# ablation_df = pd.concat([ablation_df, all_neuron_df])
# ablation_df.head()

# %%
ablation_df.sort_values(by=["Checkpoint", "Label"], inplace=True)
px.line(ablation_df[ablation_df["Label"].isin(good_neurons)], x="Checkpoint", y="AblationIncrease", color="Label", title="Ablation Increase on German prompts", width=800)

# %%
# Ablation loss for selected neurons

def run_random_ablation_analysis(neurons: list[tuple[int, int]]):
    ablation_data = []
    checkpoints = list(range(0, NUM_CHECKPOINTS, 10))
    print(checkpoints)
    with tqdm(total=len(checkpoints)*len(good_neurons)) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for layer, neuron in neurons:
                # layer, neuron = int(layer), int(neuron)
                # print(layer, neuron, checkpoint)
                # print(probe_df[probe_df["Label"]==f'C{checkpoint}L{layer}N{neuron}'])
                english_activations = get_mean_english(probe_df, neuron, layer, checkpoint)
                assert english_activations is not None
                def tmp_hook(value, hook):
                    value[:, :, neuron] = english_activations
                    return value
                tmp_hooks=[(f'blocks.{layer}.mlp.hook_post', tmp_hook)]
                original_loss = eval_loss(model, german_data)
                with model.hooks(tmp_hooks):
                    ablated_loss = eval_loss(model, german_data)
                ablation_data.append([f'L{layer}N{neuron}', checkpoint, original_loss, ablated_loss])
                pbar.update(1)

    random_ablation_df = pd.DataFrame(ablation_data, columns=["Label", "Checkpoint", "OriginalLoss", "AblatedLoss"])
    random_ablation_df["AblationIncrease"] = random_ablation_df["AblatedLoss"] - random_ablation_df["OriginalLoss"]
    random_ablation_df.to_csv("data/checkpoint_random_ablation_data.csv")

def load_random_ablation_analysis():
    random_ablation_df = pd.read_csv("data/checkpoint_random_ablation_data.csv")
    random_ablation_df["AblationIncrease"] = random_ablation_df["AblatedLoss"] - random_ablation_df["OriginalLoss"]
    return random_ablation_df

# %%
import numpy as np

# Pick as many random neurons as there are neurons with high MCC classifying German
layer_vals = np.random.randint(0, model.cfg.n_layers, good_neurons.size)
neuron_vals = np.random.randint(0, model.cfg.d_mlp, good_neurons.size)
random_neuron_indices = np.column_stack((layer_vals, neuron_vals))

# %%
random_ablation_df = load_random_ablation_analysis()
random_ablation_df.head()

# %%


# %%
random_neurons = probe_df[(probe_df['Layer'].isin(layer_vals)) & (probe_df['Neuron'].isin(neuron_vals))]
random_neurons = random_neurons["NeuronLabel"].unique()

# %%
random_ablation_df.sort_values(by=["Checkpoint", "Label"], inplace=True)
px.line(random_ablation_df[random_ablation_df["Label"].isin(random_neurons)], x="Checkpoint", y="AblationIncrease", color="Label", title="Ablation Increase of Random Neurons on German prompts", width=800)

# %%
good_neurons

# %%
neuron_out = model.W_out[5,1599]
token_boosts = neuron_out @ model.W_U
token_boosts[all_ignore] = -100
top_boosts, top_tokens = torch.topk(token_boosts, 200)
print(model.to_str_tokens(top_tokens[:30]))
px.line(top_boosts.cpu().numpy(), title="Token Boosts for Neuron 1599")

# %%
max_mcc = probe_df.groupby("NeuronLabel")["MCC"].max()
print(len(max_mcc[max_mcc < 0.1].index))
bad_neurons = []#max_mcc[max_mcc < 0.1].index[:10]
print(bad_neurons)

# %%
px.line(probe_df[probe_df["NeuronLabel"].isin(good_neurons) | probe_df["NeuronLabel"].isin(bad_neurons)], x="Checkpoint", y="F1", color="NeuronLabel", title="Neurons with F1 >= 0.85")

# %%
probe_df.head()

# %%
checkpoint_10_df = probe_df[probe_df["Checkpoint"] == 10]
checkpoint_10_df = checkpoint_10_df[checkpoint_10_df["F1"] > 0.85]

# %%
print(len(checkpoint_10_df))
print(checkpoint_10_df.groupby("Layer")["Label"].count())
checkpoint_10_df["GermanGreaterEnglish"] = checkpoint_10_df["MeanGermanActivation"] > checkpoint_10_df["MeanEnglishActivation"]
print(checkpoint_10_df.groupby("GermanGreaterEnglish")["Label"].count())

# %%


# %%
import plotly.graph_objects as go

# Melt the DataFrame
probe_df_melt = probe_df[probe_df["NeuronLabel"].isin(good_neurons)].melt(id_vars=['Checkpoint'], var_name='NeuronLabel', value_vars="F1", value_name='F1 score')
probe_df_melt['F1 score'] = pd.to_numeric(probe_df_melt['F1 score'], errors='coerce')

# Calculate percentiles at each x-coordinate
percentiles = [0.25, 0.5, 0.75]
grouped = probe_df_melt.groupby('Checkpoint')['F1 score'].describe(percentiles=percentiles).reset_index()
# Plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', showlegend=False))
fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line_color='rgba(0,100,80,0.2)', name="25th-75th percentile"))
fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color='rgb(0,100,80)', width=2), name="Median"))

fig.update_layout(title="F1 score of top neurons over time", xaxis_title="Checkpoint", yaxis_title="F1 score")

fig.show()


# %%
context_neuron_df = probe_df[probe_df["NeuronLabel"]=="L3N669"]
px.line(context_neuron_df, x="Checkpoint", y=["MeanGermanActivation", "MeanEnglishActivation"])

# %% [markdown]
# ## Baseline MCC predictions

# %%
# Baselines
# "Current word starts with space"
# Memorize top German tokens
# Memorize top English tokens

# %%
# Counts of space
def get_space_counts(data):
    space_counts = 0
    non_space_counts = 0
    for prompt in data:
        tokens = model.to_str_tokens(prompt)
        space_count = [1 if token.startswith(" ") else 0 for token in tokens]
        space_counts += sum(space_count)
        non_space_counts += len(space_count) - sum(space_count)
    return space_counts, non_space_counts

german_space, german_non_space = get_space_counts(german_data[:200])
english_space, english_non_space = get_space_counts(english_data[:200])
print(german_space, german_non_space, english_space, english_non_space)

# %%
model = get_model(142)
all_ignore, _ = haystack_utils.get_weird_tokens(model)
german_counts = haystack_utils.get_common_tokens(german_data[:200],model, k=model.cfg.d_vocab, ignore_tokens=all_ignore, return_unsorted_counts=True)
english_counts = haystack_utils.get_common_tokens(english_data[:200],model, k=model.cfg.d_vocab, ignore_tokens=all_ignore, return_unsorted_counts=True)

# %%
all_counts = german_counts + english_counts
labels = np.concatenate([np.ones(int(german_counts.sum().item())), np.zeros(int(english_counts.sum().item()))])
predictions = []
for i in range(len(all_counts)):
    if german_counts[i] > english_counts[i]:
        predictions.append(np.ones(int(german_counts[i].item())))
    else:
        predictions.append(np.zeros(int(german_counts[i].item())))
for i in range(len(all_counts)):
    if german_counts[i] > english_counts[i]:
        predictions.append(np.ones(int(english_counts[i].item())))
    else:
        predictions.append(np.zeros(int(english_counts[i].item())))
predictions = np.concatenate(predictions)
print(matthews_corrcoef(labels, predictions))

# %%
german_space / (german_space + german_non_space), english_space / (english_space + english_non_space)

# %%
labels = np.concatenate([np.ones(100), np.zeros(100)])
pred = np.concatenate([np.zeros(42), np.ones(58), np.zeros(77), np.ones(23)])
matthews_corrcoef(labels, pred)

# %% [markdown]
# ## L3N669 model eval

# %%
def run_context_neuron_eval():
    data = []
    for checkpoint in tqdm(range(NUM_CHECKPOINTS)):
        data.append(eval_checkpoint(checkpoint))

    df = pd.DataFrame(data, columns=["checkpoint", "german_loss", "f1", "mcc"])

    ablation_losses = []
    for checkpoint in tqdm(range(NUM_CHECKPOINTS)):
        model = get_model(checkpoint)
        with model.hooks(deactivate_neurons_fwd_hooks):
            ablated_loss = eval_loss(model, german_data)
        ablation_losses.append(ablated_loss)

    english_losses = []
    for checkpoint in tqdm(range(NUM_CHECKPOINTS)):
        model = get_model(checkpoint)
        english_loss = eval_loss(model, english_data)
        english_losses.append(english_loss)

    df["english_loss"] = english_losses
    df["ablation_loss"] = ablation_losses
    df.to_csv("data/checkpoint_eval.csv", index=False)

def load_context_neuron_eval():
    return pd.read_csv("data/checkpoint_eval.csv", index_col=0).reset_index()

df = load_context_neuron_eval()
df.head()

# %%
ablation_percent_increase = (df["ablation_loss"] - df["german_loss"]) / df["german_loss"]
print(ablation_percent_increase[100:])

# %%


# Create a subplot with 2 y-axes
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for german_loss and english_loss
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['german_loss'], name='German Loss'), secondary_y=False)
#fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['english_loss'], name='English Loss'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['ablation_loss'], name='Ablated German Loss'), secondary_y=False)

# Add traces for f1 and mcc
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['f1'], name='F1'), secondary_y=True)
#fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['mcc'], name='MCC'), secondary_y=True)

# Set y-axes titles
fig.update_yaxes(title_text="Loss", secondary_y=False)
fig.update_yaxes(title_text="Score", secondary_y=True)

fig.update_layout(title_text="Context neuron evaluation over training checkpoints")
fig.update_layout(
    #yaxis=dict(type='log'),
    #yaxis2=dict(type='linear')
    yaxis=dict(range=[0, 12]),
    yaxis2=dict(range=[0, 1.2])
)


fig.show()


# %%
for checkpoint in [8, 9, 10, 11, 12]:
    model = get_model(checkpoint)
    prompt = german_data[1]
    print_loss(model, prompt)

# %% [markdown]
# ## Vorschlägen

# %%
end_prompt = " Vorschlägen"
prompts = haystack_utils.generate_random_prompts(end_prompt, model, common_tokens, 500, length=20)
print(model.to_str_tokens(prompts[0]))

# %%
def save_losses():
    losses = []
    ablated_losses = []
    for checkpoint in tqdm(range(NUM_CHECKPOINTS)):
        model = get_model(checkpoint)
        loss = eval_prompts(prompts, model)
        losses.append(loss)
        with model.hooks(deactivate_neurons_fwd_hooks):
            loss = eval_prompts(prompts, model)
        ablated_losses.append(loss)

    with open('data/checkpoint_losses.pkl', 'wb') as f:
        pickle.dump({"losses": losses, "ablated_losses": ablated_losses}, f)

def load_losses():
    with open('data/checkpoint_losses.pkl', 'rb') as f:
        return pickle.load(f)

losses, ablated_losses = load_losses().values()
print(len(losses), len(ablated_losses))

# %%
df.head()

# %%
df["Vorschlägen loss"] = losses[:-1]
df["Ablated Vorschlägen loss"] = ablated_losses[:-1]

# Create a subplot with 2 y-axes
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for german_loss and english_loss
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['german_loss'], name='German Loss'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['ablation_loss'], name='Ablated German Loss'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['f1'], name='F1'), secondary_y=True)
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['Vorschlägen loss'], name='Vorschlägen'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['Ablated Vorschlägen loss'], name='Ablated Vorschlägen'), secondary_y=False)



# Set y-axes titles
fig.update_yaxes(title_text="Loss", secondary_y=False)
fig.update_yaxes(title_text="Score", secondary_y=True)

fig.update_layout(title_text="Vorschlägen loss over training checkpoints")
fig.update_layout(
    #yaxis=dict(type='log'),
    #yaxis2=dict(type='linear'),
    yaxis=dict(range=[0, 12]),
    yaxis2=dict(range=[0, 1.2])
)


fig.show()


# %%
logit_atts = []
checkpoints = [10, 15, 20, 25, 40, 80, 120, NUM_CHECKPOINTS]
for checkpoint in checkpoints:
    model = get_model(checkpoint)
    logit_attribution, labels = haystack_utils.pos_batch_DLA(prompts, model)
    logit_atts.append(logit_attribution.cpu().numpy())

# %%
df = pd.DataFrame()
for idx, logit_attribution in enumerate(logit_atts):
    temp_df = pd.DataFrame()
    temp_df['logit_attribution'] = logit_attribution
    temp_df['checkpoint'] = checkpoints[idx]
    temp_df['index'] = range(len(logit_attribution))
    df = pd.concat([df, temp_df])

# Plotting the line plot
fig = px.line(df, x='index', y='logit_attribution', color='checkpoint', title='Vorschlägen DLA by Checkpoint', width=1000)
fig.update_xaxes(title='Index', tickmode='array', tickvals=list(range(len(logit_atts[0]))), ticktext=labels)
fig.update_yaxes(title='Logit Attribution')
fig.show()


# %%
def save_neuron_dlas():
    neuron_dlas = []
    checkpoints = [i for i in range(20, 150, 10)]
    print(checkpoints)
    for checkpoint in tqdm(checkpoints):
        model = get_model(checkpoint)
        for layer in range(model.cfg.n_layers):
            neuron_dla = haystack_utils.pos_batch_neuron_dla(prompts, model, layer, pos=-1)
            neurons = [i for i in range(model.cfg.d_mlp)]
            names = [f"L{layer}N{n}" for n in neurons]
            tmp_df = pd.DataFrame({"DLA": neuron_dla.tolist(), "Labels": names})
            tmp_df["Checkpoint"] = checkpoint
            tmp_df["Layer"] = layer
            tmp_df["Neuron"] = neurons
            neuron_dlas.append(tmp_df)

    with open('data/checkpoint_neuron_dlas.pkl', 'wb') as f:
        pickle.dump(neuron_dlas, f)

def load_neuron_dlas():
    with open('data/checkpoint_neuron_dlas.pkl', 'rb') as f:
        return pickle.load(f)

neuron_dlas = load_neuron_dlas()

# %%
neuron_dla = pd.concat(neuron_dlas)
top_neurons = neuron_dla.groupby("Labels")["DLA"].max().sort_values(ascending=False).index[:50]
neuron_dla = neuron_dla[neuron_dla["Labels"].isin(top_neurons)]
px.scatter(neuron_dla, x="Labels", y="DLA", color="Checkpoint", title="Vorschlä->gen Top 50 neurons DLA", width=1000)

# %%
model = get_model(NUM_CHECKPOINTS)
neuron_dla = haystack_utils.pos_batch_neuron_dla(prompts, model, 3, pos=-1)
neuron_dla = neuron_dla.cpu().numpy()

# %%
print(np.sum(neuron_dla))
px.scatter(neuron_dla, title="Layer 3 Neuron DLA on Vorschlägen")

# %% [markdown]
# ## Context neuron DLA   
# 

# %%
df

# %%
context_neuron_df = probe_df[probe_df["NeuronLabel"]=="L3N669"]
px.line(context_neuron_df, x="Checkpoint", y=["MeanGermanActivation", "MeanEnglishActivation"])

# %%
def get_context_neuron_dla(prompts: list[str], model: HookedTransformer, expand_neurons=False):
    prompt_dlas = []
    for prompt in prompts:
        context_w_in = model.W_in[3, :, 669]
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens[:, 10:min(150, tokens.shape[1])]) # ignore early tokens
        accumulated_residual, labels = cache.get_full_resid_decomposition(layer=3, pos_slice=None, expand_neurons=expand_neurons, return_labels=True)
        scaled_residual_stack = cache.apply_ln_to_stack(accumulated_residual, layer = 3, pos_slice=None)
        logit_attribution = einops.einsum(scaled_residual_stack, context_w_in, "component batch pos d_model, d_model -> component batch pos").mean((1,2)).cpu().numpy()
        prompt_dlas.append(logit_attribution)
    prompt_dlas = np.stack(prompt_dlas)
    return prompt_dlas.mean(0), labels


checkpoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + list(range(20, 150, 10))
neurons_dlas = []
for checkpoint in tqdm(checkpoints):
    model = get_model(checkpoint)
    model.set_use_attn_result(True)
    #neuron_dla, labels = get_context_neuron_dla(english_data[:100], model)
    neuron_dla, labels = get_context_neuron_dla(german_data[:100], model)
    neurons_dlas.append(neuron_dla)

# %%
df = pd.DataFrame()
for idx, logit_attribution in enumerate(neurons_dlas):
    temp_df = pd.DataFrame()
    temp_df['logit_attribution'] = logit_attribution
    temp_df['checkpoint'] = checkpoints[idx]
    temp_df['index'] = range(len(logit_attribution))
    df = pd.concat([df, temp_df])

# Plotting the line plot
fig = px.line(df, x='index', y='logit_attribution', color='checkpoint', title='L3N669 W_in DLA', width=1000)
fig.update_xaxes(title='Index', tickmode='array', tickvals=list(range(len(neurons_dlas[0]))), ticktext=labels)
fig.update_yaxes(title='Logit Attribution')
fig.show()


# %%
import plotly.graph_objs as go
import pandas as pd

# Assuming 'df' is your DataFrame and it contains a 'checkpoint' column
unique_checkpoints = df['checkpoint'].unique()

fig = go.Figure()

# Add traces
for checkpoint in unique_checkpoints:
    trace = go.Scatter(x=df[df['checkpoint'] == checkpoint]['index'], 
                       y=df[df['checkpoint'] == checkpoint]['logit_attribution'], 
                       mode='lines', 
                       name=str(checkpoint),
                       visible='legendonly' if checkpoint != unique_checkpoints[0] else True)
    fig.add_trace(trace)

# Set y-axis range
y_min = df['logit_attribution'].min()
y_max = df['logit_attribution'].max()
fig.update_yaxes(range=[y_min, y_max])

# Add slider
steps = []
for i, checkpoint in enumerate(unique_checkpoints):
    step_visibility = ['legendonly'] * len(unique_checkpoints)
    step_visibility[i] = True
    step = dict(
        method='update',
        label=str(checkpoint),
        args=[
            {"visible": step_visibility},
            {"title": f"Currently selected checkpoint: {checkpoint}"}
        ]
    )
    steps.append(step)

slider = dict(steps=steps, active=0)
fig.update_layout(sliders=[slider])

# Update axes and initial title
fig.update_layout(title=f"Currently selected checkpoint: {unique_checkpoints[0]}")
fig.update_xaxes(title='Index', tickmode='array', tickvals=list(range(len(neurons_dlas[0]))), ticktext=labels)
#fig.update_yaxes(range=[-1, 1])
fig.show()


# %%
model = get_model(11)
model.set_use_attn_result(True)
mlp_dla, labels = get_context_neuron_dla(english_data[:100], model, expand_neurons=True)

# %%
mlp_0_dla = [mlp_dla[i] for i in range(len(mlp_dla)) if labels[i].startswith("L0N")]
mlp_1_dla = [mlp_dla[i] for i in range(len(mlp_dla)) if labels[i].startswith("L1N")]
mlp_2_dla = [mlp_dla[i] for i in range(len(mlp_dla)) if labels[i].startswith("L2N")]

df = pd.DataFrame()
for idx, logit_attribution in enumerate([mlp_0_dla, mlp_1_dla, mlp_2_dla]):
    temp_df = pd.DataFrame()
    temp_df['logit_attribution'] = logit_attribution
    temp_df["name"] = f"MLP{idx}"
    temp_df['index'] = range(len(logit_attribution))
    df = pd.concat([df, temp_df])

# Plotting the line plot
fig = px.line(df, x='index', y='logit_attribution', color='name', title='L3N669 W_in MLP DLA')
fig.update_yaxes(title='Logit Attribution')
fig.show()

# %% [markdown]
# ## Attention heads

# %%
layer, head, checkpoint = 0, 5, 50
patterns = []
model = get_model(checkpoint)
for prompt in german_data[:100]:
    _, cache = model.run_with_cache(prompt)
    pattern = cache["pattern", layer][:, head, :50, :50]
    patterns.append(pattern)
mean_pattern = torch.stack(patterns).mean((0, 1)).cpu().numpy()
px.imshow(mean_pattern, title=f"Mean Pattern for Layer {layer} head {head} checkpoint {checkpoint}", zmax=1, width=500)

# %%
prompt = german_data[2][:200]
_, cache = model.run_with_cache(prompt)
pattern = cache["pattern", 2][:, 5]
#px.imshow(pattern.mean(0).cpu().numpy(), title="Mean Pattern for Layer 2 Neuron 5", zmax=1)

# %%
pattern.shape

# %%
import circuitsvis as cv
display(cv.attention.attention_patterns(
        attention = pattern.cpu(),
        tokens = model.to_str_tokens(prompt),
        attention_head_names = ["L2H5"],
    ))

# %%
head_index = 5
layer = 2

W_O = model.W_O[layer, head_index]
W_V = model.W_V[layer, head_index]
W_E = model.W_E
W_U = model.W_U
from transformer_lens import FactoredMatrix
OV_circuit = W_V @ W_O
full_OV_circuit = W_E @ OV_circuit @ W_U

# %%
mean_OV = full_OV_circuit.mean(1)
print(mean_OV.shape)

# %%
px.histogram(mean_OV.cpu().numpy(), title="Mean OV Circuit Activation")

# %%
top_means, top_tokens = torch.topk(mean_OV, 30)
print(model.to_str_tokens(top_tokens))

# %% [markdown]
# ## Activation Patching

# %%
prompt = " Wir sind einig, dass der Präsident ein Statement geben wird"
corrupted_prompt = " We all are in total agreement that the president will now make a statement"
#corrupted_prompt = " We all are in total agreement that the president will now make a wird"

tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(prompt)
acts = cache["blocks.3.mlp.hook_post"][0, :, 669]
acts[acts<0]=0
print(len(acts))

# %%
from transformer_lens import patching, ActivationCache
from transformer_lens.hook_points import HookPoint
from functools import partial
from typing import Callable

# %%
def average_context_metric(
    cache: ActivationCache, 
    context_activations: Float[Tensor, "seq"] = acts,
) -> Float[Tensor, ""]:
    corrupted_acts = cache["blocks.3.mlp.hook_post"][0, :, 669]
    corrupted_acts[corrupted_acts<0]=0
    #percent= (corrupted_acts[-1]+1e-8) / (context_activations[-1]+1e-8)
    percent= (corrupted_acts[1:]+1e-8) / (context_activations[1:]+1e-8)
    percent[percent>1] = 1
    return percent.mean()

def last_pos_context_metric(
    cache: ActivationCache, 
    context_activations: Float[Tensor, "seq"] = acts,
) -> Float[Tensor, ""]:
    corrupted_acts = cache["blocks.3.mlp.hook_post"][0, :, 669]
    corrupted_acts[corrupted_acts<0]=0
    percent= (corrupted_acts[-1]+1e-8) / (context_activations[-1]+1e-8)
    percent[percent>1] = 1  
    return percent.mean()

context_metric = average_context_metric
_, cache = model.run_with_cache(corrupted_prompt)
context_metric(cache)

# %%
def patch_residual_component(
    corrupted_residual_component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    pos: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component

# %%

def get_act_patch_resid_pre(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric
) -> Float[Tensor, "layer pos"]:
    '''
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    # SOLUTION
    model.reset_hooks()
    seq_len = corrupted_tokens.size(1)
    results = torch.zeros(2, seq_len, device=device, dtype=torch.float32)

    for layer in tqdm(range(2)):
        for position in range(seq_len):
            hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
            with model.hooks([(utils.get_act_name("resid_pre", layer), hook_fn)]):
                _, corrupted_cache = model.run_with_cache(corrupted_tokens)
            results[layer, position] = patching_metric(corrupted_cache)

    return results

_, clean_cache = model.run_with_cache(prompt)
_, corrupted_cache = model.run_with_cache(corrupted_prompt)
clean_tokens = model.to_tokens(prompt)
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
corrupted_tokens = model.to_tokens(corrupted_prompt)
results = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, context_metric)

px.imshow(
    results.cpu().numpy(), 
    x=labels, 
    title="Logit Difference From Patched Residual Stream", 
    labels={"x":"Sequence Position", "y":"Layer"},
    width=600, # If you remove this argument, the plot will usually fill the available space
    zmin=0,
    zmax=1
)

# %%
def patch_attn_patterns(
    corrupted_head_vector: Float[Tensor, "batch head_index pos_q pos_k"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the attn patterns of a given head at every sequence position, using 
    the value from the clean cache.
    '''
    corrupted_head_vector[:, head_index] = clean_cache[hook.name][:, head_index]
    return corrupted_head_vector

def patch_head_vector(
    corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    '''
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

def get_act_patch_attn_head_all_pos_every(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable
) -> Float[Tensor, "layer head"]:
    '''
    Returns an array of results of patching at all positions for each head in each
    layer (using the value from the clean cache) for output, queries, keys, values
    and attn pattern in turn.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    n_layers = 3
    results = torch.zeros(5, n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    # Loop over each component in turn
    for component_idx, component in enumerate(["z", "q", "k", "v", "pattern"]):
        for layer in tqdm(range(n_layers)):
            for head in range(model.cfg.n_heads):
                # Get different hook function if we're doing attention probs
                hook_fn_general = patch_attn_patterns if component == "pattern" else patch_head_vector
                hook_fn = partial(hook_fn_general, head_index=head, clean_cache=clean_cache)
                # Get patched logits
                with model.hooks([(utils.get_act_name(component, layer), hook_fn)]):
                    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
                results[component_idx, layer, head] = patching_metric(corrupted_cache)

    return results

act_patch_attn_head_all_pos_every_own = get_act_patch_attn_head_all_pos_every(
    model,
    clean_tokens, #corrupted_tokens,
    corrupted_cache, #clean_cache,
    context_metric
)

imshow_pos(
    act_patch_attn_head_all_pos_every_own,
    facet_col=0,
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos)",
    labels={"x": "Head", "y": "Layer"},
    width=1200,
    xaxis="Head",
    yaxis="Layer",
)

# %%
def print_context_activations(prompt, model):
    _, cache = model.run_with_cache(prompt)
    tmp = cache["blocks.3.mlp.hook_post"][0, :, 669].tolist()
    str_prompt = model.to_str_tokens(prompt)
    haystack_utils.clean_print_strings_as_html(str_prompt, tmp, 4)

# %%
tmp_prompt = " Wir all are in total agreement that the Präsident will now make ein statement wird"
_, cache = model.run_with_cache(tmp_prompt)
tmp = cache["blocks.3.mlp.hook_post"][0, :, 669].tolist()
str_prompt = model.to_str_tokens(tmp_prompt)
haystack_utils.clean_print_strings_as_html(str_prompt, tmp, 4)

tmp_prompt = " Wir all are in total agreement that the president will now make ein statement wird"
_, cache = model.run_with_cache(tmp_prompt)
tmp = cache["blocks.3.mlp.hook_post"][0, :, 669].tolist()
str_prompt = model.to_str_tokens(tmp_prompt)
haystack_utils.clean_print_strings_as_html(str_prompt, tmp, 4)

tmp_prompt = " Wir all are in total agreement that the president will now make a statement wird"
_, cache = model.run_with_cache(tmp_prompt)
tmp = cache["blocks.3.mlp.hook_post"][0, :, 669].tolist()
str_prompt = model.to_str_tokens(tmp_prompt)
haystack_utils.clean_print_strings_as_html(str_prompt, tmp, 4)

tmp_prompt = " We all are in total agreement that the president will now make a statement wird"
_, cache = model.run_with_cache(tmp_prompt)
tmp = cache["blocks.3.mlp.hook_post"][0, :, 669].tolist()
str_prompt = model.to_str_tokens(tmp_prompt)
haystack_utils.clean_print_strings_as_html(str_prompt, tmp, 4)

# %%
# Check earlier heads

# l0_pattern = clean_cache["pattern", 0][:, 1]
# l1_pattern = clean_cache["pattern", 1][:, 3]
# l2_pattern = clean_cache["pattern", 2][:, 5]

l0_pattern = corrupted_cache["blocks.0.attn.hook_pattern"][:, 1]
l1_pattern = corrupted_cache["blocks.1.attn.hook_pattern"][:, 3]
l2_pattern = corrupted_cache["blocks.2.attn.hook_pattern"][:, 5]
pattern = torch.concat([l0_pattern, l1_pattern, l2_pattern], dim=0)
# Neels plot doesn't support column wrapping - swaps row order
#labels = [f"L{i}H{h}" for i in range(2, -1, -1) for h in range(8)]
labels = ["L0H1", "L1H3", "L2H5"]
pattern.shape

# %%
imshow_base(
    pattern,
    facet_col=0,
    facet_labels=labels,
    title="Attention patterns on English prompt",
    width=900,
    xaxis="Dest",
    yaxis="Source",
    height=400,
    #facet_col_wrap=8,
    zmin=0,
    zmax=1
)

# %% [markdown]
# ## Specific tokens

# %%
def full_context_metric(
    cache: ActivationCache, 
    context_activations: Float[Tensor, "seq"],
    baseline: Float[Tensor, "seq"] | None = None,
    pos: int | None = -1,
) -> Float[Tensor, ""]:
    corrupted_acts = cache["blocks.3.mlp.hook_post"][0, :, 669]
    if baseline is not None:
        corrupted_acts = corrupted_acts - baseline
    corrupted_acts[corrupted_acts<0]=0
    percent= (corrupted_acts[pos]+1e-8) / (context_activations[pos]+1e-8)
    percent[percent>1] = 1  
    return percent.mean()

# %%
prompt = " Wir sind einig, dass der Präsident ein Statement geben wird"
corrupted_prompt = " We all are in total agreement that the president will now make a statement"

# %%
prompt = " We all are in total agreement that the president will now makeäsident wird"
corrupted_prompt = " We all are in total agreement that the president will now make a wird"
print(model.to_str_tokens(model.to_tokens(prompt)))
print(model.to_str_tokens(model.to_tokens(corrupted_prompt)))
print_context_activations(prompt, model)
print_context_activations(corrupted_prompt, model)

# %%
_, clean_cache = model.run_with_cache(prompt)
_, corrupted_cache = model.run_with_cache(corrupted_prompt)
clean_tokens = model.to_tokens(prompt)
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
corrupted_tokens = model.to_tokens(corrupted_prompt)
clean_acts = clean_cache["blocks.3.mlp.hook_post"][0, :, 669]
clean_acts[clean_acts<0]=0
corrupted_acts = corrupted_cache["blocks.3.mlp.hook_post"][0, :, 669]
corrupted_acts[corrupted_acts<0]=0
acts = clean_acts

context_metric = partial(full_context_metric, context_activations=acts, baseline=corrupted_acts)

# %%
act_patch_attn_head_all_pos_every_own = get_act_patch_attn_head_all_pos_every(
    model,
    corrupted_tokens,
    clean_cache,
    context_metric
)

imshow_pos(
    act_patch_attn_head_all_pos_every_own,
    facet_col=0,
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos)",
    labels={"x": "Head", "y": "Layer"},
    width=1200,
    xaxis="Head",
    yaxis="Layer",
    zmin=0, 
    zmax=1
)

# %%
results = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, context_metric)

px.imshow(
    results.cpu().numpy(), 
    x=labels, 
    title="Logit Difference From Patched resid_pre", 
    labels={"x":"Sequence Position", "y":"Layer"},
    width=600, # If you remove this argument, the plot will usually fill the available space
    zmin=0,
    zmax=1
)

# %%
def get_act_patch_components(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric
) -> Float[Tensor, "act layer pos"]:
    # SOLUTION
    model.reset_hooks()
    seq_len = corrupted_tokens.size(1)
    act_names = ["post", "attn_out"]
    results = torch.zeros(len(act_names), 3, seq_len, device=device, dtype=torch.float32)

    for act_index, act_name in enumerate(["post", "attn_out"]):
        for layer in tqdm(range(3)):
            for position in range(seq_len):
                hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
                with model.hooks([(utils.get_act_name(act_name, layer), hook_fn)]):
                    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
                results[act_index, layer, position] = patching_metric(corrupted_cache)

    return results

act_patch_components = get_act_patch_components(
    model,
    corrupted_tokens,
    clean_cache,
    context_metric
)

imshow_pos(
    act_patch_components,
    facet_col=0,
    facet_labels=["MLP Post", "Attn Out"],
    title="Activation Patching Per Component Per Head",
    labels={"x":"Sequence Position", "y":"Layer"},
    width=1200,
    zmin=0, 
    zmax=1
)

# %%


# %% [markdown]
# ## Dataset examples

# %%
dataset = load_dataset("NeelNanda/pile-10k")
pile_prompts = [x["text"] for x in dataset["train"]]
print(len(pile_prompts))

# %%
checkpoints = [9, 10, 15, 20, 40, 80, 120]

dfs = load_probe_analysis()
df = pd.concat(dfs)
df["NeuronLabel"] = df.apply(lambda row: f"L{row['Layer']}N{row['Neuron']}", axis=1)
neurons = df[(df["MCC"] > 0.85) & (df["MeanGermanActivation"]>df["MeanEnglishActivation"])][["NeuronLabel", "MCC"]].copy()
neurons = neurons.sort_values(by="MCC", ascending=False)
print(len(neurons["NeuronLabel"].unique()))
good_neurons = neurons["NeuronLabel"].unique()[:50]
print(good_neurons)

# %%
px.line(df[df["NeuronLabel"].isin(good_neurons)], x="Checkpoint", y="MCC", color="NeuronLabel", title="Neurons with max MCC >= 0.85")

# %%
model = get_model(9)
neuron = 5, 1225

# %%
def print_activations(prompt, model, neuron, log=False):
    layer, neuron = neuron
    hook_name = utils.get_act_name("post", layer)
    _, cache = model.run_with_cache(prompt, names_filter=[hook_name])
    activations = cache[hook_name][0, :, neuron].tolist()
    str_prompt = model.to_str_tokens(prompt)
    if log:
        print("Mean activation", np.mean(activations))
    haystack_utils.clean_print_strings_as_html(str_prompt, activations, 4)

# %%
def get_max_activating_examples(prompts, model, neuron):
    layer, neuron = neuron
    hook_name = utils.get_act_name("post", layer)
    max_activation_per_prompt = []
    for prompt in tqdm(prompts):
        _, cache = model.run_with_cache(prompt, names_filter=[hook_name])
        max_activation = cache[hook_name][0, :, neuron].max().item()
        max_activation_per_prompt.append(max_activation)
    return max_activation_per_prompt

max_activation_per_prompt = get_max_activating_examples(pile_prompts[:1000], model, neuron)
max_activating_indices = np.argsort(max_activation_per_prompt)[::-1]
max_activating_prompts = [pile_prompts[i] for i in max_activating_indices]

# %%
for prompt in max_activating_prompts[:20]:
    print_activations(prompt, model, neuron)

# %% [markdown]
# ## General loss curve

# %%
german_data[:1]

# %%
types = []
for x in dataset["train"]:
    types.append(x["meta"]["pile_set_name"])

counter = Counter(types)
types = list(counter.keys())
counts = list(counter.values())

used_types = [types[i] for i in range(len(types)) if counts[i] > 30]
print(used_types)

# %%
dfs = []

checkpoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 40, 80, 120, NUM_CHECKPOINTS]
with tqdm(total=len(dataset["train"])*len(checkpoints)) as pbar:
    for checkpoint in checkpoints:
        model = get_model(checkpoint)
        losses = {type:0 for type in used_types}
        for x in dataset["train"]:
            data_type = x["meta"]["pile_set_name"]
            if data_type in used_types:
                loss = model(x["text"], return_type="loss")
                losses[data_type] += loss.item()
            pbar.update(1)
        for type in used_types:
            losses[type] /= counter[type]
        df = pd.DataFrame(losses, index=[0]).melt(value_vars=used_types, var_name="Dataset", value_name="Loss")
        df["Checkpoint"] = checkpoint
        dfs.append(df)

# %%
english_dfs = []
checkpoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 40, 80, 120, NUM_CHECKPOINTS]
with tqdm(total=len(english_data[:500])*len(checkpoints)) as pbar:
    for checkpoint in checkpoints:
        model = get_model(checkpoint)
        losses = []
        for x in english_data[:500]:
            loss = model(x, return_type="loss")
            losses.append(loss.item())
            pbar.update(1)
        df = pd.DataFrame({"Loss": np.mean(losses), "Checkpoint": checkpoint, "Dataset": "Europarl English"}, index=[0])
        english_dfs.append(df)

# %%
german_dfs = []
checkpoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 40, 80, 120, NUM_CHECKPOINTS]
with tqdm(total=len(german_data[:500])*len(checkpoints)) as pbar:
    for checkpoint in checkpoints:
        model = get_model(checkpoint)
        losses = []
        for x in german_data[:500]:
            loss = model(x, return_type="loss")
            losses.append(loss.item())
            pbar.update(1)
        df = pd.DataFrame({"Loss": np.mean(losses), "Checkpoint": checkpoint, "Dataset": "Europarl German"}, index=[0])
        german_dfs.append(df)

# %%
df = pd.concat(dfs+german_dfs+english_dfs)
df.sort_values(by="Checkpoint", inplace=True)
px.line(df, x="Checkpoint", y="Loss", color="Dataset", title="Loss on Pile Datasets", width=900)

# %% [markdown]
# ## Gradients

# %%
torch.autograd.set_grad_enabled(True)
for param in model.parameters():
    param.requires_grad = True

# %%
def get_checkpoints_df(hook_name="post"):
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
    
    model = get_model(NUM_CHECKPOINTS)
    all_checkpoint_dfs = []
    checkpoints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22, 23, 24, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, NUM_CHECKPOINTS]
    with tqdm(total=len(checkpoints)*model.cfg.n_layers) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for layer in range(model.cfg.n_layers):
                neurons = [i for i in range(model.cfg.d_mlp)]
                bwd_activations_german = get_backward_activations(german_data[:40], model, layer, neurons)
                bwd_activations_english = get_backward_activations(english_data[:60], model, layer, neurons)
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

checkpoint_df = load_checkpoints_df()

# %%
checkpoint_df[checkpoint_df["Label"]=="L3N669"]["GradientDiff"].mean()

# %%
fig = px.scatter(checkpoint_df.groupby("Label").mean().reset_index(), x="GermanGradients", y="EnglishGradients", color="Checkpoint", 
           height=800, width=1000,
           hover_name="Label", title="German vs English Backward Activations")
axis_range = [-0.00007, 0.00007]
fig.update_xaxes(scaleanchor="y", scaleratio=1)
#fig.update_yaxes(range=axis_range)

fig.show()

# %%
x = checkpoint_df.groupby("Label").mean()["GradientDiff"].sort_values(ascending=False)
px.histogram(x, title="Mean (English - German) Gradient per Neuron", histnorm="probability", width=900)

# %%
px.line(checkpoint_df[checkpoint_df["Label"]=="L0N341"], y=["GradientDiff", "EnglishGradients", "GermanGradients"], x="Checkpoint", width=800, title="Gradient L3N669 between English and German", facet_col="Layer", facet_col_wrap=3)

# %%
df.sort_values(by=["Checkpoint", "NeuronLabel"], inplace=True)
px.line(df[df["NeuronLabel"].isin(good_neurons)], x="Checkpoint", y="F1", color="NeuronLabel", title="Top german context neurons", width=800)

# %%
checkpoint_df.sort_values(by=["Checkpoint", "Label"], inplace=True)
px.line(checkpoint_df[checkpoint_df["Label"].isin(good_neurons)], y="GradientDiff", x="Checkpoint", color="Label", width=800, title="Gradient difference English-German (post)")

# %%
baseline_neurons = [f"L{random.randint(0, model.cfg.n_layers)}N{random.randint(0, model.cfg.d_mlp)}" for _ in range(25)]

# %%
px.line(checkpoint_df[checkpoint_df["Label"].isin(baseline_neurons)], y="GermanGradients", x="Checkpoint", color="Label", width=800, title="German gradients random neurons")

# %%
checkpoint_df.head()

# %%
checkpoint_df[["Checkpoint", "GermanGradients", "EnglishGradients"]].groupby(["Checkpoint"]).mean().reset_index().head()

# %%
grouped_df = checkpoint_df[["Checkpoint", "GermanGradients", "EnglishGradients", "Layer"]].groupby(["Checkpoint", "Layer"]).mean().reset_index()
grouped_df = grouped_df.melt(id_vars=["Checkpoint", "Layer"], value_vars=["GermanGradients", "EnglishGradients"], var_name="Language", value_name="Gradient")
grouped_df["LayerLanguage"] = grouped_df.apply(lambda row: f"L{row['Layer']} {row['Language']}", axis=1)
print(grouped_df.head())
px.line(grouped_df, x="Checkpoint", y="Gradient", color="LayerLanguage")

# %%
grouped_df = checkpoint_df.loc[checkpoint_df["Layer"]>2, ["Checkpoint", "GermanGradients", "EnglishGradients", "Layer"]].groupby(["Checkpoint"]).mean().reset_index()
px.line(grouped_df, x="Checkpoint", y=["GermanGradients", "EnglishGradients"], title="Mean gradients for Layers 3, 4, and 5")

# %%


# %% [markdown]
# ## Random ngrams

# %%
from nltk import ngrams
def get_german_ngrams(prompts: list[str], model: HookedTransformer, n: int, top_k=100):
    all_ngrams = []
    for prompt in tqdm(prompts):
        str_tokens = model.to_str_tokens(prompt)
        all_ngrams.extend(ngrams(str_tokens, n))
    print(len(all_ngrams))
    all_ngrams = [x for x in all_ngrams if all([y.strip() not in ["\n", "-", "(", ")", ".", ",", ";", "!", "?", ""] for y in x])]
    print(len(all_ngrams))
    return Counter(all_ngrams).most_common(top_k)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:500]
top_trigrams = get_german_ngrams(german_data, model, 3, 200)

# %%
random_ngram_indices = np.random.choice(range(len(top_trigrams)), 20, replace=False)
random_ngrams = ["".join(top_trigrams[i][0]) for i in random_ngram_indices]
random_ngrams

# %%
data = []
checkpoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + list(range(20, 150, 10))
for checkpoint in checkpoints:
    model = get_model(checkpoint)
    for ngram in random_ngrams:
        prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, 100, 20)
        loss = eval_prompts(prompts, model)
        data.append([loss, checkpoint, ngram])
df = pd.DataFrame(data, columns=["Loss", "Checkpoint", "Ngram"])

# %%
data = []
checkpoints = [0, 5, 10, 15] + list(range(20, 150, 10))
for checkpoint in tqdm(checkpoints):
    model = get_model(checkpoint)
    for ngram in random_ngrams:
        prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, 100, 20)
        loss = eval_prompts(prompts, model)
        with model.hooks(deactivate_neurons_fwd_hooks):
            ablated_loss = eval_prompts(prompts, model)
        data.append([loss, ablated_loss, ablated_loss-loss, checkpoint, ngram])
df = pd.DataFrame(data, columns=["OriginalLoss", "AblatedLoss", "LossIncrease", "Checkpoint", "Ngram"])

# %%
df = df.sort_values(by="Checkpoint")
px.line(df, x="Checkpoint", y="OriginalLoss", color="Ngram", title="Loss increase on 20 random German trigrams from top 200 common trigrams", width=900)

# %% [markdown]
# ## Investigating high MCC neurons
# 

# %%
good_neurons

# %%
mmc_neuron_df = df[(df["Neuron"]==neuron[1]) & (df["Layer"]==neuron[0])].sort_values("Checkpoint")
mmc_neuron_df.head()

# %%
# neuron = (5, 395)


# mmc_neuron_df = df[(df["Neuron"]==neuron[1]) & (df["Layer"]==neuron[0])].sort_values("Checkpoint")

# # Create a subplot with 2 y-axes
# fig = make_subplots(specs=[[{"secondary_y": True}]])

# # Add traces for german_loss and english_loss
# #fig.add_trace(go.Scatter(x=df['checkpoint'], y=df['german_loss'], name='German Loss'), secondary_y=False)w

# # Add traces for f1 and mcc
# fig.add_trace(go.Scatter( , x="checkpoint", y="MCC", name='MCC'), secondary_y=True)

# # Set y-axes titles
# fig.update_yaxes(title_text="Loss", secondary_y=False)
# fig.update_yaxes(title_text="Score", secondary_y=True)

# fig.update_layout(title_text=f"L{neuron[0]}N{neuron[1]}")

# fig.show()



# %%
def run_ablation_analysis(neuron_name, data, get_activation_fn):
    ablation_data = []
    checkpoints = range(40)
    with tqdm(total=len(checkpoints)) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            layer, neuron = neuron_name[1:].split("N")
            layer, neuron = int(layer), int(neuron)
            activations = get_activation_fn(df, neuron, layer, checkpoint)
            assert activations is not None
            def tmp_hook(value, hook):
                value[:, :, neuron] = activations
                return value
            tmp_hooks=[(f'blocks.{layer}.mlp.hook_post', tmp_hook)]
            original_loss = eval_loss(model, data)
            with model.hooks(tmp_hooks):
                ablated_loss = eval_loss(model, data)
            ablation_data.append([neuron_name, checkpoint, original_loss, ablated_loss])
            pbar.update(1)

    ablation_df = pd.DataFrame(ablation_data, columns=["Label", "Checkpoint", "OriginalLoss", "AblatedLoss"])
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    return ablation_df

single_neuron_df_english = run_ablation_analysis("L5N1712", english_data, get_mean_german)
single_neuron_df_german = run_ablation_analysis("L5N1712", german_data, get_mean_english)

# %%
single_neuron_df_english["Language"] = "English"
single_neuron_df_german["Language"] = "German"
single_neuron_df = pd.concat([single_neuron_df_english, single_neuron_df_german])
px.line(single_neuron_df, x="Checkpoint", y="AblationIncrease", color="Language", title="Ablation increase for neuron L5N1712", width=900)

# %%
for checkpoint in [8, 9, 10, 11, 12]:
    model = get_model(checkpoint)
    prompt = german_data[1]
    print_activations(prompt, model, (5, 1712), log=True)
    prompt = english_data[1]
    print_activations(prompt, model, (5, 1712), log=True)

# %%
layer, neuron = 5, 1712
checkpoint = 10
model = get_model(checkpoint)

activations = get_mean_english(df, neuron, layer, checkpoint)
assert activations is not None

def tmp_hook(value, hook):
    value[:, :, neuron] = activations
    return value
tmp_hooks=[(f'blocks.{layer}.mlp.hook_post', tmp_hook)]

loss_differences = []
for prompt in german_data[:200]:
    loss = model(prompt, return_type="loss", loss_per_token=True)
    with model.hooks(tmp_hooks):
        ablated_loss = model(prompt, return_type="loss", loss_per_token=True)
    loss_difference = (ablated_loss - loss).flatten().max().item()
    loss_differences.append(loss_difference)

# %%
max_loss_diff_examples = np.argsort(loss_differences)[::-1][:20]

loss_differences = []
for prompt_index in max_loss_diff_examples[:10]:
    prompt = german_data[prompt_index]
    loss = model(prompt, return_type="loss", loss_per_token=True)
    with model.hooks(tmp_hooks):
        ablated_loss = model(prompt, return_type="loss", loss_per_token=True)
    loss_difference = (ablated_loss - loss).flatten().tolist()
    str_tokens = model.to_str_tokens(prompt)[1:]
    haystack_utils.clean_print_strings_as_html(str_tokens, loss_difference, 0.5)

# %% [markdown]
# ## Ablate whole layers loss

# %%
def zero_ablate_hook(value, hook):
    value[:, :, :] = 0
    return value

loss_data = []
checkpoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + list(range(20, 150, 10))
with tqdm(total=len(checkpoints)*6) as pbar:
    for checkpoint in checkpoints:
        model = get_model(checkpoint)
        for layer in range(6):
            for prompt in german_data[:100]:
                loss = model(prompt, return_type="loss").item()
                with model.hooks([(f'blocks.{layer}.mlp.hook_post', zero_ablate_hook)]):
                    ablated_loss = model(prompt, return_type="loss").item()
                loss_difference = (ablated_loss - loss)
                loss_data.append([checkpoint, layer, loss_difference, loss, ablated_loss])
            pbar.update(1)

layer_df = pd.DataFrame(loss_data, columns=["Checkpoint", "Layer", "LossDifference", "OriginalLoss", "AblatedLoss"])

# %%
px.line(layer_df.groupby(["Checkpoint", "Layer"]).mean().reset_index(), x="Checkpoint", y="LossDifference", color="Layer", title="Loss difference for zero-ablating MLP layers on German data", width=900)

# %% [markdown]
# ## Common tokens per language

# %%
german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]
common_german_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, 200)
common_english_tokens = haystack_utils.get_common_tokens(english_data, model, all_ignore, 200)

# %%
model = get_model(NUM_CHECKPOINTS)
torch.cosine_similarity(model.W_in[3, :, 669], model.W_out[3, 669], dim=0)

# %% [markdown]
# ## English losses

# %%
def run_ablation_analysis():
    ablation_data = []
    checkpoints = list(range(0, NUM_CHECKPOINTS, 10))
    print(checkpoints)
    with tqdm(total=len(checkpoints)*len(good_neurons)) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for neuron_name in good_neurons:
                layer, neuron = neuron_name[1:].split("N")
                layer, neuron = int(layer), int(neuron)
                activations = get_mean_german(df, neuron, layer, checkpoint)
                assert activations is not None
                def tmp_hook(value, hook):
                    value[:, :, neuron] = activations
                    return value
                tmp_hooks=[(f'blocks.{layer}.mlp.hook_post', tmp_hook)]
                original_loss = eval_loss(model, english_data)
                with model.hooks(tmp_hooks):
                    ablated_loss = eval_loss(model, english_data)
                ablation_data.append([neuron_name, checkpoint, original_loss, ablated_loss])
                pbar.update(1)

    ablation_df = pd.DataFrame(ablation_data, columns=["Label", "Checkpoint", "OriginalLoss", "AblatedLoss"])
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    ablation_df.to_csv("data/checkpoint_ablation_data_english.csv")

run_ablation_analysis()

# %%
ablation_english_df = pd.read_csv("data/checkpoint_ablation_data_english.csv")

# %%
px.line(ablation_english_df, x="Checkpoint", y="AblationIncrease", color="Label", title="Ablation Increase on English prompts", width=800)

# %%



