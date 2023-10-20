from pathlib import Path
import json

import torch
from torch import Tensor
from jaxtyping import Float, Int
import plotly.express as px
from einops import einsum
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import einops


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def get_model(model_name: str, checkpoint: int) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        model_name,
        checkpoint_index=checkpoint,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=get_device(),
    )
    return model


def preload_models(model_name: str) -> int:
    """Preload models into cache so we can iterate over them quickly and return the model checkpoint count."""
    i = 0
    try:
        with tqdm(total=None) as pbar:
            while True:
                get_model(model_name, i)
                i += 1
                pbar.update(1)

    except IndexError:
        return i


def load_txt_data(path: str) -> list[str]:
    with open(Path(path), "r") as f:
        return f.read().split("\n")


def load_json_data(path: str | Path) -> list[str]:
    with open(path, "r") as f:
        data = json.load(f)

    min_len = min([len(example) for example in data])
    max_len = max([len(example) for example in data])
    print(
        f"{path}: Loaded {len(data)} examples with {min_len} to {max_len} characters each."
    )
    return data


def load_language_data(path: Path) -> dict[str, list[str]]:
    """
    Returns: dictionary keyed by language code.
    """
    lang_data = {}
    lang_data["en"] = load_json_data(path.joinpath("english_europarl.json"))
    lang_data["de"] = load_json_data(path.joinpath("german_europarl.json"))
    return lang_data


def pos_batch_DLA(
    tokens: Tensor, model: HookedTransformer, pos=-1
) -> tuple[Float[Tensor, "component"], list[str]]:
    """Direct logit attribution for a batch of tokens."""
    answers = tokens[:, pos]
    answer_residual_directions = model.tokens_to_residual_directions(
        answers
    )  # [batch pos d_model]
    _, cache = model.run_with_cache(tokens)
    accumulated_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=pos - 1, return_labels=True
    )
    scaled_residual_stack = cache.apply_ln_to_stack(
        accumulated_residual, layer=-1, pos_slice=pos - 1
    )
    logit_attribution = einsum(
        scaled_residual_stack,
        answer_residual_directions,
        "component batch d_model, batch d_model -> batch component",
    )
    return logit_attribution.mean(0), labels


def get_random_selection(tensor: torch.Tensor, n=12):
    indices = torch.randint(0, len(tensor), (n,))
    return tensor[indices]


def generate_random_prompts(end_string, model, random_tokens, n=50, length=12):
    """Generate a batch of random prompts ending with a specific ngram"""
    end_tokens = model.to_tokens(end_string).flatten()[1:]
    prompts = []
    for i in range(n):
        prompt = get_random_selection(random_tokens, n=length).to(get_device())
        prompt = torch.cat([prompt, end_tokens])
        prompts.append(prompt)
    prompts = torch.stack(prompts)
    return prompts


def get_weird_tokens(
    model: HookedTransformer, w_e_threshold=0.4, w_u_threshold=15, plot_norms=False
) -> tuple[Int[Tensor, "d_vocab"], Int[Tensor, "d_vocab"]]:
    w_u_norm = model.W_U.norm(dim=0)
    w_e_norm = model.W_E.norm(dim=1)
    w_u_ignore = torch.argwhere(w_u_norm > w_u_threshold).flatten()
    w_e_ignore = torch.argwhere(w_e_norm < w_e_threshold).flatten()
    all_ignore = torch.argwhere(
        (w_u_norm > w_u_threshold) | (w_e_norm < w_e_threshold)
    ).flatten()
    not_ignore = torch.argwhere(
        (w_u_norm <= w_u_threshold) & (w_e_norm >= w_e_threshold)
    ).flatten()

    if plot_norms:
        print(f"Number of W_U neurons to ignore: {len(w_u_ignore)}")
        print(f"Number of W_E neurons to ignore: {len(w_e_ignore)}")
        print(f"Number of unique W_U and W_E neurons to ignore: {len(all_ignore)}")

        fig = px.line(
            w_u_norm.cpu().numpy(),
            title="W_U Norm",
            labels={"value": "W_U.norm(dim=0)", "index": "Vocab Index"},
        )
        fig.add_hline(y=w_u_threshold, line_dash="dash", line_color="red")
        fig.show()
        fig = px.line(
            w_e_norm.cpu().numpy(),
            title="W_E Norm",
            labels={"value": "W_E.norm(dim=1)", "index": "Vocab Index"},
        )
        fig.add_hline(y=w_e_threshold, line_dash="dash", line_color="red")
        fig.show()
    return all_ignore, not_ignore


def get_common_tokens(data, model, ignore_tokens, k=100) -> tuple[Tensor, Tensor]:
    """Get top common german tokens excluding punctuation"""
    token_counts = torch.zeros(model.cfg.d_vocab).to(get_device())
    for example in tqdm(data):
        tokens = model.to_tokens(example)
        for token in tokens[0]:
            token_counts[token.item()] += 1

    punctuation = [
        "\n",
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        "-",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "/",
        "\\",
        '"',
        "'",
    ]
    leading_space_punctuation = [" " + char for char in punctuation]
    punctuation_tokens = model.to_tokens(
        punctuation + leading_space_punctuation + [" –", " ", "  ", "<|endoftext|>"]
    )[:, 1].flatten()
    token_counts[punctuation_tokens] = 0
    token_counts[ignore_tokens] = 0
    top_counts, top_tokens = torch.topk(token_counts, k)
    return top_counts, top_tokens


def get_context_effect(
    prompt: str | list[str],
    model: HookedTransformer,
    context_ablation_hooks: list,
    context_activation_hooks: list,
    downstream_components=[],
    pos=None,
):
    return_type = "loss"
    original_metric = model(prompt, return_type=return_type, loss_per_token=True)
    # 1. Activated loss: activate context
    with model.hooks(fwd_hooks=context_activation_hooks):
        activated_metric, activated_cache = model.run_with_cache(
            prompt, return_type=return_type, loss_per_token=True
        )

    # 2. Total effect: deactivate context
    with model.hooks(fwd_hooks=context_ablation_hooks):
        ablated_metric, ablated_cache = model.run_with_cache(
            prompt, return_type=return_type, loss_per_token=True
        )

    # 3. Direct effect: activate context, deactivate later components
    def deactivate_components_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value

    deactivate_components_hooks = [
        (freeze_act_name, deactivate_components_hook)
        for freeze_act_name in downstream_components
    ]
    with model.hooks(fwd_hooks=deactivate_components_hooks + context_activation_hooks):
        direct_effect_metric = model(
            prompt, return_type=return_type, loss_per_token=True
        )

    # 4. Indirect effect: deactivate context, activate later components
    def activate_components_hook(value, hook: HookPoint):
        value = activated_cache[hook.name]
        return value

    activate_components_hooks = [
        (freeze_act_name, activate_components_hook)
        for freeze_act_name in downstream_components
    ]
    with model.hooks(fwd_hooks=activate_components_hooks + context_ablation_hooks):
        indirect_effect_metric = model(
            prompt, return_type=return_type, loss_per_token=True
        )

    if pos is None:
        return (
            original_metric,
            activated_metric,
            ablated_metric,
            direct_effect_metric,
            indirect_effect_metric,
        )
    else:
        return (
            original_metric[:, pos],
            activated_metric[:, pos],
            ablated_metric[:, pos],
            direct_effect_metric[:, pos],
            indirect_effect_metric[:, pos],
        )


def get_mlp_activations(
    prompts: list[str], layer: int, model: HookedTransformer
) -> torch.Tensor:
    def save_activation(value, hook):
        hook.ctx["activation"] = value
        return value

    act_label = f"blocks.{layer}.mlp.hook_post"

    acts = []
    for prompt in prompts:
        with model.hooks([(act_label, save_activation)]):
            model(prompt)
            act = model.hook_dict[act_label].ctx["activation"][:, 10:400, :]
        act = einops.rearrange(act, "batch pos n_neurons -> (batch pos) n_neurons")
        acts.append(act)
    return torch.concat(acts, dim=0)
