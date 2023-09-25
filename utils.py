import torch
import plotly.express as px
from einops import einsum
from tqdm.auto import tqdm


def load_txt_data(path: str) -> list[str]:
    with open(Path(path), "r") as f:
        return f.read().split("\n")


def load_json_data(path: str) -> list[str]:
    with open(path, 'r') as f:
        return json.load(f)


def pos_batch_DLA(tokens: Tensor, model: HookedTransformer, pos=-1) -> tuple[Float[Tensor, "component"], list[str]]:
    '''Direct logit attribution for a batch of tokens.'''
    answers = tokens[:, pos]
    answer_residual_directions = model.tokens_to_residual_directions(answers)  # [batch pos d_model]
    _, cache = model.run_with_cache(tokens)
    accumulated_residual, labels = cache.decompose_resid(layer=-1, pos_slice=pos-1, return_labels=True)
    scaled_residual_stack = cache.apply_ln_to_stack(accumulated_residual, layer = -1, pos_slice=pos-1)
    logit_attribution = einsum(scaled_residual_stack, answer_residual_directions, "component batch d_model, batch d_model -> batch component")
    return logit_attribution.mean(0), labels


def generate_random_prompts(end_string, model, random_tokens, n=50, length=12):
    '''Generate a batch of random prompts ending with a specific ngram'''
    end_tokens = model.to_tokens(end_string).flatten()[1:]
    prompts = []
    for i in range(n):
        prompt = get_random_selection(random_tokens, n=length).cuda()
        prompt = torch.cat([prompt, end_tokens])
        prompts.append(prompt)
    prompts = torch.stack(prompts)
    return prompts


def get_weird_tokens(model: HookedTransformer, w_e_threshold=0.4, w_u_threshold=15, plot_norms=False) -> Int[Tensor, "d_vocab"]:
    w_u_norm = model.W_U.norm(dim=0)
    w_e_norm = model.W_E.norm(dim=1)
    w_u_ignore = torch.argwhere(w_u_norm > w_u_threshold).flatten()
    w_e_ignore = torch.argwhere(w_e_norm < w_e_threshold).flatten()
    all_ignore = torch.argwhere((w_u_norm > w_u_threshold) | (w_e_norm < w_e_threshold)).flatten()
    not_ignore = torch.argwhere((w_u_norm <= w_u_threshold) & (w_e_norm >= w_e_threshold)).flatten()

    if plot_norms:
        print(f"Number of W_U neurons to ignore: {len(w_u_ignore)}")
        print(f"Number of W_E neurons to ignore: {len(w_e_ignore)}")
        print(f"Number of unique W_U and W_E neurons to ignore: {len(all_ignore)}")

        fig = px.line(w_u_norm.cpu().numpy(), title="W_U Norm", labels={"value": "W_U.norm(dim=0)", "index": "Vocab Index"})
        fig.add_hline(y=w_u_threshold, line_dash="dash", line_color="red")
        fig.show()
        fig = px.line(w_e_norm.cpu().numpy(), title="W_E Norm", labels={"value": "W_E.norm(dim=1)", "index": "Vocab Index"})
        fig.add_hline(y=w_e_threshold, line_dash="dash", line_color="red")
        fig.show()
    return all_ignore, not_ignore


def get_common_tokens(data, model, ignore_tokens, k=100, return_counts=False, return_unsorted_counts=False) -> Tensor:
    '''Get top common german tokens excluding punctuation'''
    token_counts = torch.zeros(model.cfg.d_vocab).cuda()
    for example in tqdm(data):
        tokens = model.to_tokens(example)
        for token in tokens[0]:
            token_counts[token.item()] += 1

    punctuation = ["\n", ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "\"", "'"]
    leading_space_punctuation = [" " + char for char in punctuation]
    punctuation_tokens = model.to_tokens(punctuation + leading_space_punctuation + [' –', " ", '  ', "<|endoftext|>"])[:, 1].flatten()
    token_counts[punctuation_tokens] = 0
    token_counts[ignore_tokens] = 0
    if return_unsorted_counts:
        return token_counts
    top_counts, top_tokens = torch.topk(token_counts, k)
    if return_counts:
        return top_counts, top_tokens
    return top_tokens