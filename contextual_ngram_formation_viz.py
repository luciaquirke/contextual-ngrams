import os
from pathlib import Path
import pickle
import gzip
import argparse
import plotly.express as px
import pandas as pd


def get_good_mcc_neurons(probe_df: pd.DataFrame):
    neurons = probe_df[(probe_df["MCC"] > 0.85) & (probe_df["MeanGermanActivation"]>probe_df["MeanNonGermanActivation"])][["NeuronLabel", "MCC"]].copy()
    neurons = neurons.sort_values(by="MCC", ascending=False)
    good_neurons = neurons["NeuronLabel"].unique()[:10]

    return good_neurons


def load_probe_data(save_path):
    with gzip.open(save_path.joinpath("checkpoint_probe_df.pkl.gz"), "rb") as f:
        return pickle.load(f)


def load_ablation_analysis():
    ablation_df = pd.read_csv("data/checkpoint_ablation_data.csv")
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    return ablation_df


def produce_images(model_name: str, save_data_path: Path, save_image_path: Path):
    with gzip.open(
        save_data_path.joinpath("checkpoint_ablation_data.pkl.gz"), "rb", compresslevel=9
    ) as f:
        data = pickle.load(f)
    df = data['ngram']

    df = df.sort_values(by="Checkpoint")
    fig = px.line(df, x="Checkpoint", y="OriginalLoss", color="Ngram", title="Loss increase on 20 random German trigrams from top 200 common trigrams", width=900)
    fig.save_image(save_image_path.joinpath("ngrams_loss_increase.png"))

    probe_df = load_probe_data(save_data_path)
    good_neurons = get_good_mcc_neurons(probe_df)
    ablation_df = load_ablation_analysis()
    print(ablation_df.head())
    ablation_df.sort_values(by=["Checkpoint", "Label"], inplace=True)
    fig = px.line(ablation_df[ablation_df["Label"].isin(good_neurons)], x="Checkpoint", y="AblationIncrease", color="Label", title="Ablation Increase on German prompts", width=800)
    fig.write_image(save_image_path.joinpath("top_ctx_neurons_ablation_increase_german_prompts.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="pythia-70m",
        help="Name of model from TransformerLens",
    )
    # parser.add_argument("--data_dir", default="data/europarl")
    parser.add_argument("--output_dir", default="output")

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    save_image_path = os.path.join(save_path, "images")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_image_path, exist_ok=True)
    
    produce_images(args.model, Path(save_path), Path(save_image_path))