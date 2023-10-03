import os
from pathlib import Path
import pickle
import gzip
import argparse
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


def get_good_mcc_neurons(probe_df: pd.DataFrame):
    neurons = probe_df[(probe_df["MCC"] > 0.85) & (probe_df["MeanGermanActivation"]>probe_df["MeanNonGermanActivation"])][["NeuronLabel", "MCC"]].copy()
    neurons = neurons.sort_values(by="MCC", ascending=False)
    good_neurons = neurons["NeuronLabel"].unique()

    return good_neurons


def load_probe_data(save_path):
    with gzip.open(save_path.joinpath("checkpoint_probe_df.pkl.gz"), "rb") as f:
        return pickle.load(f)


def load_ablation_analysis():
    ablation_df = pd.read_csv("data/checkpoint_ablation_data.csv")
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    return ablation_df


def produce_images(save_data_path: Path, save_image_path: Path):
    # with gzip.open(
    #     save_data_path.joinpath("checkpoint_ablation_data.pkl.gz"), "rb", compresslevel=9
    # ) as f:
    #     data = pickle.load(f)
    # df = data['ngram']

    # df = df.sort_values(by="Checkpoint")
    # fig = px.line(df, x="Checkpoint", y="OriginalLoss", color="Ngram", title="Loss increase on 20 random German trigrams from top 200 common trigrams", width=900)
    # fig.write_image(save_image_path.joinpath("ngrams_loss_increase.png"))

    probe_df = load_probe_data(save_data_path)
    # good_neurons = get_good_mcc_neurons(probe_df)
    ablation_df = load_ablation_analysis()
    ablation_df.sort_values(by=["Checkpoint", "Label"], inplace=True)

    L3N669_df = ablation_df[ablation_df["Label"] == "L3N669"]
    L3N669_df['Loss Increase'] = L3N669_df['AblationIncrease']
    non_L3N669_df = ablation_df[(ablation_df["Label"] != "L3N669")]
    non_L3N669_df['Loss Increase'] = non_L3N669_df['AblationIncrease']
    non_L3N669_df['Label'] = 'Other neurons with MCC > 0.85'

    percentiles = [0.25, 0.5, 0.75]
    grouped = non_L3N669_df.groupby('Checkpoint')['Loss Increase'].describe(percentiles=percentiles).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color='rgba(31, 119, 180, 0.4)', showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)', line_color='rgba(31, 119, 180, 0.4)', showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color='#1F77B4', width=2), name="Median of Other<br>Context Neurons"))
    fig.add_trace(go.Line(x=L3N669_df['Checkpoint'], y=L3N669_df['Loss Increase'], mode='lines', line=dict(color='#FF7F0E', width=2), name="L3N669"))

    fig.update_layout(title="Loss Increase on German Text when Ablating Context Neurons", xaxis_title="Checkpoint", yaxis_title="Loss Increase", font=dict(size=24))

    fig.write_image(save_image_path.joinpath("ablation_increase_german_text.png"), width=2000)

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

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    save_image_path = os.path.join(save_path, "images")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_image_path, exist_ok=True)
    
    produce_images(Path(save_path), Path(save_image_path))