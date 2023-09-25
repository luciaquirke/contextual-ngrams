def produce_images(model_name: str, save_data_path: Path, save_image_path: Path):
    with gzip.open(
        save_path.joinpath("checkpoint_ablation_data.pkl.gz"), "rb", compresslevel=9
    ) as f:
        data = pickle.load(f)
    df = data['ngram']

    df = df.sort_values(by="Checkpoint")
    fig = px.line(df, x="Checkpoint", y="OriginalLoss", color="Ngram", title="Loss increase on 20 random German trigrams from top 200 common trigrams", width=900)
    fig.save_image(save_image_path.joinpath("ngrams_loss_increase.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="EleutherAI/pythia-70m",
        help="Name of model from TransformerLens",
    )
    parser.add_argument("--output_dir", default="output")

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    save_image_path = os.path.join(save_path, "images")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_image_path, exist_ok=True)
    
    produce_images(args.model, Path(save_path), Path(save_image_path))