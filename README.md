# contextual-ngrams

A minimal replication of finding and evaluating contextual n-grams in Pythia series models.

## Instructions

`pip install nltk kaleido tqdm einops seaborn plotly-express fancy-einsum scikit-learn torchmetrics ipykernel ipywidgets nbformat git+https://github.com/neelnanda-io/TransformerLens git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python git+https://github.com/neelnanda-io/neelutils.git git+https://github.com/neelnanda-io/neel-plotly.git`

Generate data:

`python feature_formation.py --model pythia-70m`

`python contextual_ngram_formation.py --model pythia-70m`

The scripts are extremely slow as they run over hundreds of model checkpoints. We advise using an A6000 with 100GB of RAM or equivalent.

Generate visualizations:

`python feature_formation_viz.py --model pythia-70m`

`python contextual_ngram_formation_viz.py --model pythia-70m`
