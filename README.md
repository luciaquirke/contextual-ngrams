# contextual-ngrams

A minimal replication of finding and evaluating contextual n-grams in Pythia series models.

## Setup

Use the included Dockerfile, or alternatively install PyTorch then run:

`pip install nltk kaleido tqdm einops seaborn plotly-express fancy-einsum scikit-learn torchmetrics ipykernel ipywidgets nbformat git+https://github.com/neelnanda-io/TransformerLens git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python git+https://github.com/neelnanda-io/neelutils.git git+https://github.com/neelnanda-io/neel-plotly.git`

### Instructions

Generate data by running each file from the command line:

`python feature_formation.py --model pythia-70m`

`python contextual_ngram_formation.py --model pythia-70m`

`. . .`

Some scripts are extremely slow because they run over hundreds of model checkpoints. We advise using an A6000 with 100GB of RAM or equivalent.

Then replicate figures by running figures.py

`python figures.py --model pythia-70m`
