# contextual-ngrams

A minimal replication of finding and evaluating contextual n-grams in Pythia series models.

## Instructions

`pip install requirements.txt`

`python feature_formation.py --model pythia-70m`

`python contextual_ngram_formation.py --model pythia-70m --neuron 3,669`

These scripts are extremely slow as they run over hundreds of model checkpoints. We advise using an A6000 with 100GB of RAM or equivalent.
