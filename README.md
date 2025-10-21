Run: python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
Single: python -m src.train --activation elu --optimizer adam --bn on --dropout 0.2 --subset_size 10000 --config config.yaml
Grid: python -m src.experiments --config config.yaml --activations elu relu leakyrelu --optimizers adam sgd --bn on off --dropouts 0.0 0.2 0.5 --subset_sizes 5000 10000
Summary: python -m src.report_utils --glob "results/*/metrics.csv" --out results/summary
