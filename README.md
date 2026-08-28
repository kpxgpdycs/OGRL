# RMNL Assortment and Dynamic Cache Optimization

**English** | [中文](README_zh.md)

This repository contains an exact RMNL assortment solver, a PPO-based dynamic caching experiment, and a synthetic-data generator.

## Assortment optimization

The assortment solver constructs the sorted critical points of the parametric item-score functions and locates the optimal interval by binary search using the sign of the monotone oracle $D(\lambda)$.

<p align="center">
  <img src="assets/assortment_search.gif" alt="Binary search over RMNL critical points" width="950">
</p>

The animation is generated from the same binary-search logic used by `assortment.py`: the search repeatedly updates `lo`, `mid`, and `hi` according to the sign of $D(\lambda_{mid})$ until the final critical-point bracket is identified.

## Repository structure

```text
.
├── assortment.py
├── cache_rl.py
├── generate_simulation_data.py
├── runtime_params.example.json
├── requirements.txt
├── requirements-rl.txt
├── demo/
│   └── generate_binary_search_gif.py
└── assets/
    └── assortment_search.gif
```

## Installation

For the assortment solver, GIF generation, and simulation-data generator:

```bash
pip install -r requirements.txt
```

For the full PPO caching experiment:

```bash
pip install -r requirements-rl.txt
```

## Test the assortment solver

```bash
python assortment.py
```

The script runs a deterministic example and randomized checks against exhaustive search.

## Regenerate the binary-search animation

```bash
python demo/generate_binary_search_gif.py
```

The generated file is written to:

```text
assets/assortment_search.gif
```

## Generate synthetic data

```bash
python generate_simulation_data.py --output-dir ./data
```

The generator creates:

```text
data/records.pkl
data/user_prefs.npy
data/social_graph.npy
```

Request records use the format `(user_id, time_slot, item_id)`.

## Configure runtime parameters

Copy the public template:

```bash
cp runtime_params.example.json runtime_params.json
```

The JSON file is the primary experiment configuration. It can configure:

- system size and capacities: `U`, `I`, `T`, `K`, `C_cs`;
- preference and social dynamics;
- item-size generation;
- reward and cache-cost parameters;
- PPO hyperparameters and training schedule;
- evaluation settings and random seeds;
- request, preference, and social-graph data paths.



## Run the caching experiment

With paths and experiment settings stored in the JSON file, the basic command is:

```bash
python cache_rl.py --runtime-config ./runtime_params.json
```

Selected command-line options can temporarily override JSON values, for example:

```bash
python cache_rl.py \
  --runtime-config ./runtime_params.json \
  --time-slots 200 \
  --data-path ./data/alternative_records.pkl
```

Use `python cache_rl.py --help` to see the available CLI overrides.


## License

MIT License. See `LICENSE`.
