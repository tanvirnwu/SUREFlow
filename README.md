# RoboVLA

RoboVLA supports training on LIBERO suites and evaluating a trained checkpoint on either:
- the same vanilla LIBERO suite, or
- a LIBERO-PRO variant of that suite.

This is done by decoupling:
- **train suite**: dataset + training language embeddings
- **eval suite**: simulator benchmark + evaluation language embeddings

## Train suites
Supported `--train_suite` values:
- `libero_object`
- `libero_spatial`
- `libero_goal`
- `libero_90`
- `libero_10`

## Eval suites (LIBERO-PRO suffixes)
Optional `--eval_suite` values:
- `swap`
- `object`
- `lan`
- `task`
- `temp`

When `--eval_suite` is set, simulator benchmark becomes:
`<train_suite>_<eval_suite>`

Examples:
- `train_suite=libero_goal`, `eval_suite=object` -> sim benchmark `libero_goal_object`

## Training
Train on a vanilla LIBERO suite:

```bash
python run.py --train_suite libero_goal
```

## Evaluation with a checkpoint
### 1) Vanilla LIBERO evaluation
Evaluate a checkpoint on the same vanilla suite:

```bash
python run.py --train_suite libero_goal --checkpoint_path /path/to/ckpt.pth
```

### 2) LIBERO-PRO evaluation on the same checkpoint
Evaluate the same checkpoint on a LIBERO-PRO suite:

```bash
python run.py --train_suite libero_goal --eval_suite object --checkpoint_path /path/to/ckpt.pth
```

In this mode:
- dataset benchmark remains `libero_goal`
- simulator benchmark is `libero_goal_object`
- evaluation embeddings are loaded from `language_embeddings/libero_goal_object.pkl`

## Embedding loading behavior
During evaluation, simulation loads embeddings from:

```text
language_embeddings/{sim_benchmark_type}.pkl
```

Error behavior:
- Missing embedding file -> `FileNotFoundError`
- Missing task key in loaded embedding dict -> `KeyError` with example keys

## Model expense profiling (`expense.py`)
Use `expense.py` to sweep model feature-flag combinations and report:
- total parameters
- trainable parameters
- FLOPs in GFLOPs (and MACs when available)

The script evaluates all valid combinations of:
- `use_sigma_film` in `{True, False}`
- `use_action_decoder` in `{True, False}`
- `decoder_use_action_tokens_as_queries` in `{True, False}`
- `use_uncertainty` in `{True, False}`

Constraint enforced by the script:
- if `use_action_decoder=False`, then `decoder_use_action_tokens_as_queries` is forced to `False`

### Usage

Basic run:

```bash
python expense.py --suite libero_object
```

Run with checkpoint, explicit device, batch size, and sequence length:

```bash
python expense.py \
  --suite libero_object \
  --checkpoint /path/to/checkpoint.pt \
  --device cpu \
  --batch_size 1 \
  --seq_len 1
```

### Arguments
- `--suite` (required): one of `libero_object`, `libero_spatial`, `libero_goal`, `libero_90`, `libero_10`
- `--checkpoint` (optional): checkpoint file path; script attempts non-strict loading and continues if it cannot be loaded
- `--device` (optional, default `cpu`): torch device string
- `--batch_size` (optional, default `1`): dummy batch size for FLOPs measurement
- `--seq_len` (optional): observation sequence length; defaults to config `perception_seq_len`

### Output format
Each valid combination prints one line like:

```text
use_sigma_film=T use_action_decoder=F decoder_use_action_tokens_as_queries=F use_uncertainty=T | total_params=123 trainable_params=123 | flops_g=1.234567
```

At the end, a summary is printed with:
- number of combinations evaluated
- FLOPs backend used (`fvcore` or `thop`)
- number of skipped invalid combinations


## Standalone evaluation + paper figures (`visuals.py`)
Use `visuals.py` to evaluate a trained checkpoint on LIBERO or LIBERO-PRO and automatically generate figures + raw metrics.

### What it does
- loads a checkpoint + model scaler
- runs single-process rollouts using the existing simulator/model stack
- supports runtime overrides (without editing config files):
  - URFlow on/off (`use_uncertainty`)
  - horizon (`max_step_per_episode`)
  - rollouts, tau list, refinement steps, seed
- writes PNG plots + JSON/CSV outputs for reproducibility
- accepts optional `--scaler_path` when `model_scaler.pkl` is not colocated with checkpoint files

### CLI
```bash
python visuals.py \
  --checkpoint_path /path/to/checkpoint_or_dir \
  --train_suite libero_goal \
  --output_dir ./logs/visuals_goal
```

### LIBERO-PRO evaluation examples
`--eval_suite` can be either a suffix (`temp`) or full benchmark name (`libero_goal_temp`).

```bash
python visuals.py \
  --checkpoint_path /path/to/checkpoint_or_dir \
  --train_suite libero_goal \
  --eval_suite temp \
  --output_dir ./logs/visuals_goal_temp

python visuals.py \
  --checkpoint_path /path/to/checkpoint_or_dir \
  --train_suite libero_goal \
  --eval_suite libero_goal_temp \
  --output_dir ./logs/visuals_goal_temp
```

### Checkpoint + scaler note
`visuals.py` tries scaler loading in this order:
1. `--scaler_path` (if provided)
2. `<checkpoint_dir>/model_scaler.pkl`
3. `<checkpoint_dir>/../model_scaler.pkl`
4. fallback: build scaler from dataset statistics via trainer config

If your checkpoint is a `.pth` file and scaler is stored elsewhere, pass it explicitly:

```bash
python visuals.py \
  --checkpoint_path /path/to/checkpoints/final_model.pth \
  --scaler_path /path/to/model_scaler.pkl \
  --train_suite libero_spatial \
  --output_dir ./logs/visuals/libero_spatial
```

### Useful overrides
```bash
python visuals.py \
  --checkpoint_path /path/to/checkpoint_or_dir \
  --train_suite libero_goal \
  --output_dir ./logs/visuals_custom \
  --horizons 50 100 200 400 600 \
  --rollouts 10 \
  --tau_list 0.6 0.8 0.9 \
  --refinement_steps 3 \
  --seed 0
```

### Outputs
By default, outputs are saved inside `--output_dir`:
- `evaluation_results.json` (all settings + condition results + per-step sequences)
- `episode_metrics.csv` (per-episode summary rows)
- `figure_A_success_vs_horizon.png`
- `figure_B_success_over_time_H*.png`
- `figure_C_compounding_metric_H*.png`
- `figure_D_refinement_activation_vs_horizon.png`
- `figure_E_long_horizon_heatmap.png`
- `figure_E_long_horizon_heatmap.csv`
- `figure_F_calibration.png` (when enough samples are available)

### Notes
- If `--eval_suite` is omitted, evaluation runs on vanilla LIBERO (`train_suite`).
- If `--eval_suite` is provided, evaluation runs on LIBERO-PRO using `<train_suite>_<suffix>`.
- The script runs in single-process mode (`use_multiprocessing=False`) for stability and reproducibility.
