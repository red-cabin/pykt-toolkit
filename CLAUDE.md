# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

pyKT is a Python library built on PyTorch for training and benchmarking deep learning based knowledge tracing (DLKT) models. It provides 30+ DLKT model implementations, standardized data preprocessing for 7+ educational datasets, and infrastructure for training, evaluation, and experiment tracking via Weights & Biases.

Python >= 3.5 required. Dependencies: `torch>=1.7.0`, `wandb>=0.12.9`, `numpy`, `pandas`, `scikit-learn`, `entmax`.

## Build and Test

```bash
# Build for PyPI distribution
bash build.sh        # runs: python -m build && twine upload dist/*

# Run tests (end-to-end smoke test on assist2015)
cd tests && bash test.sh
# Steps: preprocess data -> train DKT -> predict -> evaluate
```

Tests are minimal — a single integration script that preprocesses, trains, and evaluates a DKT model on the assist2015 dataset.

## Architecture

### Package structure (`pykt/`)

**`pykt/models/`** — The core of the library. Each model is in its own file (e.g., `dkt.py`, `akt.py`). Key infrastructure files:

- [`pykt/models/init_model.py`](pykt/models/init_model.py) — Factory function `init_model()` that instantiates any model by name, mapping model_name strings to classes (e.g., `"dkt"` → `DKT`). Also provides `load_model()` to load from a checkpoint.
- [`pykt/models/train_model.py`](pykt/models/train_model.py) — `train_model()`: main training loop with epoch iteration, loss computation via `cal_loss()`, and forward pass dispatch via `model_forward()`. These two functions contain large if-elif chains keyed on `model.model_name` — every new model must add branches here.
- [`pykt/models/evaluate_model.py`](pykt/models/evaluate_model.py) — `evaluate()`: standard concept-level AUC/ACC evaluation; `evaluate_question()`: question-level evaluation with early/late fusion; `evaluate_splitpred_question()`: split-prediction scenario evaluation. All contain large model-name dispatch chains.
- [`pykt/models/loss.py`](pykt/models/loss.py) — Loss functions.
- PromptKT variants have separate train/init modules: `train_model4promptkt.py`, `init_model4promptkt.py`.

**Model dispatch pattern**: The `train_model.py` and `evaluate_model.py` files use `model.model_name` to branch behavior across models — different forward signatures, different loss calculations, different evaluation logic. Adding a new model requires updating the if-elif chains in `cal_loss`, `model_forward`, `init_model`, and `evaluate`.

**`pykt/datasets/`** — Data loading layer:
- [`data_loader.py`](pykt/datasets/data_loader.py) — Base `KTDataset` PyTorch Dataset. Loads preprocessed CSV sequences, supports train/valid/test folds and question-level evaluation. Preprocessed `.pkl` files are cached after first load.
- [`init_dataset.py`](pykt/datasets/init_dataset.py) — `init_dataset4train()` and `init_test_datasets()` create DataLoaders with the correct Dataset class (specialized DataLoaders exist for models needing extra features: `dkt_forget`, `atdkt`, `lpkt`, `dimkt`, `mockt`, `que_type_models`).
- Specialized dataloaders: `dkt_forget_dataloader.py`, `atdkt_dataloader.py`, `lpkt_dataloader.py`, `que_data_loader.py`, `dimkt_dataloader.py`, `mockt_data_loader.py`, `que_data_loader_promptkt.py`.

**`pykt/preprocess/`** — Data preprocessing scripts, one per dataset (e.g., `assist2009_preprocess.py`, `algebra2005_preprocess.py`). The entry point is `data_proprocess.py` → `process_raw_data()`.

**`pykt/config/`** — `config.py` defines `que_type_models` list (models using question-level dataset format: `["iekt","qdkt","qikt","lpkt","rkt","promptkt","denoisekt"]` and their variants). This is imported by the training/evaluation dispatch logic.

**`pykt/utils/`** — Utility functions including `debug_print`, `set_seed`, and Weights & Biases utilities.

### Configuration files (`configs/`)

- [`configs/data_config.json`](configs/data_config.json) — Dataset metadata: paths, number of questions/concepts, input types, fold definitions, max sequence length, file names for each dataset.
- [`configs/kt_config.json`](configs/kt_config.json) — Training defaults (batch_size, num_epochs, optimizer) and per-model hyperparameter defaults (learning_rate, emb_size, dropout, etc.).
- [`configs/wandb.json`](configs/wandb.json) — Weights & Biases API key configuration.
- [`configs/best_model.json`](configs/best_model.json) — Best model checkpoint references.

### Examples (`examples/`)

The primary workflow uses W&B integration:
- **Training**: `wandb_train.py` is the main training entry point. Per-model wrappers exist (`wandb_dkt_train.py`, `wandb_akt_train.py`, etc.) that pass model-specific params to `wandb_train.py`.
- **Prediction**: `wandb_predict.py` loads a trained checkpoint and runs evaluation.
- **Data preprocessing**: `data_preprocess.py` / `dataprocess.sh`.
- **Result extraction**: `extract_raw_result.py`, `merge_wandb_results.py`, `generate_wandb.py`.

### Training flow (conceptual)

```
data_preprocess.py → raw CSV → preprocessed sequences
        ↓
wandb_train.py → init_dataset4train() → KTDataset → DataLoader
        ↓
init_model(model_name, model_config, data_config, emb_type) → model instance
        ↓
train_model() → iterate epochs → model_forward() → cal_loss() → backprop
        ↓                          (dispatch on model_name)
evaluate() on valid set each epoch → save best checkpoint by AUC
```

## Key conventions

- **Model identification**: Every model class must set `self.model_name` (e.g., `"dkt"`). This string is the key used by `init_model()`, `train_model()`, and `evaluate()` to dispatch behavior.
- **Embedding types**: Models support an `emb_type` parameter (e.g., `"qid"`, `"qid_cl"`, etc.) that controls input representation. Training saves checkpoints as `{emb_type}_model.ckpt`.
- **Data format**: Sequences are stored as CSV files with columns: `uid`, `questions`, `concepts`, `responses`, `timestamps`, `selectmasks`, `fold`, etc. Preprocessed `.pkl` files are cached.
- **Folds**: 5-fold cross-validation is standard. Fold `-1` denotes test data.
- **Device**: Code auto-detects CUDA availability; falls back to CPU.
- **Sequence shifting**: Input features are offset by 1 for next-step prediction — `qseqs`/`cseqs`/`rseqs` are steps 0..n-2, `shft_qseqs`/`shft_cseqs`/`shft_rseqs` are steps 1..n-1.
