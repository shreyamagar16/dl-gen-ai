# Messy Mashup Genre Classification

CNN, CRNN, and Transformer experiments on mel spectrograms for noisy mashup-style audio, with augmentation for robustness.

## Setup

Create a Python environment (3.10+ recommended), then install dependencies with `pip install -r requirements.txt`. Point `train.py` (or your notebook) at your messy mashup dataset root—the default layout expects genre folders with stem WAVs (e.g. `drums.wav`). Optionally log runs with [Weights & Biases](https://wandb.ai) via `wandb login`.

## Project Structure

`src/` holds the training script (`train.py`), dataset and feature code (`dataset.py`, `features.py`), and the CNN definition (`models.py`). `notebooks/` contains exploratory and model-specific notebooks (CNN, CRNN, transformer, pipeline). `requirements.txt` lists core libraries (PyTorch, torchaudio, librosa, etc.).

## Models

The shared baseline is a small **CNN** on single-channel mel spectrograms (`CNNModel` in `src/models.py`). **CRNN** and **Transformer** variants are developed in the notebooks, using the same mel feature pipeline where applicable. All heads target the same 10-genre classification task.

## Results

Metrics and loss curves are meant to be tracked in W&B or captured in notebook outputs after training. Report accuracy (and confusion matrices) from your best checkpoint; numbers depend on data path, augmentation, and epoch count—reproduce locally with the notebooks or `src/train.py`.
