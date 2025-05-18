# Training a 1B Parameter LLM

This repo contains scripts to train a roughly 1B parameter language model using
public data and to interact with the trained model from the terminal.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Training

Run the training script (this requires significant compute resources):

```bash
python train.py
```

The model and tokenizer will be saved to the `model/` directory.

## Interaction

After training (or after placing a compatible model in `model/`), start an
interactive session:

```bash
python interact.py
```

Type a prompt and press Enter to get completions from the model.
