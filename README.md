
---

### README for N-Gram Language Model Project

---

# N-Gram Language Model Implementation

This project implements N-Gram language models (Unigram, Bigram, Trigram) and an Interpolated N-Gram model using Python. The models calculate the probability of word sequences and estimate perplexity on a given dataset, with optional handling of out-of-vocabulary (OOV) words and smoothing techniques for better generalization. The implementation includes a command-line interface to train and evaluate different models on specified datasets.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Command-Line Interface](#command-line-interface)
- [Models](#models)
- [Evaluation](#evaluation)
- [Smoothing Techniques](#smoothing-techniques)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

N-Gram models are used for language modeling by capturing probabilities of word sequences. This project supports:

- **Unigram, Bigram, and Trigram Models** using Maximum Likelihood Estimation (MLE).
- **Interpolated N-Gram Model** combining multiple N-Gram models through linear interpolation for enhanced predictions.
- Evaluation through perplexity scores on different datasets.

---

## Requirements

- Python >= 3.7

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/ngram_language_model.git
cd ngram_language_model
```

---

## Dataset Structure

Ensure the following files are present in the `A2-Data` directory:

- `1b_benchmark.train.tokens`: Training data file with tokenized sentences.
- `1b_benchmark.dev.tokens`: Development data file.
- `1b_benchmark.test.tokens`: Test data file.

---

## Usage

### Running the Model

Use the following command to run the script:

```bash
python main.py --model <model_type> --set <dataset_type> --oov_threshold <threshold> --l1 <weight1> --l2 <weight2> --l3 <weight3>
```

#### Command-Line Arguments:

- `--model`: Specifies the N-Gram model to use (`unigram`, `bigram`, `trigram`, `interpolate`).
- `--set`: Chooses which dataset to evaluate (`train`, `dev`, `test`).
- `--oov_threshold`: Specifies the OOV threshold for replacing rare words with `<UNK>`.
- `--l1`, `--l2`, `--l3`: Weights for the unigram, bigram, and trigram probabilities in the interpolation model (must sum to 1).

#### Example Usage:

To run a trigram model with an OOV threshold of 3 on the development set:

```bash
python main.py --model trigram --set dev --oov_threshold 3
```

To run an interpolated model with specified weights:

```bash
python main.py --model interpolate --l1 0.2 --l2 0.3 --l3 0.5 --set test
```

---

## Command-Line Interface

Use the `-h` flag to view available options:

```bash
python main.py -h
```

This outputs:

```
usage: main.py [-h] [--model {unigram,bigram,trigram,interpolate}] [--set {train,dev,test}] [--fraction FRACTION] [--oov_threshold OOV_THRESHOLD] [--l1 L1] [--l2 L2] [--l3 L3]

N-Gram Language Model Trainer and Evaluator

options:
  -h, --help            show this help message and exit
  --model {unigram,bigram,trigram,interpolate}
                        Specify the type of n-gram model to use.
  --set {train,dev,test}
                        Choose which dataset to evaluate.
  --fraction FRACTION   Fraction of the dataset to use (default: 1.0).
  --oov_threshold OOV_THRESHOLD
                        Specify the OOV threshold.
  --l1 L1, --l2 L2, --l3 L3
                        Weights for the unigram, bigram, and trigram probabilities.
```

---

## Models

### Unigram Model
- Calculates word probabilities independently.
- Handles out-of-vocabulary words using the `<UNK>` token.

### Bigram Model
- Considers pairs of consecutive words.
- Improves predictions by incorporating context.

### Trigram Model
- Considers sequences of three words for context.
- Provides more accurate predictions than unigram and bigram models.

### Interpolated N-Gram Model
- Combines probabilities from Unigram, Bigram, and Trigram models.
- Uses specified weights for interpolation.

---

## Evaluation

The models are evaluated using **perplexity**, a measure of how well a probability model predicts a test sample. Lower perplexity indicates better generalization and prediction capabilities.

---

## Smoothing Techniques

### Handling OOV Words
- The `oov_threshold` parameter specifies the minimum number of occurrences a word must have to avoid being replaced with `<UNK>`.

### Linear Interpolation
- Combines probabilities from different N-Gram models using specified weights (`l1`, `l2`, `l3`).

---

## Limitations and Future Work

- **Data Sparsity**: Higher-order models like Trigrams may face data sparsity.
- **Hyperparameter Tuning**: Automated tuning methods for interpolation weights could improve results.
- **Expansion**: Incorporation of advanced models like Neural Language Models could enhance performance.

---

### Example Code Walkthrough

The provided code includes:
- **`feature_extractor`**: Processes and tokenizes text, adding `<START>` and `<STOP>` tokens.
- **Model Classes**: Inherits from a base `NGram` class, providing methods for reading and counting n-grams, calculating probabilities, and estimating perplexity.
- **Linear Interpolation**: Combines multiple N-Gram models for robust predictions.

---
