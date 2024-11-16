### README for N-Gram Language Model Project

---

# N-Gram Language Model Implementation

This project implements N-Gram language models (Unigram, Bigram, Trigram) and an Interpolated N-Gram model using Python. The models can calculate the probability of word sequences and estimate perplexity on a given dataset, with optional smoothing techniques to handle unseen data and improve generalization. The implementation includes command-line interface options to train and evaluate different models on specified datasets.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Smoothing Techniques](#smoothing-techniques)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

N-Gram models are essential for language modeling, capturing probabilities of word sequences in a text corpus. This project supports:

- **Unigram, Bigram, and Trigram Models** using Maximum Likelihood Estimation (MLE).
- **Interpolated N-Gram Model** that combines multiple N-Gram models using linear interpolation for better predictions.
- Evaluation using perplexity scores on different datasets.

---

## Requirements

To set up the project, ensure you have the following dependencies installed:

- Python >= 3.7

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/iamsorenl/unigram_bigram_trigram_smoothing_analysis.git
cd unigram_bigram_trigram_smoothing_analysis
```

---

## Dataset Structure

Ensure the following files are present in the `A2-Data` directory:

- `1b_benchmark.train.tokens`: Training dataset with tokenized sentences.
- `1b_benchmark.dev.tokens`: Development dataset.
- `1b_benchmark.test.tokens`: Test dataset.

---

## Usage

### Running the Model

The script can be run using the following command:

```bash
python main.py --model <model_type> --smoothing <smoothing_value> --set <dataset_type>
```

#### Command-Line Arguments:

- `--model`: Specifies the N-Gram model to use (`unigram`, `bigram`, `trigram`, `interpolate`).
- `--smoothing`: Specifies the additive smoothing value (float).
- `--set`: Chooses which dataset to evaluate (`train`, `dev`, `test`).

#### Example Usage:

```bash
python main.py --model bigram --smoothing 0.1 --set dev
```

This command trains a bigram model with a smoothing value of 0.1 and evaluates on the development set.

---

## Models

### Unigram Model

- Calculates word probabilities independently.
- Handles out-of-vocabulary (OOV) words using `<UNK>` tokens.

### Bigram Model

- Considers pairs of consecutive words.
- Provides context-based word probabilities.

### Trigram Model

- Considers triplets of words for context.
- Provides more accurate predictions for sequences of three words.

### Interpolated N-Gram Model

- Combines Unigram, Bigram, and Trigram models using linear interpolation.
- Provides robust predictions even for sparse data by leveraging lower-order models.

---

## Evaluation

### Metrics

The models are evaluated using **perplexity**, which measures how well the probability model predicts a sample. Lower perplexity indicates better generalization.

---

## Smoothing Techniques

### Additive Smoothing

- Additive (Laplace) smoothing is used to handle zero-probability issues for unseen n-grams.

### Linear Interpolation

- Combines probabilities from Unigram, Bigram, and Trigram models using specified weights to enhance model robustness.

---

## Limitations and Future Work

- **Data Sparsity**: Higher-order models like the Trigram may suffer from data sparsity.
- **Hyperparameter Tuning**: Automated techniques for tuning interpolation weights can be explored.
- **Expansion**: Consideration for integrating advanced models like Neural Language Models (e.g., RNNs, Transformers).

---

### Example Code Walkthrough

#### Feature Extraction

The `feature_extractor` function processes sentences and appends `<START>` and `<STOP>` tokens to mark sentence boundaries.

#### Model Training and Evaluation

Each N-Gram model inherits from a base `NGram` class and implements methods for:

- Reading and counting n-grams.
- Calculating probabilities.
- Evaluating log-likelihood and perplexity.

#### Linear Interpolation

The `InterpolatedNGram` class combines Unigram, Bigram, and Trigram probabilities using user-specified weights for robust predictions.

---
