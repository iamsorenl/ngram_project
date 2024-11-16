import argparse
from ngrams import *

def feature_extractor(filepath, fraction=1.0):
    """
    Takes in the file and parses the sentences using <space> as a delimiter.
    Returns a tokenized version of the inputs with the <START> and <STOP> tokens appended.
    Optionally, only uses a fraction of the data (e.g., 0.5 for half).
    """
    features = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        lines = lines[:int(len(lines) * fraction)]
        for line in lines:
            splitted = line.strip().split(" ")
            words = [word for word in splitted]
            words = ["<START>"] + words + ["<STOP>"]
            features.append(words)
    return features


def main():
    parser = argparse.ArgumentParser(description="N-Gram Language Model Trainer and Evaluator")
    parser.add_argument('--model', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram', 'interpolate'],
                        help='Specify the type of n-gram model to use: unigram, bigram, trigram, or interpolate (default: unigram).')
    parser.add_argument('--set', type=str, default='dev',
                        choices=['train', 'dev', 'test'],
                        help='Choose which dataset to evaluate: train, dev, or test (default: dev).')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of the dataset to use (default: 1.0).')
    parser.add_argument('--oov_threshold', type=int, default=3,
                        help='Specify the OOV threshold (default: 3).')
    parser.add_argument('--l1', type=float, default=0.1, help='Weight for the unigram probability in interpolation (default: 0.1).')
    parser.add_argument('--l2', type=float, default=0.3, help='Weight for the bigram probability in interpolation (default: 0.3).')
    parser.add_argument('--l3', type=float, default=0.6, help='Weight for the trigram probability in interpolation (default: 0.6).')
    args = parser.parse_args()

    # Ensure the lambda values sum to 1
    if not (0 <= args.l1 <= 1 and 0 <= args.l2 <= 1 and 0 <= args.l3 <= 1):
        raise ValueError("Lambda values must be between 0 and 1.")
    if abs(args.l1 + args.l2 + args.l3 - 1) > 1e-5:
        raise ValueError("Lambda values must sum to 1.")

    print(f"\n{args}\n")

    # Convert text into features using fixed A2-Data path
    train = feature_extractor("A2-Data/1b_benchmark.train.tokens", args.fraction)
    validate = feature_extractor(f"A2-Data/1b_benchmark.{args.set}.tokens")

    # Instantiate the model based on the selected argument
    if args.model == "unigram":
        model = Unigram(oov_threshold=args.oov_threshold)
    elif args.model == "bigram":
        model = Bigram(oov_threshold=args.oov_threshold)
    elif args.model == "trigram":
        model = Trigram(oov_threshold=args.oov_threshold)
    elif args.model == 'interpolate':
        model = InterpolatedNGram(oov_threshold=args.oov_threshold)
    else:
        raise Exception("Pass unigram, bigram, trigram, or interpolate to --model")

    print(f"Evaluating the {args.set} dataset using the {args.model} model\n")

    if args.model == 'interpolate':
        model.train(train)
        test = ["<START> HDTV . <STOP>".split(" ")]
        perplexity = model.interpolate(0.1, 0.3, 0.6, test)
        print(f"Perplexity of {args.model} model for the string \"{test}\" is {perplexity}")
        perplexity = model.interpolate(args.l1, args.l2, args.l3, validate)
        print(f"Perplexity of {args.model} model for the {args.set} dataset is {perplexity}")
    else:
        vocab_size = model.read_ngram(train)
        print(f"There are {vocab_size} unique tokens in the {args.model} model (excluding \"<START>\")")
        vocab = model.get_vocab()
        print(f"There are {vocab} tokens in the vocabulary")

        perplexity = model.model_perplexity(validate)
        print(f"Perplexity of {args.model} model on the {args.set} data is {perplexity}")

        test = "<START> HDTV . <STOP>"
        perplexity = model.model_perplexity([test.split(" ")])
        print(f"Perplexity of {args.model} model for the string \"{test}\" is {perplexity}")

if __name__ == '__main__':
    main()
