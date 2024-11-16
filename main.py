import argparse
from ngrams import *

def feature_extractor(filepath):
    """
    Takes in the file and parses the sentences using <space> as a delimiter.
    Returns a tokenized version of the inputs with the <START> and <STOP> tokens appended.
    """
    features = []
    with open(filepath, "r") as f:
        for line in f:
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
    args = parser.parse_args()

    print(f"\n{args}\n")

    # Convert text into features using fixed A2-Data path
    train = feature_extractor("A2-Data/1b_benchmark.train.tokens")
    validate = feature_extractor(f"A2-Data/1b_benchmark.{args.set}.tokens")

    # Instantiate the model based on the selected argument
    if args.model == "unigram":
        model = Unigram()
    elif args.model == "bigram":
        model = Bigram()
    elif args.model == "trigram":
        model = Trigram()
    elif args.model == 'interpolate':
        model = InterpolatedNGram()
    else:
        raise Exception("Pass unigram, bigram, trigram, or interpolate to --model")

    print(f"Evaluating the {args.set} dataset using the {args.model} model\n")

    if args.model != 'interpolate':
        vocab_size = model.read_ngram(train)
        print(f"There are {vocab_size} unique tokens in the {args.model} model (excluding \"<START>\")")
        vocab = model.get_vocab()
        print(f"There are {vocab} tokens in the vocabulary")

        perplexity = model.model_perplexity(validate)
        print(f"Perplexity of {args.model} model on the {args.set} data is {perplexity}")

        test = "<START> HDTV . <STOP>"
        perplexity = model.model_perplexity([test.split(" ")])
        print(f"Perplexity of {args.model} model for the string \"{test}\" is {perplexity}")

    else:
        model.train(train)
        test = ["<START> HDTV . <STOP>".split(" ")]
        perplexity = model.interpolate(0.1, 0.3, 0.6, test)
        print(f"Perplexity of {args.model} model for the string \"{test}\" is {perplexity}")
        perplexity = model.interpolate(0.1, 0.3, 0.6, validate)
        print(f"Perplexity of {args.model} model for the {args.set} dataset is {perplexity}")

if __name__ == '__main__':
    main()
