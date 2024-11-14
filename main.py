import sys

def extract_features(data, add_extra_start=False):
    features = []
    with open(data, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Tokenize the line (assuming whitespace tokenization)
            tokens = line.strip().split()

            # Add <START> and <STOP> tokens
            if add_extra_start:
                tokens = ["<START>", "<START>"] + tokens  # Add two <START> tokens for trigram
            else:
                tokens = ["<START>"] + tokens  # Add one <START> token for bigram

            tokens.append("<STOP>")  # Add <STOP> token

            # Add the processed tokens to features
            features.append(tokens)

    return features

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python main.py <ngram_type>")
        sys.exit(1)

    # Path to the training data
    ngram_type = sys.argv[1]

    # Validate ngram_type
    if ngram_type not in ["unigram", "bigram", "trigram"]:
        print("Error: Invalid ngram_type. Please choose from 'unigram', 'bigram', or 'trigram'.")
        sys.exit(1)

    # Determine if extra start tokens should be added based on ngram_type
    add_extra_start = ngram_type == "trigram"

    # Path to the training data
    train_data = "A2-Data/1b_benchmark.train.tokens"

    # Extract features for the training data
    features = extract_features(train_data, add_extra_start=add_extra_start)
    
    # Print the first 50 features to verify
    for sentence in features[:50]:
        print(sentence)

if __name__ == "__main__":
    main()
