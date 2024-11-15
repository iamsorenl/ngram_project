import sys

def count_word_frequencies(features):
    # Initialize an empty dictionary to store the word frequencies
    word_frequencies = {}
    
    # Iterate over each sentence in the features
    for sentence in features:
        # Iterate over each word in the sentence
        for word in sentence:
            # Increment the frequency of the word by 1 (or set it to 1 if it doesn't exist)
            # exclude <START> tokens from the word frequencies
            if word != "<START>":
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    # Collect words that have 3 occurrences or less
    words_to_delete = []
    for word, count in list(word_frequencies.items()):
        if count < 3:
            word_frequencies["<UNK>"] = word_frequencies.get("<UNK>", 0) + 1
            words_to_delete.append(word)
    
    # Delete the collected words
    for word in words_to_delete:
        del word_frequencies[word]

    return word_frequencies


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
    
    '''
    # Print the first 10 features to verify
    for i, sentence in enumerate(features[:10], start=1):
        print(i, ": ", sentence, '\n')
    '''

    # Count the word frequencies in the features
    word_frequencies = count_word_frequencies(features)
    print("Number of unique words: ", len(word_frequencies))

if __name__ == "__main__":
    main()
