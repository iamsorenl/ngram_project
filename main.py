import sys
from ngram import Ngram

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python main.py <ngram_type>")
        sys.exit(1)

    # Retrieve the ngram type from command line arguments
    ngram_type = sys.argv[1]

    # Validate ngram_type and set n-gram order accordingly
    if ngram_type == "unigram":
        n = 1
    elif ngram_type == "bigram":
        n = 2
    elif ngram_type == "trigram":
        n = 3
    else:
        print("Error: Invalid ngram_type. Please choose from 'unigram', 'bigram', or 'trigram'.")
        sys.exit(1)

    # Initialize the Ngram model
    ngram_model = Ngram(n)

    # Path to the training data (adjust as needed)
    train_data = "A2-Data/1b_benchmark.train.tokens"

    # Tokenize and prepare the data
    ngram_model.tokenize_and_prepare(train_data)

    # Preprocess to count word frequencies and handle <UNK>
    ngram_model.preprocess_and_count(ngram_model.preprocessed_sentences)

    # Count n-grams
    ngram_model.count_ngrams()

    # Output results
    print("Number of unique n-grams:", len(ngram_model.ngram_counts))
    print("Total n-gram count:", sum(ngram_model.ngram_counts.values()))

if __name__ == "__main__":
    main()
