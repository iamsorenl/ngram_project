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

    # Tokenize and prepare the training data
    training_data_file = "A2-Data/1b_benchmark.train.tokens"
    train_sentences = ngram_model.tokenize_and_count(training_data_file)
    print("train_sentence before:", train_sentences[0])
    train_sentences = ngram_model.replace_with_unk(train_sentences)
    print("train_sentence after:", train_sentences[0])

    # frequencies
    print("word frequency length:", len(ngram_model.word_frequencies))
    print("word frequency length with <UNK>:", len(ngram_model.word_frequencies_unk))

     # set up example file
    example_file = "hdtv"

if __name__ == "__main__":
    main()
