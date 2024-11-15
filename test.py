from ngram import Ngram

def test_ngram_model():
    # Create a simple test corpus
    corpus = ["the cat sat on the mat"]
    with open("test_corpus.txt", "w") as f:
        f.write("\n".join(corpus))

    # Initialize and prepare the Ngram model
    ngram_model = Ngram(n=3)  # Set to 3 for trigrams; can adjust to 1 or 2 for testing unigrams/bigrams
    preprocessed_sentences = ngram_model.tokenize_and_prepare("test_corpus.txt")
    ngram_model.count_ngrams(preprocessed_sentences)

    # Print counts for verification
    print("Unigram Counts:", ngram_model.unigram_counts)
    print("Bigram Counts:", ngram_model.bigram_counts)
    print("Trigram Counts:", ngram_model.trigram_counts)

    # Test probability calculations
    unigram = ("the",)
    bigram = ("the", "cat")
    trigram = ("the", "cat", "sat")

    # Calculate probabilities
    unigram_prob = ngram_model.calculate_probability(unigram)
    bigram_prob = ngram_model.calculate_probability(bigram)
    trigram_prob = ngram_model.calculate_probability(trigram)

    # Print probabilities
    print(f"Probability of unigram {unigram}: {unigram_prob}")
    print(f"Probability of bigram {bigram}: {bigram_prob}")
    print(f"Probability of trigram {trigram}: {trigram_prob}")

    # Assertions to verify expected probabilities
    assert abs(unigram_prob - 0.25) < 1e-6, f"Expected 0.25, but got {unigram_prob}"
    assert abs(bigram_prob - 0.5) < 1e-6, f"Expected 0.5, but got {bigram_prob}"
    assert abs(trigram_prob - 1.0) < 1e-6, f"Expected 1.0, but got {trigram_prob}"

def main():
    test_ngram_model()

if __name__ == "__main__":
    main()
