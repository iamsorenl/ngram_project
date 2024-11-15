import numpy as np

class Ngram:
    def __init__(self, n):
        self.n = n  # Specifies the n-gram order (1 for unigram, 2 for bigram, etc.)
        self.unigram_counts = {}  # Dictionary to store counts of unigrams
        self.bigram_counts = {}  # Dictionary to store counts of bigrams
        self.trigram_counts = {}  # Dictionary to store counts of trigrams
        self.word_frequencies = {}  # Dictionary to store word frequencies (including <UNK> handling)

    def preprocess_and_count(self, preprocessed_sentences):
        """Counts word frequencies and handles <UNK> replacement."""
        self.word_frequencies = {}
        for sentence in preprocessed_sentences:
            for word in sentence:
                if word != "<START>":  # Exclude <START> tokens from counting
                    self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1

        # Replace words with <UNK> if their frequency is less than 3
        words_to_delete = []
        for word, count in list(self.word_frequencies.items()):
            if count < 3:
                self.word_frequencies["<UNK>"] = self.word_frequencies.get("<UNK>", 0) + 1
                words_to_delete.append(word)

        # Remove the words with less than 3 occurrences
        for word in words_to_delete:
            del self.word_frequencies[word]

    def tokenize_and_prepare(self, data):
        """Reads data, tokenizes it, and adds <START> and <STOP> tokens."""
        preprocessed_sentences = []
        with open(data, 'r') as file:
            lines = file.readlines()
            for line in lines:
                tokens = line.strip().split()  # Tokenize by whitespace
                tokens = ["<START>"] + tokens + ["<STOP>"]  # Add one <START> and one <STOP> token
                preprocessed_sentences.append(tokens)
        return preprocessed_sentences

    def count_ngrams(self, preprocessed_sentences):
        """Counts n-grams in the tokenized sentences."""
        for sentence in preprocessed_sentences:
            for i in range(len(sentence)):
                unigram = tuple(sentence[i:i + 1])
                self.unigram_counts[unigram] = self.unigram_counts.get(unigram, 0) + 1

                if i < len(sentence) - 1:
                    bigram = tuple(sentence[i:i + 2])
                    self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1

                if i < len(sentence) - 2:
                    trigram = tuple(sentence[i:i + 3])
                    self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + 1

    def calculate_probability(self, ngram):
        """Calculates the probability of a given n-gram."""
        if len(ngram) == 1:  # Unigram
            count_ngram = self.unigram_counts.get(ngram, 0)  # Count of the unigram
            total_unigrams = sum(self.unigram_counts.values())  # Total count of unigrams
            if total_unigrams == 0:  # Handle case where there are no unigrams
                return 0
            return count_ngram / total_unigrams
        elif len(ngram) == 2:  # Bigram
            count_ngram = self.bigram_counts.get(ngram, 0)  # Count of the bigram
            preceding_word = (ngram[0],)  # Access preceding word as a tuple
            count_preceding_word = self.unigram_counts.get(preceding_word, 0)  # Access the unigram count
            if count_preceding_word == 0:  # Avoid division by zero
                return 0
            return count_ngram / count_preceding_word
        elif len(ngram) == 3:  # Trigram
            count_ngram = self.trigram_counts.get(ngram, 0) # Count of the trigram
            preceding_bigram = (ngram[0], ngram[1]) # Access preceding bigram as a tuple
            count_preceding_bigram = self.bigram_counts.get(preceding_bigram, 0)
            if count_preceding_bigram == 0: # Avoid division by zero
                return 0
            return count_ngram / count_preceding_bigram
        return 0  # Return zero probability for unknown n-grams
    
    def calculate_perplexity(self, preprocessed_sentences):
        """Calculates the perplexity of the n-gram model for a list of tokenized sentences."""
        N = sum(len(sentence) - (self.n - 1) for sentence in preprocessed_sentences)  # Number of words, adjusted for n-grams
        log_sum = 0

        for sentence in preprocessed_sentences:
            for i in range(self.n - 1, len(sentence)):
                ngram = tuple(sentence[i - self.n + 1:i + 1])
                probability = self.calculate_probability(ngram)

                if probability > 0:
                    log_sum += -np.log2(probability)
                else:
                    # If the probability is zero, it means the n-gram was unseen in training
                    # Return infinite perplexity (or use a small smoothing constant as desired)
                    return float('inf')

        # Calculate perplexity
        perplexity = 2 ** (log_sum / N)
        return perplexity



