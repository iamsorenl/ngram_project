import numpy as np

class Ngram:
    def __init__(self, n):
        self.n = n  # Specifies the n-gram order (1 for unigram, 2 for bigram, etc.)
        self.unigram_counts = {}  # Dictionary to store counts of unigrams
        self.bigram_counts = {}  # Dictionary to store counts of bigrams
        self.trigram_counts = {}  # Dictionary to store counts of trigrams
        self.word_frequencies_unk = {}  # Dictionary to store word frequencies (including <UNK> handling)
        self.word_frequencies = {}  # Dictionary to store word frequencies

    def tokenize_and_count(self, data):
        """Tokenizes sentences, adds <START> and <STOP>, and counts word frequencies in one pass."""
        preprocessed_sentences = []
        self.word_frequencies = {}

        with open(data, 'r') as file:
            lines = file.readlines()
            for line in lines:
                tokens = line.strip().split()
                # Add <START> and <STOP> tokens
                tokens = ["<START>"] + tokens + ["<STOP>"]
                preprocessed_sentences.append(tokens)

                # Count word frequencies (excluding <START>)
                for token in tokens:
                    if token != "<START>":
                        self.word_frequencies[token] = self.word_frequencies.get(token, 0) + 1

        return preprocessed_sentences

    def replace_with_unk(self, preprocessed_sentences):
        """Replaces infrequent words with <UNK> based on the counted word frequencies and updates word_frequencies_unk."""
        #self.word_frequencies_unk = {}  # Reset the dictionary for <UNK> frequencies

        for i, sentence in enumerate(preprocessed_sentences):
            # Replace infrequent words with <UNK> in the sentence (<START> doesnt have a frequency)
            preprocessed_sentences[i] = [
                token if token == "<START>" or self.word_frequencies.get(token, 0) >= 3 else "<UNK>"
                for token in sentence
            ]

            # Update frequencies for <UNK> replaced sentences
            for token in preprocessed_sentences[i]:
                if token != "<START>":  # Exclude <START> tokens from counting
                    self.word_frequencies_unk[token] = self.word_frequencies_unk.get(token, 0) + 1

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
        """Calculates the probability of a given n-gram with handling for the special trigram case."""
        if len(ngram) == 1:  # Unigram
            count_ngram = self.unigram_counts.get(ngram, 0)
            total_unigrams = sum(self.unigram_counts.values())
            return count_ngram / total_unigrams if total_unigrams > 0 else 0
        elif len(ngram) == 2:  # Bigram
            count_ngram = self.bigram_counts.get(ngram, 0)
            count_preceding_word = self.unigram_counts.get((ngram[0],), 0)
            return count_ngram / count_preceding_word if count_preceding_word > 0 else 0
        elif len(ngram) == 3:  # Trigram
            # Special case: Use bigram probability for the first word following <START>
            if ngram[0] == "<START>":
                count_bigram = self.bigram_counts.get((ngram[0], ngram[1]), 0)
                count_start = self.unigram_counts.get((ngram[0],), 0)
                return count_bigram / count_start if count_start > 0 else 0

            # Regular trigram case
            count_ngram = self.trigram_counts.get(ngram, 0)
            preceding_bigram = (ngram[0], ngram[1])
            count_preceding_bigram = self.bigram_counts.get(preceding_bigram, 0)
            return count_ngram / count_preceding_bigram if count_preceding_bigram > 0 else 0
        return 0  # Unknown n-gram probability

    
    def calculate_perplexity(self, sentences):
        """Calculates the perplexity of the n-gram model for a list of tokenized sentences."""
        N = sum(len(sentence) - (self.n - 1) for sentence in sentences)  # Number of words, adjusted for n-grams
        log_sum = 0

        for sentence in sentences:
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



