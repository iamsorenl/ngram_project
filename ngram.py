class Ngram:
    def __init__(self, n):
        self.n = n  # Specifies the n-gram order (1 for unigram, 2 for bigram, etc.)
        self.ngram_counts = {}  # Dictionary to store counts of n-grams
        self.word_frequencies = {}  # Dictionary to store word frequencies (including <UNK> handling)
        self.preprocessed_sentences = []  # List to store preprocessed sentences

    def preprocess_and_count(self, preprocessed_sentences):
        """Counts word frequencies and handles <UNK> replacement (internal use only)."""
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
        """Reads data, tokenizes it, and adds <START> and <STOP> tokens based on n-gram type (internal use only)."""
        self.preprocessed_sentences = []
        with open(data, 'r') as file:
            lines = file.readlines()
            for line in lines:
                tokens = line.strip().split()  # Tokenize by whitespace

                # Add <START> and <STOP> tokens based on the n-gram order
                if self.n == 3:
                    tokens = ["<START>", "<START>"] + tokens  # Add two <START> tokens for trigrams
                elif self.n == 2:
                    tokens = ["<START>"] + tokens  # Add one <START> token for bigrams
                
                tokens.append("<STOP>")  # Add <STOP> token
                self.preprocessed_sentences.append(tokens)

    def count_ngrams(self):
        """Counts n-grams in the tokenized sentences."""
        for sentence in self.preprocessed_sentences:
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i:i + self.n])
                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
