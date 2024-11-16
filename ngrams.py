'''Go to line 49 to adjust the OOV <UNK> threshold. The default is 3.'''
import numpy as np

class NGram(object):
    def read_ngram(self, features):
        """
        Takes the feature vector and extracts unique words and their respective counts.
        Keys in the n-gram dictionary depend on the type of model.
        Returns the number of unique tokens.
        """
        pass

    def calc_loglikelihoods(self, sentence):
        """
        Calculates the sum of the log likelihoods of the sentence.
        """
        pass

    def model_perplexity(self, sequence):
        """
        Calculates the perplexity of the model given the data set and prior mapping.
        """
        llp = 0
        M = 0
        for sentence in sequence:
            llp += self.calc_loglikelihoods(sentence)
            M += len(sentence) - 1
        avg_llp = (- 1. / M) * llp
        return 2. ** avg_llp


class Unigram(NGram):
    def __init__ (self, oov_threshold=3):
        self.unigram = {"<START>": 0, "<STOP>" : 0, "<UNK>" : 0}
        self.num_words = 0
        self.vocab_size = None
        self.oov_threshold = oov_threshold
        
    def read_ngram(self, features):
        """
        Reads the features and counts the occurrences of each word.
        Words with less than 3 occurrences are treated as unknown.
        """
        initial_unigram = {"<START>": 0, "<STOP>" : 0}
        for sentence in features:
            for word in sentence:
                initial_unigram[word] = initial_unigram.get(word, 0) + 1

        for key, val in initial_unigram.items():
            if val < self.oov_threshold:
                self.unigram["<UNK>"] += val
            else:
                self.unigram[key] = val
        
        self.num_words = sum(self.unigram.values()) - self.unigram["<START>"]
        self.vocab_size = len(self.unigram) - 1
        return len(self.unigram) - 1    
    
    def probability(self, word):
        """
        Calculates the probability of a word.
        """
        count = self.unigram.get(word, self.unigram["<UNK>"])
        return count / self.num_words if self.num_words > 0 else 0

    def calc_loglikelihoods(self, sentence):
        """
        Calculates the log likelihood of a sentence.
        """
        llp = 0
        for word in sentence[1:]:
            prob = self.probability(word)
            if prob > 0:
                llp += np.log2(prob)
        return llp

    def get_unigram(self):
        """
        Returns the unigram dictionary.
        """
        return self.unigram

    def get_vocab(self):
        """
        Returns the vocabulary size.
        """
        return self.vocab_size

class Bigram(NGram):
    def __init__ (self, oov_threshold=3):
        self.bigram = {}
        self.unigram = Unigram(oov_threshold=oov_threshold)
        self.num_words = 0
        self.vocab_size = None

    def read_ngram(self, features):
        """
        Reads the features and counts the occurrences of each bigram.
        """
        for sentence in features:
            for i, word in enumerate(sentence[:-1]):
                bi = (word, sentence[i+1])
                self.bigram[bi] = self.bigram.get(bi, 0) + 1

        self.unigram.read_ngram(features)
        self.num_words = sum(self.bigram.values())
        self.vocab_size = self.unigram.get_vocab()
        return len(self.bigram)

    def probability(self, words):
        """
        Calculates the probability of a bigram.
        """
        bigram_count = self.bigram.get(words, 0)
        unigram_count = self.unigram.get_unigram().get(words[0], self.unigram.get_unigram()["<UNK>"])
        return bigram_count / unigram_count if unigram_count > 0 else 0

    def calc_loglikelihoods(self, sentence):
        """
        Calculates the log likelihood of a sentence.
        """
        llp = 0
        for i, word in enumerate(sentence[:-1]):
            bi = (word, sentence[i+1])
            prob = self.probability(bi)
            if prob > 0:
                llp += np.log2(prob)
        return llp

    def get_vocab(self):
        """
        Returns the vocabulary size.
        """
        return self.vocab_size
    
    def get_bigram(self):
        """
        Returns the bigram dictionary.
        """
        return self.bigram
    
    
class Trigram(NGram):
    def __init__ (self, oov_threshold=3):
        self.trigram = {}
        self.bigram = Bigram(oov_threshold=oov_threshold)
        self.num_words = 0
        self.vocab_size = None
    
    def read_ngram(self, features):
        """
        Reads the features and counts the occurrences of each trigram.
        """
        for sentence in features:
            for i, word in enumerate(sentence[:-2]):
                tri = (word, sentence[i+1], sentence[i+2])
                self.trigram[tri] = self.trigram.get(tri, 0) + 1

        self.bigram.read_ngram(features)
        self.num_words = sum(self.trigram.values())
        self.vocab_size = self.bigram.get_vocab()
        return len(self.trigram)

    def probability(self, words):
        """
        Calculates the probability of a trigram.
        """
        trigram_count = self.trigram.get(words, 0)
        bigram_count = self.bigram.get_bigram().get((words[0], words[1]), 0)
        return trigram_count / bigram_count if bigram_count > 0 else 0

    def calc_loglikelihoods(self, sentence):
        """
        Calculates the log likelihood of a sentence.
        """
        bigram = self.bigram.get_bigram()
        first = tuple(sentence[0:2])
        llp = 0
        if first in bigram:
            prob = self.bigram.probability(first)
            if prob > 0:
                llp += np.log2(prob)
        for i, word in enumerate(sentence[:-2]):
            tri = (word, sentence[i+1], sentence[i+2])
            prob = self.probability(tri)
            if prob > 0:
                llp += np.log2(prob)
        return llp

    def get_trigram(self):
        """
        Returns the trigram dictionary.
        """
        return self.trigram
    
    def get_vocab(self):
        """
        Returns the vocabulary size.
        """
        return self.vocab_size

class InterpolatedNGram():
    def __init__ (self, oov_threshold=3):
        self.unigram = Unigram(oov_threshold=oov_threshold)
        self.bigram = Bigram(oov_threshold=oov_threshold)
        self.trigram = Trigram(oov_threshold=oov_threshold)

    def train(self, features):
        """
        Trains the unigram, bigram, and trigram models with the given features.
        """
        self.unigram.read_ngram(features)
        self.bigram.read_ngram(features)
        self.trigram.read_ngram(features)

    def calc_loglikelihoods(self, sentence, lams):
        """
        Calculates the log likelihood of a sentence using interpolation.
        """
        l1, l2, l3 = lams
        bigram = self.bigram.get_bigram()
        unigram = self.unigram.get_unigram()
        trigram = self.trigram.get_trigram()
        first_two = tuple(sentence[0:2])

        llp = 0
        if first_two in bigram:
            pbi = self.bigram.probability(first_two)
            pun = self.unigram.probability(sentence[1])
            ptri = pbi
            llp += np.log2(l1 * pun + l2 * pbi + l3 * ptri)
        
        for i, word in enumerate(sentence[:-2]):
            tri = (word, sentence[i+1], sentence[i+2])
            bi = (sentence[i+1], sentence[i+2])
            un = sentence[i+2]
            ptri = self.trigram.probability(tri)
            pbi = self.bigram.probability(bi)
            pun = self.unigram.probability(un)
            if ptri > 0 or pbi > 0 or pun > 0:
                llp += np.log2(l1 * pun + l2 * pbi + l3 * ptri)
        return llp

    def interpolate(self, lam1, lam2, lam3, predict):
        """
        Interpolates the probabilities using the given lambda values and calculates the perplexity.
        """
        assert lam1 + lam2 + lam3 == 1
        llp = 0
        M = 0
        for sentence in predict:
            llp += self.calc_loglikelihoods(sentence=sentence, lams=(lam1,lam2,lam3))
            M += len(sentence) - 1
        avg_llp = (- 1. / M) * llp
        return 2. ** avg_llp
