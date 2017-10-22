"""Vocabulary Management Framework

A set of tools for building vocabs and dictionaries for NLP neural nets.
"""

import os
import spacy
import numpy as np
import logging

class VocabularyManager:

    def __init__(self, include_unk=True, lang="en", custom_vectors_file=None):
        """Set up empty manager"""
        self._embeddings = np.zeros(shape=(1,))
        self._index = {}
        self.include_unk = include_unk
        self.nlp = spacy.load(lang)
        self.custom_vectors = custom_vectors
        self.logger = logging.getLogger(self.__class__.__qualname__)


    def get_vocab_size(self):
        """Calculate and return vocab size"""
        return len(self._embeddings)


    def generate_vocab_from_samples(self, samples):
        """Given a list or iterator over some text, generate a vocabulary"""

        # iterate over documents and identify all unique words
        dictionary = set()
        for doc in self.nlp.pipe(samples):
            dictionary.add([word.text for word in doc])

        # list the word alphabetically for embedding
        sorted_words = sorted(words, key=lambda x: words[x], reverse=True)

        #optionally prepend UNK special word to list
        if self.include_unk:
            sorted_words.insert(0, "UNK")



        #initialise the word embeddings with the new dictionary

    def _init_embeddings(self, dictionary):
        """Once a dictionary of words is identified, initialise embeddings."""

        if self.custom_vectors_file:
            self.logger.info("Using custom vectors instead of spacy")
            from gensim.models.word2vec import Word2Vec

            w2v = Word2Vec.load_word2vec_format(self.custom_vectors_file)

            self.word_dimension = w2v.vector_size

            self.embeddings = np.array([np.zeros(self.word_dimension)] +
                                       [np.array(w2v[x]) if x in w2v
                                       else np.random.rand(self.word_dimension)
                                       for x in sorted_words])

        else:
            self.logger.info("Using spacy vectors as base vocabulary")

            self.word_dimension = self.en_nlp.vocab.vectors_length

            self.embeddings = np.array([np.zeros(self.word_dimension)] +
                                       [self.en_nlp.vocab[x].vector
                                        for x in sorted_words])
