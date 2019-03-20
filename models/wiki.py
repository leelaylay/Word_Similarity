#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gensim


class LDAModel:
    def __init__(self):
        pass
    
    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                cos_similarity = self.model.similarity(word1, word2)
            except KeyError:
                cos_similarity = 0
            return cos_similarity


class LSAModel:
    def __init__(self):
        pass
    
    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                cos_similarity = self.model.similarity(word1, word2)
            except KeyError:
                cos_similarity = 0
            return cos_similarity