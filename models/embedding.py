#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gensim
import sklearn
import tensorflow as tf
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow_hub as hub
from elmoformanylangs import Embedder
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Word2Vec_Model():
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin', binary=True)

    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                cos_similarity = self.model.similarity(word1, word2)
            except KeyError:
                cos_similarity = 0
            return cos_similarity


class Fasttext_Model():
    def __init__(self):
        self.model = FastText.load_fasttext_format('cc.en.300.bin')

    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                cos_similarity = self.model.similarity(word1, word2)
            except KeyError:
                cos_similarity = 0
            return cos_similarity


class GloVe_Model():
    def __init__(self):
        glove_file = 'glove.840B.300d.txt'
        tmp_file = "glove.840B.300d.tmp"
        glove2word2vec(glove_file, tmp_file)
        self.model = KeyedVectors.load_word2vec_format(tmp_file)

    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                cos_similarity = self.model.similarity(word1, word2)
            except KeyError:
                cos_similarity = 0
            return cos_similarity

class ELMo_Model():
    # follow the below repo to download the data
    # https://github.com/HIT-SCIR/ELMoForManyLangs
    def __init__(self):
        self.model = Embedder('elmo_model/')

    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                vec_word1 = self.model.sents2elmo([[word1]])[0]
                vec_word2 = self.model.sents2elmo([[word2]])[0]
                cos_similarity = cosine_similarity(vec_word1,vec_word2)[0][0]
            except KeyError:
                cos_similarity = 0
            return cos_similarity

class ELMo_Model_Old():
    # old version, too slow
    def __init__(self):
        self.model = hub.Module(
            "https://tfhub.dev/google/elmo/2", trainable=False)

    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                init = tf.initialize_all_variables()
                sess = tf.Session()
                sess.run(init)

                embedding = self.model(
                    [word1, word2], signature="default", as_dict=True)["elmo"]
                normalize_a = tf.nn.l2_normalize(embedding[0][0], 0)
                normalize_b = tf.nn.l2_normalize(embedding[1][0], 0)
                results = tf.reduce_sum(
                    tf.multiply(normalize_a, normalize_b))
                cos_similarity = sess.run(results)
            except KeyError:
                cos_similarity = 0
            return cos_similarity


class BERT_Model():
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-model-path')
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-model-path/bert-base-uncased-vocab.txt')

    def similarity(self, word1: str, word2: str):
        if self.model is None:
            self.model = BertModel.from_pretrained('bert-base-model-path')
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-model-path/bert-base-uncased-vocab.txt')
        if word1 is None or word2 is None:
            return None
        else:
            try:
                word1_index = self.tokenizer.convert_tokens_to_ids(word1)
                word1_index = torch.tensor([word1_index])
                word2_index = self.tokenizer.convert_tokens_to_ids(word2)
                word2_index = torch.tensor([word2_index])
                with torch.no_grad():
                    embedding1 = self.model.embeddings(word1_index)
                    embedding2 = self.model.embeddings(word2_index)
                normalize_a = F.normalize(embedding1, dim=0, p=2)
                normalize_b = F.normalize(embedding2, dim=0, p=2)
                cos_similarity = F.cosine_similarity(normalize_a.sum(
                    1), normalize_b.sum(1), 1).detach().numpy()[0]
                # with torch.no_grad():
                #     encoded_layers1, pooled_output1 = self.model(word1_index)
                #     encoded_layers2, pooled_output2 = self.model(word2_index)
                # cos_similarity = F.cosine_similarity(
                #     pooled_output1, pooled_output2).detach().numpy()[0]
            except KeyError:
                cos_similarity = 0
            return cos_similarity
