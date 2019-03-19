#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gensim
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec 
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch
import torch.nn.functional as F      

import tensorflow as tf
import tensorflow_hub as hub


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
        glove2word2vec(glove_file,tmp_file) 
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
    def __init__(self):
        self.model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    def similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        else:
            try:
                embedding1 = self.model(tf.squeeze(tf.cast(word1, tf.string)), signature="default", as_dict=True)["default"]
                embedding2 = self.model(tf.squeeze(tf.cast(word2, tf.string)), signature="default", as_dict=True)["default"]
                cos_similarity = tf.reduce_sum(tf.multiply(embedding1,embedding2))
                init = tf.initialize_all_variables()
                sess = tf.Session()
                sess.run(init)
            except KeyError:
                cos_similarity = 0
            return cos_similarity

class BERT_Model():
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-model-path')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-model-path/bert-base-uncased-vocab.txt')

    def similarity(self, word1: str, word2: str):
        if self.model is None:
            self.model = BertModel.from_pretrained('bert-base-model-path')
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-model-path/bert-base-uncased-vocab.txt')
        if word1 is None or word2 is None:
            return None
        else:
            try:
                word1_len = len(word1)
                word2_len = len(word2)
                word1_index = self.tokenizer.convert_tokens_to_ids([word1,word2])
                word1_index = torch.tensor([word1_index])
                word2_index = self.tokenizer.convert_tokens_to_ids([word2,word1])
                word2_index = torch.tensor([word2_index])
                # with torch.no_grad():
                #     embedding1 = self.model.embeddings(word1_index)
                #     embedding2 = self.model.embeddings(word2_index)
                # cos_similarity = F.cosine_similarity(embedding1.sum(1), embedding2.sum(1), 1).detach().numpy()[0]
                with torch.no_grad():
                    encoded_layers1, pooled_output1 = self.model(word1_index)
                    encoded_layers2, pooled_output2 = self.model(word2_index)
                result1 = F.cosine_similarity(encoded_layers1[word1_len-1].mean(1), encoded_layers1[-1].mean(1)).detach().numpy()[0]
                result2 = F.cosine_similarity(encoded_layers2[word2_len-1].mean(1), encoded_layers2[-1].mean(1)).detach().numpy()[0]
                cos_similarity = (result1+result2)/2
                print(cos_similarity)
            except KeyError:
                cos_similarity = 0
            return cos_similarity