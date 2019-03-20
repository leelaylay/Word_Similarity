#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import logging
import multiprocessing
import os

import nltk
import scipy
from nltk.corpus import wordnet, wordnet_ic
from scipy.stats import spearmanr

from models.embedding import (BERT_Model, ELMo_Model, Fasttext_Model,
                              GloVe_Model, Word2Vec_Model)
from models.googlesearch import GoogleSearch
from models.wiki import LDAModel, LSAModel

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="word_similarity_computing.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
logger = logging.getLogger("word_similarity")

dataset_dict = {
    "MEN": "EN-MEN-TR-3k.txt",
    "MTurk-771": "EN-MTurk-771.txt",
    "RW-STANFORD": "EN-RW-STANFORD.txt",
    "SimLex-999": "EN-SIMLEX-999.txt",
    "SimVerb-3500": "EN-SimVerb-3500.txt"
}
def get_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--method_id', type=int, default=3, help='method id')
    return parser.parse_args()


def loop_wordnet(method,sysnet1,sysnet2,ic):
    score_list = []
    for item1 in sysnet1:
        for item2 in sysnet2:
            if item1.name().split(".")[1] == item2.name().split(".")[1]:
                try:
                    score = getattr(wordnet, method)(item1, item2,ic)
                    score_list.append(score)
                except:
                    continue

    return score_list
def wordnet_based_methods(dataset):
    print("WordNet-based methods start")
    logger.info("WordNet-based methods start")
    methods = ["path_similarity", "wup_similarity", "lch_similarity",
               "res_similarity", "jcn_similarity", "lin_similarity"]

    results = {}
    for method in methods:
        print("Method {} on dataset {}".format(method, dataset))
        filename = dataset_dict[dataset]
        simScore, predScore = [], []
        unk, total_size = 0, 0
        for line in open(os.path.join("data", filename)):
            line = line.strip().lower()
            word1, word2, val = line.split()
            simScore.append(float(val))
            sysnet1 = wordnet.synsets(word1)
            sysnet2 = wordnet.synsets(word2)
            # use brown corpus as information content
            brown_ic = wordnet_ic.ic('ic-brown.dat')
            if method ==  "path_similarity" or method ==  "wup_similarity":
                score_list = [getattr(wordnet, method)(item1, item2) for item1 in sysnet1 for item2 in sysnet2]
            elif method ==  "lch_similarity":
                score_list = [getattr(wordnet, method)(item1, item2) for item1 in sysnet1 for item2 in sysnet2 if item1.name().split(".")[1] == item2.name().split(".")[1]]
            else:
                score_list = loop_wordnet(method,sysnet1,sysnet2,brown_ic)
            # remove None object
            score_list = [item for item in score_list if item is not None]
            score = max(score_list) if len(score_list) > 0 else None
            if score is None:
                predScore.append(0)
                unk += 1
            else:
                predScore.append(score)

            total_size += 1
        logger.info("Method {} on dataset {}".format(method, dataset))
        logger.info("Total size: {}, Not Found {}".format(total_size, unk))
        correlation = spearmanr(simScore, predScore)
        logger.info("Spearman correlation: {}".format(correlation))
        results[method] = correlation

    print("WordNet-based methods end")
    logger.info("WordNet-based methods end")
    return dataset,results


def wiki_based_methods(dataset):
    print("Wiki-based methods start")
    logger.info("Wiki-based methods start")

    methods = ["LDAModel","LSAModel"]
    results = {}
    predScore = {}
    for method in methods:
        predScore[method] = []

    filename = dataset_dict[dataset]
    simScore = []
    for line in open(os.path.join("data", filename)):
        line = line.strip().lower()
        word1, word2, val = line.split()
        simScore.append(float(val))
        model = GoogleSearch(word1,word2)
        for method in methods:
            score = getattr(model,method)
            print(score)
            predScore[method].append(score)

    for method in methods:
        correlation = spearmanr(simScore, predScore[method])
        logger.info("Spearman correlation: {}".format(correlation))
        results[method] = correlation

    print("Wiki-based methods end")
    logger.info("Wiki-based methods end")
    print(results)
    return dataset, results


def googlesearch_based_methods(dataset):
    print("GoogleSearch-based methods start")
    logger.info("GoogleSearch-based methods start")
    methods = ["WebJaccard","WebOverlap", "WebDice", "WebPMI",
               "NGD"]
    results = {}
    predScore = {}
    for method in methods:
        predScore[method] = []

    filename = dataset_dict[dataset]
    simScore = []
    for line in open(os.path.join("data", filename)):
        line = line.strip().lower()
        word1, word2, val = line.split()
        simScore.append(float(val))
        model = GoogleSearch(word1,word2)
        for method in methods:
            score = getattr(model,method)
            print(score)
            predScore[method].append(score)

    for method in methods:
        correlation = spearmanr(simScore, predScore[method])
        logger.info("Spearman correlation: {}".format(correlation))
        results[method] = correlation

    print("GoogleSearch-based methods end")
    logger.info("GoogleSearch-based methods end")
    print(results)
    return dataset, results


def representation_learning_methods(dataset):
    print("Embedding methods start")
    logger.info("Embedding methods start")
    
    methods = ["Word2Vec_Model", "Fasttext_Model", "GloVe_Model", "ELMo_Model", "BERT_Model"]
    results = {}
    for method in methods:
        print("Method {} on dataset {}".format(method, dataset))
        model = eval(method)()
        filename = dataset_dict[dataset]
        simScore, predScore = [], []
        unk, total_size = 0, 0
        for line in open(os.path.join("data", filename)):
            line = line.strip().lower()
            word1, word2, val = line.split()
            simScore.append(float(val))
            score = model.similarity(word1,word2)
            if score is None:
                predScore.append(0)
                unk += 1
            else:
                predScore.append(score)

            total_size += 1
        logger.info("Method {} on dataset {}".format(method, dataset))
        logger.info("Total size: {}, Not Found {}".format(total_size, unk))
        correlation = spearmanr(simScore, predScore)
        logger.info("Spearman correlation: {}".format(correlation))
        results[method] = correlation

    print("Embedding methods end")
    logger.info("Embedding methods end")
    return dataset,results


def main():
    args = get_args() 
    pool = multiprocessing.Pool(processes=len(dataset_dict))
    if args.method_id == 0:
        results_list = pool.map(wordnet_based_methods, dataset_dict)   
    elif args.method_id == 1:
        results_list = pool.map(representation_learning_methods, dataset_dict)
    elif args.method_id == 2:
        results_list = pool.map(googlesearch_based_methods, dataset_dict)
    elif args.method_id == 3:
        results_list = pool.map(wiki_based_methods, dataset_dict)
    else: 
        raise NotImplementedError("The method not implement yet")

    print(results_list)





if __name__ == "__main__":
    main()
