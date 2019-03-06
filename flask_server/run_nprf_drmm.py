# -*- coding: utf8 -*-
from __future__ import print_function
from comet_ml import Experiment

import json
import argparse
import sys
import random
import math
random.seed(49999)
import numpy as np
np.random.seed(49999)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow
tensorflow.set_random_seed(49999)

import flask
from flask import request
from flask import jsonify

import keras
import keras.backend as K
from keras.models import Sequential, Model

from matchzoo.utils import *
# from matchzoo.optimizers import *
# from matchzoo.models import *
from matchzoo.inputs.preprocess import *

import sys
sys.path.append("../model/")
from nprf_drmm import NPRFDRMM
from nprf_drmm_config import NPRFDRMMConfig

sys.path.append("../utils/")
from file_operation import df_map_from_file
from nprf_drmm_pair_generator import NPRFDRMMPairGenerator

sys.path.append("../preprocess/")
from matrix import similarity_matrix, hist_from_matrix

from collections import OrderedDict, Counter, deque
from gensim.models.keyedvectors import KeyedVectors

# config = tensorflow.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tensorflow.Session(config = config)

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
nprf_model = None
embed = None
word_dict, iword_dict = None, None
df_map = None
train_generator = None

query_max_length, hist_size = 5, 30
doc_topk_term = 20
nb_supervised_doc = 10
nb_docs = 528030  # Total nr. of docs in the robust04 collection
fill_word = 0

OOV_dict = OrderedDict()
stem_word_map = dict()


def load_word_dict():
    global word_dict, iword_dict
    word_dict, iword_dict = read_word_dict("/home/fernando/NPRF/data/word_dict.v2.txt")


def load_df_map():
    global df_map
    df_map = df_map_from_file("/home/fernando/NPRF/data/word_stats.v2.txt")


def load_embeddings():
    global embed
    embed = KeyedVectors.load_word2vec_format("/home/fernando/drrm/wordembedding/rob04.d300.txt", binary=False)


def load_pair_generator(nprf_drmm_config, fold):
    pair_generator = NPRFDRMMPairGenerator(**nprf_drmm_config.generator_params)
    qid_list = deque(nprf_drmm_config.qid_list)
    rotate = fold - 1
    map(qid_list.rotate(rotate), qid_list)

    train_qid_list, valid_qid_list, test_qid_list = qid_list[0] + qid_list[1] + qid_list[2], qid_list[3], qid_list[4]
    global train_generator
    train_generator = pair_generator.generate_pair_batch(train_qid_list, nprf_drmm_config.pair_sample_size)


def preprocess_input_str(docs):
    docs = Preprocess.word_seg_en(docs)
    docs = Preprocess.word_lower(docs)
    docs_lower_copy = docs[0]
    docs = Preprocess.word_stem(docs)

    for idx, w in enumerate(docs[0]):
        stem_word_map[w] = docs_lower_copy[idx]
    return docs[0]


def prepare_input_data(query, doc, top_nb_doc_list, top_nb_doc_scores):
    query = preprocess_input_str([query])
    doc = preprocess_input_str([doc])
    doc = [w for w in doc if w in iword_dict]
    doc_to_print = [stem_word_map[w] for w in doc if w in stem_word_map]
    print(len(doc), doc)
    # print(len(doc), doc_to_print)
    dd_q_feat = np.zeros((nb_supervised_doc, doc_topk_term), dtype=np.float32)
    dd_d_hist_array = np.zeros((nb_supervised_doc, doc_topk_term, hist_size), dtype=np.float32)
    dd_d_feature = np.zeros((1, nb_supervised_doc, doc_topk_term,
                             hist_size), dtype=np.float32)
    dd_q_feature = np.zeros((1, nb_supervised_doc,
                             doc_topk_term, 1), dtype=np.float32)
    score_gate = np.zeros((1, nb_supervised_doc, 1), dtype=np.float32)

    supervised_doc_topk_terms = []
    for i, sup_d in enumerate(top_nb_doc_list[:nb_supervised_doc]):
        sup_d = preprocess_input_str([sup_d])
        sup_d = [iword_dict[w] for w in sup_d if w in iword_dict]
        sup_d, d_idf_pad = get_topk_terms(sup_d, doc_topk_term)
        sup_d = [word_dict[wid] for wid in sup_d if wid in word_dict]
        sup_d_to_print = [stem_word_map[w] for w in sup_d if w in stem_word_map]
        supervised_doc_topk_terms.append(sup_d_to_print)
        # print(len(sup_d), sup_d)
        # print(len(sup_d), sup_d_to_print)
        sim_mat_sup_d_doc = similarity_matrix(sup_d, doc, embed, OOV_dict)
        hist_sup_d_doc = hist_from_matrix(doc_topk_term, hist_size, sim_mat_sup_d_doc)
        # print(hist_sup_d_doc)
        dd_d_hist_array[i] = hist_sup_d_doc
        dd_q_feat[i] = d_idf_pad

    dd_q_feat = dd_q_feat.reshape((nb_supervised_doc, doc_topk_term, 1))
    dd_q_feature[0, :] = dd_q_feat

    dd_d_feature[0, :] = dd_d_hist_array
    score_gate[0, :] = get_normalized_baseline_scores(top_nb_doc_scores[:nb_supervised_doc])

    return [dd_q_feature, dd_d_feature, score_gate], supervised_doc_topk_terms


def get_normalized_baseline_scores(score_list):
    max_score, min_score = score_list[0], score_list[-1]
    baseline_score = np.asarray(score_list, dtype=np.float32)
    baseline_score = 0.5 * (baseline_score - min_score) / (max_score - min_score) + 0.5
    baseline_score = baseline_score.reshape((nb_supervised_doc, 1))
    return baseline_score


def get_topk_terms(doc_content, topk=20):
    tf_idf_score_list = []
    idf_score_list = []
    term_list = []
    doc_content = list(map(str, doc_content))
    term_count_pair = Counter(doc_content).most_common()
    for term, count in term_count_pair:
        count = int(count)
        if df_map.get(term) is None:
            # if term in stop_list or df_map.get(term) == None:
            pass
        else:
            df = df_map.get(term)
            idf = np.log((nb_docs - df + 0.5) / (df + 0.5))
            tfidf = count * idf
            tf_idf_score_list.append(tfidf)
            term_list.append(term)
            idf_score_list.append(idf)

    # print(len(term_list), len(idf_score_list))
    # sort the list
    tuple_list = zip(term_list, tf_idf_score_list, idf_score_list)
    sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1], reverse=True)
    topk_terms, _, topk_terms_idf = zip(*sorted_tuple_list)

    qualified_topk_terms = []
    topk_terms_idf_pad = np.zeros((topk,), dtype=np.float32)
    # idf_pad[:len(idf)] = idf
    index = 0
    while len(qualified_topk_terms) < topk and index < len(topk_terms):
        t = topk_terms[index]
        qualified_topk_terms.append(t)
        topk_terms_idf_pad[index] = topk_terms_idf[index]
        index += 1

    qualified_topk_terms = list(map(int, qualified_topk_terms))
    return qualified_topk_terms, topk_terms_idf_pad


@app.route('/')
def api_root():
    return 'NPRF DRMM model is already loaded\n'


@app.route("/score", methods=["POST"])
def predict():
    input_params = request.json
    query = input_params['query']
    doc = input_params['doc']
    topk_docs_content = input_params['topk_docs_content']
    topk_docs_scores = input_params['topk_docs_score']
    print(query)
    input_data, _ = prepare_input_data(query, doc, topk_docs_content, topk_docs_scores)
    # print(input_data)
    score = nprf_model.predict(input_data)
    print(score[0][0])
    return str(score[0][0])


@app.route("/prepare_test_input", methods=["POST"])
def prepare_test_input():
    input_params = request.json
    query = input_params['query']
    doc = input_params['doc']
    top_nb_doc_list = input_params["top_nb_doc_list"]
    top_nb_doc_scores = input_params["top_nb_doc_scores"]

    input_data, sup_doc_topk_terms = prepare_input_data(query, doc, top_nb_doc_list, top_nb_doc_scores)

    ret_val = dict()
    ret_val["dd_q_feature"] = input_data[0].tolist()
    ret_val["dd_d_feature"] = input_data[1].tolist()
    ret_val["score_gate"] = input_data[2].tolist()
    ret_val["sup_doc_topk_terms"] = sup_doc_topk_terms
    return jsonify(ret_val)


@app.route('/fetch_model_input_data')
def fetch_model_input_data():
    [dd_q, dd_d, score_gate], y = next(train_generator)

    ret_val = dict()
    ret_val["dd_q_feature"] = dd_q.tolist()
    ret_val["dd_d_feature"] = dd_d.tolist()
    ret_val["score_gate"] = score_gate.tolist()
    return jsonify(ret_val)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Shutting down NPRF DRMM flask server...\n'


def main(argv):

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    nprf_drmm_config = NPRFDRMMConfig()
    nprf_drmm_model = NPRFDRMM(nprf_drmm_config)
    global nprf_model
    nprf_model = nprf_drmm_model.build()

    weights_file = os.path.join(nprf_drmm_config.save_path, "fold4.h5")
    nprf_model.load_weights(weights_file)
    nprf_model._make_predict_function()

    load_word_dict()
    load_df_map()
    load_embeddings()
    load_pair_generator(nprf_drmm_config, fold=4)

    app.run(host="127.0.0.1", port=5010)
    # return


if __name__ == '__main__':
    main(sys.argv)
