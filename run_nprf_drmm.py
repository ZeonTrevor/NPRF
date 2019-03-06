# -*- coding: utf8 -*-
from __future__ import print_function

import json
import argparse
import sys
import argparse
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
import keras
import keras.backend as K
from keras.models import Sequential, Model

from matchzoo.utils import *
from matchzoo.optimizers import *
from matchzoo.models import *
from matchzoo.inputs.preprocess import *

# config = tensorflow.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tensorflow.Session(config = config)

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
embed = None
word_dict, iword_dict = None, None
query_max_length, hist_size = 5, 30
fill_word = 0


def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo


def load_word_dict():
    global word_dict, iword_dict
    word_dict, iword_dict = read_word_dict("/home/fernando/MatchZoo/data/robust04/word_dict_new_n_stem_filtered_rob04_embed.txt")


def cal_hist(t1_rep, t2_rep, qnum, hist_size):
    mhist = np.zeros((qnum, hist_size), dtype=np.float32)
    mm = t1_rep.dot(np.transpose(t2_rep))
    for (i,j), v in np.ndenumerate(mm):
        if i >= qnum:
            break
        vid = int((v + 1.) / 2. * (hist_size - 1.))
        mhist[i][vid] += 1.
    mhist += 1.
    mhist = np.log10(mhist)
    return mhist


def preprocess_input_str(docs):
    docs = Preprocess.word_seg_en(docs)
    docs = Preprocess.word_stem(docs)
    docs = Preprocess.word_lower(docs)
    return docs[0]


def prepare_input_data(query, doc):
    query = preprocess_input_str([query])
    doc = preprocess_input_str([doc])
    query = [iword_dict[w] for w in query if w in iword_dict]
    doc = [iword_dict[w] for w in doc if w in iword_dict]

    X1 = np.zeros((1, query_max_length), dtype=np.int32)
    X1_len = np.zeros((1,), dtype=np.int32)
    X2 = np.zeros((1, query_max_length, hist_size), dtype=np.float32)
    X2_len = np.zeros((1,), dtype=np.int32)
    X1[:] = fill_word

    q_cont = list(query)
    q_len = min(query_max_length, len(query))
    d_cont = list(doc)
    d_len = len(d_cont)
    X1[0, :q_len], X1_len[0] = q_cont[:q_len], q_len
    X2[0], X2_len[0] = cal_hist(embed[q_cont], embed[d_cont], query_max_length, hist_size), d_len
    #return {'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}
    return {'query': X1, 'doc': X2}


@app.route('/')
def api_root():
    return 'DRMM model is already loaded\n'


@app.route("/score", methods=["POST"])
def predict():
    input_params = request.json
    query = input_params['query']
    doc = input_params['doc']
    input_data = prepare_input_data(query, doc)
    print(input_data)
    score = model.predict(input_data)
    print(score[0][0])
    return str(score[0][0])


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Shutting down DRMM flask server...\n'


def main(argv):

    parser = argparse.ArgumentParser()
    #parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    #parser.add_argument('--model_file', default='./models/arci.config', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file = "/home/fernando/MatchZoo/examples/robust04/config/drmm_ranking.config"  # args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)

    global_conf = config["global"]
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    global query_max_length, hist_size, fill_word
    query_max_length = share_input_conf['text1_maxlen']
    hist_size = share_input_conf['hist_size']
    fill_word = share_input_conf['vocab_size'] - 1

    global embed
    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'],), dtype=np.float32)
        embed = np.float32(
            np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed=embed)
    else:
        embed = np.float32(
            np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed

    embed = share_input_conf['embed']
    print('[Embedding] Embedding Load Done.', end='\n')

    global model
    model = load_model(config)

    weights_file = global_conf["best_weights_file"]
    model.load_weights(weights_file)

    model._make_predict_function()
    load_word_dict()
    app.run(host="127.0.0.1", port=5000)
    #return


if __name__=='__main__':
    main(sys.argv)
