from comet_ml import Experiment

import os
import logging
import tensorflow as tf
import sys
import math
sys.path.append('../utils/')
sys.path.append('../metrics/')

from file_operation import retain_file, load_pickle
from keras.callbacks import Callback
from keras import backend as K
from collections import deque

from rank_losses import rank_hinge_loss


class BasicModel(object):

  def __init__(self, config):
    self.config = config

  def build(self, *args, **kwargs):
    pass

  @staticmethod
  def _step_decay(epoch, initial_lr):
    initial_lrate = initial_lr
    drop = 0.9
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    if lrate > 0.00001:
      return lrate
    else:
      return 0.00001

  def train(self, model, pair_generator, fold, output_file, use_nprf=False):
    '''Driver function for training

    Args:
      model: a keras Model
      pair_generator: a instantiated pair generator
      fold: which fold to run. partitions will be automatically rotated.
      output_file: temporary file for validation
      use_nprf: whether to use nprf

    Returns:

    '''

    # set tensorflow not to use the full GPU memory
    # session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    initial_lrate = self.config.learning_rate
    experiment = Experiment(api_key="PhzBYNpSC304fMjGUoU42dX9b",
                            project_name="nprf-drmm", workspace="neural-ir", auto_param_logging=False)
    experiment_params = {
      "batch_size": self.config.batch_size,
      "optimizer": self.config.optimizer,
      "epochs": self.config.max_iteration,
      "initial_learning_rate": initial_lrate
    }
    experiment.log_multiple_params(experiment_params)

    # qid list config
    qid_list = deque(self.config.qid_list)
    rotate = fold - 1
    map(qid_list.rotate(rotate), qid_list)

    train_qid_list, valid_qid_list, test_qid_list = qid_list[0] + qid_list[1] + qid_list[2], qid_list[3], qid_list[4]
    print(train_qid_list, valid_qid_list, test_qid_list)
    relevance_dict = load_pickle(self.config.relevance_dict_path)
    # pair_generator = DDMPairGenerator(**self.config.generator_params)
    nb_pair_train = pair_generator.count_pairs_balanced(train_qid_list, self.config.pair_sample_size)

    valid_params = self.eval_by_qid_list_helper(valid_qid_list, pair_generator)
    test_params = self.eval_by_qid_list_helper(test_qid_list, pair_generator)

    print(valid_params[-1], test_params[-1])
    batch_logger = NBatchLogger(50)
    batch_losses = []
    met = [[], [], [], [], [], []]

    iteration = -1
    best_valid_map = 0.0
    new_lrate = initial_lrate

    for i in range(self.config.nb_epoch):
      print ("Epoch " + str(i))

      nb_batch = nb_pair_train / self.config.batch_size

      train_generator = pair_generator.generate_pair_batch(train_qid_list, self.config.pair_sample_size)
      for j in range(nb_batch / 100):
        iteration += 1
        new_lrate = self._step_decay(iteration, initial_lrate)
        K.set_value(model.optimizer.lr, new_lrate)

        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=100,  # nb_pair_train / self.config.batch_size,
                                      epochs=1,
                                      shuffle=False,
                                      verbose=0,
                                      callbacks=[batch_logger],
                                      )
        batch_losses.append(batch_logger.losses)
        print("[Iter {0}]\tLoss: {1}\tlr: {2}".format(iteration, history.history['loss'][0], new_lrate))
        experiment.log_parameter("curr_epoch", iteration+1, step=(iteration + 1))
        experiment.log_parameter("curr_lr", new_lrate, step=(iteration + 1))
        experiment.log_metric("curr_loss", history.history['loss'][0], step=(iteration + 1))

        kwargs = {'model': model,
                  'relevance_dict': relevance_dict,
                  'rerank_topk': self.config.rerank_topk,
                  'qrels_file': self.config.qrels_file,
                  'docnolist_file': self.config.docnolist_file,
                  'runid': self.config.runid,
                  'output_file': output_file}
        if use_nprf:
          kwargs.update({'nb_supervised_doc': self.config.nb_supervised_doc,
                         'doc_topk_term': self.config.doc_topk_term,})

        valid_met = self.eval_by_qid_list(*valid_params, **kwargs)
        print("[Valid]\t\tMAP\tP20\tNDCG20")
        print("\t\t{0}\t{1}\t{2}".format(valid_met[0], valid_met[1], valid_met[2]))
        met[0].append(valid_met[0])
        met[1].append(valid_met[1])
        met[2].append(valid_met[2])

        with experiment.validate():
          experiment.log_metric("map", valid_met[0], step=(iteration + 1))
          experiment.log_metric("p@20", valid_met[1], step=(iteration + 1))
          experiment.log_metric("ndcg@20", valid_met[2], step=(iteration + 1))

        if valid_met[0] > best_valid_map:
          model.save_weights(os.path.join(self.config.save_path, "fold{0}.h5".format(fold)))
          best_valid_map = valid_met[0]

        kwargs['output_file'] = os.path.join(self.config.result_path, "fold{0}.iter{1}.res".format(fold, iteration))
        # test_met = eval_partial(qid_list=test_qid_list)
        test_met = self.eval_by_qid_list(*test_params, **kwargs)
        print("[Test]\t\tMAP\tP20\tNDCG20")
        print("\t\t{0}\t{1}\t{2}".format(test_met[0], test_met[1], test_met[2]))
        met[3].append(test_met[0])
        met[4].append(test_met[1])
        met[5].append(test_met[2])

        with experiment.test():
          experiment.log_metric("map", test_met[0], step=(iteration + 1))
          experiment.log_metric("p@20", test_met[1], step=(iteration + 1))
          experiment.log_metric("ndcg@20", test_met[2], step=(iteration + 1))

      print("[Attention]\t\tCurrent best iteration {0}\n".format(met[0].index(max(met[0]))))
      if iteration > self.config.max_iteration:
        break
      # model.save_weights(os.path.join(self.config.save_path, "fold{0}.epoch{1}.h5".format(fold, i)))
    best_iter, eval_met = self._extract_max_metric(met)
    retain_file(self.config.result_path, "fold{0}".format(fold), "fold{0}.iter{1}.res".format(fold, best_iter))
    # np.save('loss.npy', batch_losses)
    # np.save('met.npy', met)
    return eval_met

  def _extract_max_metric(self, met):
    index = met[0].index(max(met[0]))
    logging.info('[EVAL RESULT] Based on best MAP on validation set')
    logging.info('Achieve best result on iteration {0}'.format(index))
    logging.info('\t\tMAP\tP@20\tNDCG@20')
    logging.info("[Valid]\t{0}\t{1}\t{2}".format(met[0][index], met[1][index], met[2][index]))
    logging.info("[Test]\t\t{0}\t{1}\t{2}".format(met[3][index], met[4][index], met[5][index]))

    return index, [met[3][index], met[4][index], met[5][index]]

  def eval_by_qid_list_helper(self, *args, **kwargs):
    pass

  def eval_by_qid_list(self, *args, **kwargs):
    pass


class NBatchLogger(Callback):
  def __init__(self, display=100):
    super(NBatchLogger, self).__init__()
    self.seen = 0
    self.display = display

  def on_train_begin(self, logs={}):
    self.losses = []
    self.map = []
    self.ndcg10 = []
    self.p10 = []

  def on_batch_end(self, batch, logs={}):
    self.seen += logs.get('size', 0)
    # if logs.get('batch') % self.display == 0:
    #  print '\nTrain on {0} Samples - Batch Loss: {1}'.format(self.seen, logs)
    self.losses.append(logs.get('loss'))
