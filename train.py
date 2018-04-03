# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import pickle
import logging
import numpy as np
import tensorflow as tf

from text_rnn import TextRNN
from utils import data_helpers as dh

if not os.path.exists('./model'):
    os.makedirs('./model')


# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R) \n")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input('✘ The format of your input is illegal, please re-input: ')
logging.info('✔︎ The format of your input is legal, now loading to next step...')

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn('tflog', 'logs/training-{0}.log'.format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn('tflog', 'logs/restore-{0}.log'.format(time.asctime()))

MODEL_DIR = 'data/'
TRAININGSET_DIR = 'data/all_training_lyrics.txt'
VALIDATIONSET_DIR = 'data/.txt'

# Data loading params
tf.flags.DEFINE_string("model_path", MODEL_DIR, "Path for store the model file.")
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")


# Model Hyperparameterss
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("avg_seq_len", 25, "Average lyrics sequence length of data (depends on the data)")
# tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
# tf.flags.DEFINE_integer("hidden_size", 256, "Hidden size for bi-lstm layer(default: 256)")
tf.flags.DEFINE_integer("num_hidden_layers", 3, "Number of hidden layers")
# tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Generation parameters
tf.flags.DEFINE_integer("save_time", 40, "Load save time to save models")
tf.flags.DEFINE_boolean("is_sample", True, "True means using sample, if not using max")
tf.flags.DEFINE_boolean("is_beams", True, "Whether or not using beam search")
tf.flags.DEFINE_integer("beam_size", 2, "Size of beam search")
tf.flags.DEFINE_integer("len_of_generation", 200, "The number of characters by generated")
tf.flags.DEFINE_string("start_sentence", "沒有菸抽的日子", "The seed sentence to generate text")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

init_scale = 0.04
max_grad_norm = 15
save_freq = 5  # The step (counted by the number of iterations) at which the model is saved to hard disk.

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), value)
                                for attr, value in sorted(FLAGS.__flags.items())], dilim]))

def train():
    train_data, DATA_SIZE, VOCAB_SIZE = dh.load_model()

    with tf.Graph().as_default() as session:
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            rnn = TextRNN(
                avg_seq_len = FLAGS.avg_seq_len,
                vocab_size=VOCAB_SIZE,
                hidden_size=,
                num_hidden_layers = FLAGS.num_hidden_layers,
                embedding_size=,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
            )
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

        # Generate batches
        batches_train = dh

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model.Model(is_training=True, config=config)

        tf.global_variables_initializer().run()

        model_saver = tf.train.Saver(tf.global_variables())

        for i in range(FLAGS.num_epochs):
            logging.info("Training Epoch: %d ..." % (i+1))
            train_perplexity = run_epoch(session, m, train_data, m.train_op)
            logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            if (i+1) % config.save_freq == 0:
                print 'model saving ...'
                model_saver.save(session, config.model_path+'-%d'%(i+1))
                print 'Done!'

def data_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)] # data 的 shape 是 (batch_size, batch_len)，每一行是連貫的一段，一次可輸入多個段落

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1] # y 就是 x 的錯一位，即下一個詞
        yield (x, y)

def run_epoch(session, m, data, eval_op):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(data_iterator(data, m.batch_size,
                                                    m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op], # x 和 y 的 shape 都是 (batch_size, num_steps)
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if step and step % (epoch_size // 10) == 0:
            print("%.2f perplexity: %.3f cost-time: %.2f s" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 (time.time() - start_time)))
            start_time = time.time()

    return np.exp(costs / iters)


if __name__ == "__main__":
    tf.app.run()