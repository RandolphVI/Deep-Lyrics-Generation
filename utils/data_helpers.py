# -*- coding: utf-8 -*-
__author__ = 'Randolph'

import os
import logging
import pickle
import numpy as np

TRAININGSET_DIR = '../data/all_training_lyrics.txt'
VALIDATIONSET_DIR = '../data/.txt'
MODEL_DIR = '../data/'


def logger_fn(name, file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    fh = logging.FileHandler(file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def create_char2vec_model(input_file=TRAININGSET_DIR):
    data = open(input_file, 'r').read()
    chars = list(set(data))
    DATA_SIZE, VOCAB_SIZE = len(data), len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    pickle.dump((char_to_idx, idx_to_char), open(MODEL_DIR + 'char2vec.voc', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    context_of_idx = [char_to_idx[ch] for ch in data]
    pickle.dump((context_of_idx, DATA_SIZE, VOCAB_SIZE),
                open(MODEL_DIR + 'text_vec.voc', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_model():
    model_file = '../data/text_vec.voc'

    if os.path.isfile(model_file):
        model = pickle.load(open(model_file, 'rb'))
        context_of_idx = model[0]
        DATA_SIZE = model[1]
        VOCAB_SIZE = model[2]
        logging.info('data has {0} characters, {1} unique.'.format(DATA_SIZE, VOCAB_SIZE))
        return context_of_idx, DATA_SIZE, VOCAB_SIZE
    else:
        logging.info("✘ The char2vec file doesn't exist. "
                     "Please use function <create_char2vec_model> to create it!")


def batch_iter(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[batch_len * i: batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1] # y 就是 x 的錯一位，即下一個詞
        yield (x, y)


raw_data, b, c = load_model()
raw_data = np.array(raw_data, dtype=np.int32)
data_len = len(raw_data)
a = batch_iter(raw_data, 32, 5)


