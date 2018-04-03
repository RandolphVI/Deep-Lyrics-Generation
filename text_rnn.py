# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf


class TextRNN(object):
    def __init__(self, avg_seq_len, vocab_size, hidden_size, num_hidden_layers,
                 embedding_size, l2_reg_lambda=0.0):
        self.batch_size = batch_size = config.batch_size
        self.lr = config.learning_rate

        self.input_x = tf.placeholder(tf.int32, [batch_size, avg_seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [batch_size, avg_seq_len], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Gobal_Step")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", [vocab_size, size])   # hidden_size?
            self.embedded_char = tf.nn.embedding_lookup(self.embedding, self.input_y)


        with tf.name_scope("lstm"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=False)

            if self.dropout_keep_prob is not None:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_hidden_layers, state_is_tuple=False)

            self.initial_state = cell.zero_state(batch_size, tf.float32)

            outputs = []
            state =
            for time_step in range(avg_seq_len):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(input())

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(avg_seq_len):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state) #inputs[:, time_step, :] 的 shape 是 (batch_size, size)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])

        with tf.name_scope("output"):
            softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])
            logits = tf.matmul(output, softmax_w) + softmax_b
            self.final_state = state

        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

