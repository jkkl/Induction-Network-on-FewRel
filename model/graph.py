# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/10/29 15:40
'''
import tensorflow as tf
from model.base_model import BaseModel
from tensorflow.contrib import rnn


class InductionGraph(BaseModel):
    def __init__(self, N, K, Q, **Kwds):
        BaseModel.__init__(self, Kwds)
        self.num_classes = N
        self.support_num_per_class = K
        self.query_num_per_class = Q
        self.build()

    def forward(self):
        with tf.name_scope('EncoderModule'):
            # embeding_words: [batch=k*c, seq_length, emb_size=embed_size + 2*pos_embed_size]
            embedded_words = self.get_embedding()

            lstm_fw_cell = rnn.BasicLSTMCell(num_units=self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(num_units=self.hidden_size)
            if self.keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, out_keep_prob=self.keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, out_keep_prob=self.keep_prob)

            outputs, states = tf.nn.Bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                              cell_bw_cell=lstm_bw_cell,
                                                              inputs=embedded_words,
                                                              dtype=tf.float32)
            output_rnn = tf.concat(outputs, axis=2)
            mask_padding = tf.cast(self, mak)



