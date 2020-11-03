# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/10/29 15:40
'''
import tensorflow as tf
from model.base_model import BaseModel


class InductionGraph(BaseModel):
    def __init__(self, N, K, Q, **Kwds):
        BaseModel.__init__(self, Kwds)
        self.num_classes = N
        self.support_num_per_class = K
        self.query_num_per_class = Q
        self.build()

    def forward(self):
        with tf.name_scope('EncoderModule'):
            embedded_words = self.get_embedding()

