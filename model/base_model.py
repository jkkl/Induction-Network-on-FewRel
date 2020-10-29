# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/10/28 11:37
'''
import tensorflow as tf

class BaseModel:

    def __init__(self, kwds):
        self.sequence_length = kwds.get('sequence_length', 40)
        self.vocab_size = kwds.get('vocab_size', None)
        self.embed_size = kwds.get('embed_size', 50)
        self.hidden_size = kwds.get('hidden_size', 128)
        self.is_training = kwds.get('is_training', True)
        self.learning_rate = kwds.get('learning_rate', 0.001)
        self.initializer = kwds.get('initializer', tf.random_normal_initializer(stddev=0.1))
        # get new learning rate every decay_steps
        self.decay_steps = kwds.get('decay_steps', 100)
        self.decay_rate = kwds.get("decay_rate", 0.9)
        self.l2_lambda = kwds.get('l2_lambda', 0.00001)
        # load word embedding
        self.embed = kwds.get('pred_embed', None)
        self.pos_embeding_dim = 5
        self.keep_prob = kwds.get('keep_prob', 0.9)
        self.alphas = None

    def build(self):
        self.initial_params()

    def initial_params(self):
        ''' step'''
        self.global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)

        # input_words batch = [(k_query + k_support)*C, sequence_length]
        self.input_words = tf.placeholder(name='input_words', shape=[None, self.sequence_length], dtype=tf.int32)
        # input_pos1 batch = [(k_query + k_support)*C, sequence_length] TODO: do what?
        self.input_pos1 = tf.placeholder(name='input_pos1', shape=[None, self.sequence_length], dtype=tf.int32)
        # input_pos2 batch = [(k_query + k_support)*C, sequence_length] TODO: do what?
        self.input_pos2 = tf.placeholder(name='input_pos1', shape=[None, self.sequence_length], dtype=tf.int32)
        # query_label [batch = key_query*C ]
        self.query_label = tf.placeholder(name='query_label', shape=[], dtype=tf.int32)
        # 每个query的真是长度
        self.mask_padding = tf.placeholder(name='mask_padding', shape=[None, self.sequence_length])
        # TODO: do what
        self.keep_prob_x = tf.placeholder(name='keep_prob_x', dtype=tf.float32)

        # embedding matrix
        with tf.name_scope('embedding'):
            if self.embed is not None:
                print('use word embedding, size:', self.embed.shape)
                self.word_embedding = tf.Variable(self.embed, trainable=False)
                self.embed_size = self.embed.shape[1]
            else:
                self.word_embedding = tf.get_variable(name='word_embedding',
                                                      shape=[self.vocab_size, self.embed_size],
                                                      initializer=self.initializer,
                                                      trainable=True
                                                      )
            # pos1_embedding: [2*seq_length, pos_embed_size=5] TODO: do what?
            self.pos1_embedding = tf.get_variable(name='pos1_embedding',
                                                  shape=[2 * self.sequence_length, self.pos_embeding_dim],
                                                  initializer=self.initializer, trainable=True
                                                  )
            # pos2_embedding: [2*seq_length, pos_embed_size=5]
            self.pos2_embedding = tf.get_variable(name='pos2_embedding',
                                                  shape=[2*self.sequence_length, self.pos_embeding_dim],
                                                  initializer=self.initializer,
                                                  trainable=True
                                                  )


    def get_embedding(self):
        # input_words: [batch, seq_length]
        # word_embedding: [vocab_size, embed_size]
        # embedded_words: [batch, seq_length, embed_size]
        embedded_words = tf.nn.embedding_lookup(self.word_embedding, self.input_words)
        # embedd_pos1: [batch, seq_length, pos_embed_size]
        embedded_pos1 = tf.nn.embedding_lookup(self.pos1_embedding, self.input_pos1)
        # embedd_pos2: [batch, seq_length, pos_embed_size]
        embedded_pos2 = tf.nn.embedding_lookup(self.pos2_embedding, self.input_pos2)
        # output: [batch, seq_length, embed_size + 2*pos_embed_size]
        output = tf.concat([embedded_words, embedded_pos1, embedded_pos2], axis=2)
        return output

    def forward(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def build_optimize(self):
        # based on the loss, use SGD to update parameter
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                        self.decay_steps, self.decay_rate, staircase=True)
        self.optimize = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                        learning_rate=self.learning_rate, optimizer='Adam')

    def build_predict(self):
        self.predict_test = tf.argmax(name='predictions', input=self.probs, axis=1)
        self.predict = tf.round(self.probs)

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.cast(self.predict_test, tf.int32), self.query_label)
        self.accuracy = tf.reduce_mean(name='accuracy', input_tensor=tf.cast(correct_prediction, tf.int32))

