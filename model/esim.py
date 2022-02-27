# coding=utf-8
import tensorflow as tf
import config

FLAGS = config.FLAGS

class Graph:
    def __init__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.seq_length), name='premise')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.seq_length), name='hypothesis')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='label')
        self.p_mask = tf.cast(tf.math.equaL(self.p, 0), tf.float32)
        self.h_mask = tf.cast(tf.math.equal(self.h, 0), tf.float32)
        self.p_seq_len = tf.placeholder(dtype=tf.int32, shape=None, name="premise_seq_len")
        self.h_seq_len = tf.placeholder(dtypetf.int32, shape=None, name="hypothesis_seq_len")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="drop_rate")

        self.embedding = tf.get_variable(dtype=tf.float32,
                                         shape=(FLAGS.vocab_size, FLAGS.char_embedding_size),
                                         initializer=tf.truncated_normal_initializer(stddev=FLAGS.initializer_range),
                                         name="embedding")
        self.foword()


    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)


    def bilstm(self, x, hidden_size, seq_len):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=seq_len, dtype=tf.float32)


    def forward(self):
        with tf.name_scope("input_embedding"):
            p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
            h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        with tf.name_scope("input_encoding"):
            with tf.variable_scope("lstm_p", reuse=tf.AUTO_REUSE):
                (p_f, p_b), _ = self.bilstm(p_embedding, FLAGS.hidden_size, self.p_seq_len)
            with tf.variable_scope("lstm_h", reuse=tf.AUTO_REUSE):
                (h_f, h_b), _ = self.bilstm(h_embedding, FLAGS.hidden_size, self.h_seq_len)

            p = tf.concat([p_f, p_b], axis=2) # [batch_size, seq_max_len, 2*hidden_size]
            h = tf.concat([h_f, h_b], axis=2)

            p = self.dropout(p)
            h = self.dropout(h)

        with tf.name_scope("local_inference"):
            e = tf.matmul(p, tf.transpose(h, perm=[0, 2, 1]))
            a_attention = tf.nn.softmax(e) # [batch_size, seq_max_len, seq_max_len]
            b_attention = tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1]))

            a = tf.matmul(a_attention, h) # [batch_size, seq_max_len, 2*hidden_size]
            b = tf.matmul(b_attention, p)

            m_a = tf.concat([p, a, p-a, tf.multiply(a, p)], axis=2) # [batch_size, seq_max_len, 8*hidden_size]
            m_b = tf.concat([h, b, h-b, tf.multiply(b, h)], axis=2)

            """全连接层， 论文里说没用，但源码里有"""
            m_a = tf.layers.dense(m_a, FLAGS.hidden_size, activation="tanh")
            m_b = tf.layers.dense(m_b, FLAGS.hidden_size, activation="tanh")

        with tf.name_scope("inference_composition"):
            """论文里所有lstm hidden_size一样（与word_embedding_size也一致）"""
            with tf.variable_scope("lstm_a", reuse=tf.AUTO_REUSE):
                (a_f, a_b), _ = self.bilstm(m_a, FLAGS.hidden_size, self.p_seq_len)
            with tf.variable_scope("lstm_h", reuse=tf.AUTO_REUSE):
                (b_f, b_b), _ = self.bilstm(m_b, FLAGS.hidden_size, self.h_seq_len)

            a = tf.concat([a_f, a_b], axis=2) # [batch_size, seq_max_len, 2*hidden_size]
            b = tf.concat([b_f, b_b], axis=2)

            """pooling"""
            a_avg = tf.div(tf.reduce_sum(a, axis=1), tf.cast(tf.tile(tf.expand_dims(self.p_seq_len, -1), [1, 2*FLAGS.hidden_size]), dtype=tf.float32)) # [batch_size, 2*hidden_size]
            b_avg = tf.div(tf.reduce_sum(b, axis=1), tf.cast(tf.tile(tf.expand_dims(self.h_seq_len, -1), [1, 2*FLAGS.hidden_size]), dtype=tf.float32))

            p_mask_expand = tf.tile(tf.expand_dims(self.p_mask, -1), [1, 1, 2 * FLAGS.hidden_size]) # [batch_size, max_seq_len, 2*hidden_size]
            h_mask_expand = tf.tile(tf.expand_dims(self.h_mask, -1), [1, 1, 2 * FLAGS.hidden_size])
            a_max = tf.reduce_max(tf.add(a, p_mask_expand * (-1e7)), axis=1) # [batch_size, 2*hidden_size]
            b_max = tf.reduce_max(tf.add(b, h_mask_expand * (-1e7)), axis=1)

            v = tf.concat([a_avg, a_max, b_avg, b_max], axis=1)
            v = tf.layers.dense(v, FLAGS.dense_hidden_sieze, activation="tanh")
            v = self.dropout(v)

        logits = tf.layers.dense(v, 2)
        l2_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.l2_lambda), tf.trainable_variables())
        self.prob = tf.nn.softmax(logits)
        self.prediction = tf.argmax(logits, axis=1)

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("metrics"):
            label_id = tf.cast(tf.argmax(self.y, axis=1), dtype=tf.int32)
            self.metrics -= {
                "accuracy": tf.metrics.accuracy(label_id, self.prediction),
                "precision": tf.metrics.precision(label_id, self.prediction),
                "recall": tf.metrics.recall(label_id, self.prediction)
            }
            tf.summary.scalar("acc", self.metrics["accuracy"][1])
            tf.summary.scalar("prec", self.metrics["precision"][1])
            tf.summary.scalar("rec", self.metrics["recall"][1])

        self.merged = tf.summary.merge_all()
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)















