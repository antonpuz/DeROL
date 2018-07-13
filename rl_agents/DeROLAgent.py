import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops

class DeROLAgent:
    # Action are = {class 0, class 1.....Request classification, Holdout}
    def __init__(self, batch_size, samples_per_batch, total_input_size, actions, lstm_units=200, learning_rate=0.001):
        # Constants
        self.policy_lstm_units = lstm_units
        self.input_size = total_input_size
        self.actions = actions
        self.number_of_classes = actions-2

        self.X = tf.placeholder("float", [batch_size, None, self.input_size])
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0, tf.float32), None)

        self.policy_lstm_cell = rnn.BasicLSTMCell(self.policy_lstm_units, forget_bias=1.0, activation=math_ops.tanh)
        self.policy_state_in = self.policy_lstm_cell.zero_state(batch_size, tf.float32)

        self.timeseries_length = tf.placeholder_with_default(tf.constant(samples_per_batch, tf.float32), None)
        self.learning_rate = tf.placeholder_with_default(tf.constant(learning_rate, tf.float32), None)

        self.policy_weights = {
            'w1': tf.Variable(tf.random_normal([self.policy_lstm_units, 128], stddev=0.01)),
            'out': tf.Variable(tf.random_normal([128, actions], stddev=0.01))
        }
        self.policy_biases = {
            'b1': tf.Variable(tf.random_normal([128])),
            'out': tf.Variable(tf.random_normal([actions]))
        }

        # DQN
        self.policy_rnnex_t, self.policy_rnn_state = tf.nn.dynamic_rnn( \
            inputs=self.X, cell=self.policy_lstm_cell, dtype=tf.float32, initial_state=self.policy_state_in, sequence_length=self.timeseries_length)
        self.policy_rnnex = tf.reshape(self.policy_rnnex_t, [-1, self.policy_lstm_units])

        self.hidden_layer = tf.nn.tanh(tf.add(tf.matmul(self.policy_rnnex, self.policy_weights['w1']), self.policy_biases['b1']))
        self.policy_logits = tf.add(tf.matmul(self.hidden_layer, self.policy_weights['out']), self.policy_biases['out'])
        self.policy_logits = tf.reshape(self.policy_logits, [-1, actions])

        # Policy
        self.predicted_action = tf.argmax(self.policy_logits, 1)

        # Training
        self.actions = tf.placeholder(shape=[batch_size, samples_per_batch], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, actions, dtype=tf.float32)
        self.Q_calculation = tf.placeholder(shape=[batch_size, samples_per_batch], dtype=tf.float32)

        self.actions_onehot_reshaped = tf.reshape(self.actions_onehot, [-1, actions])
        self.Q = tf.reduce_sum(tf.multiply(self.policy_logits, self.actions_onehot_reshaped), axis=1)
        self.Q_reshaped = tf.reshape(self.Q, [batch_size, samples_per_batch])

        self.loss = tf.reduce_sum(tf.square(self.Q_reshaped - self.Q_calculation))

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)
