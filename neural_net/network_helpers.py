"""
Copyright 2019 Rahul Gupta, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf

def get_single_cell(memory_dim, initializer, dropout, keep_prob, which):
    if which == 'LSTM':
        constituent_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(memory_dim, initializer=initializer, state_is_tuple=True)
    elif which == 'GRU':
        constituent_cell = tf.compat.v1.nn.rnn_cell.GRUCell(memory_dim, kernel_initializer=initializer)
    else:
        raise ValueError('Unsupported rnn cell type: %s' % which)
    if dropout != 0:
        constituent_cell = tf.contrib.rnn.DropoutWrapper(constituent_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    return constituent_cell

def new_RNN_cell(memory_dim, num_layers, initializer, dropout=0, keep_prob=None, which='LSTM'):

    assert memory_dim is not None and num_layers is not None and dropout is not None, 'At least one of the arguments is passed as None'

    if num_layers > 1:
        return tf.compat.v1.nn.rnn_cell.MultiRNNCell([get_single_cell(memory_dim, initializer, dropout, keep_prob, which) for _ in range(num_layers) ])
    else:
        return get_single_cell(memory_dim, initializer, dropout, keep_prob, which)


