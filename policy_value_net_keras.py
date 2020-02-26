# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Keras
Tested under Keras 2.0.5 with tensorflow-gpu 1.2.1 as backend

@author: Mingxu Zhang
"""

from __future__ import print_function

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K

from keras.utils import np_utils

import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """
    def __init__(self, model_file=None):

        self.l2_const = 1e-4  # coef of l2 penaltyd
        self.create_policy_value_net()
        self._loss_train_op()

        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)

    def create_policy_value_net(self):
        """create the policy value network """
        in_x = network = Input((7,))

        # conv layers
        network = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_const))(network)
        network = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_const))(network)
        network = Dense(32, activation='relu', kernel_regularizer=l2(self.l2_const))(network)
        network = Dense(32, activation='relu', kernel_regularizer=l2(self.l2_const))(network)


        self.policy_net = Dense(6, activation='softmax', kernel_regularizer=l2(self.l2_const))(network)
        # state value layers

        self.value_net = Dense(1, activation='tanh', kernel_regularizer=l2(self.l2_const))(network)

        self.model = Model(in_x, [self.policy_net, self.value_net])

        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        #legal_positions = board.availables
        #print(board.current_state())
        current_state = board.current_state()
        act_probs, value = self.policy_value( np.expand_dims(current_state ,0))
        #print(act_probs[0])
        #act_probs = zip(legal_positions, act_probs[0][legal_positions])

        actret = [(i, act_probs[0][i]) for i in range(6)]

        return actret, value[0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            #print(mcts_probs_union)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy

        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
