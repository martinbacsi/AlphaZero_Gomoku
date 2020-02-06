# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Theano and Lasagne

@author: Junxiao Song
"""

from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
import pickle


class PolicyValueNet():
    """policy-value network """
    def __init__(self, model_file=None):
        self.learning_rate = T.scalar('learning_rate')
        self.l2_const = 1e-4  # coef of l2 penalty
        self.create_policy_value_net()
        self._loss_train_op()
        if model_file:
            try:
                net_params = pickle.load(open(model_file, 'rb'))
            except:
                # To support loading pretrained model in python3
                net_params = pickle.load(open(model_file, 'rb'),
                                         encoding='bytes')
            lasagne.layers.set_all_param_values(
                    [self.policy_net, self.value_net], net_params
                    )

    def create_policy_value_net(self):
        print("alma")
        """create the policy value network """
        self.state_input = T.tensor4('state')
        self.winner = T.vector('winner')
        self.mcts_probs = T.matrix('mcts_probs')
        network = lasagne.layers.InputLayer(
                shape=(None, 1, 1, 12),
                input_var=self.state_input
                )
        # conv layers
        network = lasagne.layers.DenseLayer(
            network, num_units=64,
            nonlinearity=lasagne.nonlinearities.softmax)
        network = lasagne.layers.DenseLayer(
            network, num_units=64,
            nonlinearity=lasagne.nonlinearities.softmax)
        network = lasagne.layers.DenseLayer(
            network, num_units=64,
            nonlinearity=lasagne.nonlinearities.softmax)
        # action policy layers

        self.policy_net = lasagne.layers.DenseLayer(
            network, num_units=12,
            nonlinearity=lasagne.nonlinearities.softmax)
        # state value layers

        self.value_net = lasagne.layers.DenseLayer(
            network, num_units=12,
            nonlinearity=lasagne.nonlinearities.softmax)
        # get action probs and state score value
        self.action_probs, self.value = lasagne.layers.get_output(
                [self.policy_net, self.value_net])
        self.policy_value = theano.function([self.state_input],
                                            [self.action_probs, self.value],
                                            allow_input_downcast=True)



    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
            action and the score of the board state
        """
        legal_positions = board.availables()

        current_state = board.current_state()
        act_probs, value = self.policy_value(
            current_state)
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        """
        params = lasagne.layers.get_all_params(
                [self.policy_net, self.value_net], trainable=True)
        value_loss = lasagne.objectives.squared_error(
                self.winner, self.value.flatten())
        policy_loss = lasagne.objectives.categorical_crossentropy(
                self.action_probs, self.mcts_probs)
        l2_penalty = lasagne.regularization.apply_penalty(
                params, lasagne.regularization.l2)
        self.loss = self.l2_const*l2_penalty + lasagne.objectives.aggregate(
                value_loss + policy_loss, mode='mean')
        # policy entropy，for monitoring only
        self.entropy = -T.mean(T.sum(
                self.action_probs * T.log(self.action_probs + 1e-10), axis=1))
        # get the train op
        updates = lasagne.updates.adam(self.loss, params,
                                       learning_rate=self.learning_rate)
        self.train_step = theano.function(
            [self.state_input, self.mcts_probs, self.winner, self.learning_rate],
            [self.loss, self.entropy],
            updates=updates,
            allow_input_downcast=True
            )

    def get_policy_param(self):
        net_params = lasagne.layers.get_all_param_values(
                [self.policy_net, self.value_net])
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
