import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.contrib.rnn import LSTMCell, BasicRNNCell, GRUCell, DropoutWrapper
import code0_hyperParameter as code0
import pyprind as pp
import os, sys
from uril_tools import DEBUG_PRINT


class Model(object):
    def __init__(self, is_training, config):
        self._batch_size = batch_size = config.batch_size

        self._min_lr = config.min_lr
        self.hidden_size = hidden_size = config.hidden_size
        self.hidden_size_2 = hidden_size_2 = config.hidden_size_2
        self.seq_width = seq_width = config.seq_width
        self.num_steps = num_steps = config.num_steps
        self.skill_num = skill_numb = config.skill_num
        self.skill_set = config.skill_set

        DEBUG_PRINT(1)

        # load data
        self.inputs = tf.placeholder(tf.float32, [batch_size, num_steps, seq_width])
        self._target_id = tf.placeholder(tf.int32, [None])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])

        self.generNPC_idx = tf.placeholder(tf.int32, [None])
        self.generNPC_list = tf.placeholder(tf.float32, [None])

        self.first_action_idx = tf.placeholder(tf.int32, [None])
        self.first_action_list = tf.placeholder(tf.float32, [None])

        self.res_correct_idx = tf.placeholder(tf.int32, [None])
        self.res_correct_list = tf.placeholder(tf.float32, [None])

        self.wheelspin_idx = tf.placeholder(tf.int32, [None])
        self.wheelspin_list = tf.placeholder(tf.float32, [None])

        self.bored_idx = tf.placeholder(tf.int32, [None])
        self.bored_list = tf.placeholder(tf.float32, [None])

        self.concentrating_idx = tf.placeholder(tf.int32, [None])
        self.concentrating_list = tf.placeholder(tf.float32, [None])

        self.confused_idx = tf.placeholder(tf.int32, [None])
        self.confused_list = tf.placeholder(tf.float32, [None])

        self.frustrated_idx = tf.placeholder(tf.int32, [None])
        self.frustrated_list = tf.placeholder(tf.float32, [None])

        # load features
        input_RNN = self.inputs

        DEBUG_PRINT(2)

        cell = self.getCell(is_training=is_training, config=config)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        outputs = []
        state = self._initial_state

        DEBUG_PRINT(3)

        with tf.variable_scope(config.cell_type):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(input_RNN[:, time_step, :], state)
                outputs.append(cell_output)

        if config.num_layer == 1:
            size_rnn_out = hidden_size
        elif config.num_layer == 2:
            size_rnn_out = hidden_size_2
        else:
            raise ValueError("only support 1-2 layers, check your layer number!")

        DEBUG_PRINT(4)
        # print (np.array(outputs))
        output_RNN = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size_rnn_out])

        softmax_w = tf.get_variable("softmax_w", [size_rnn_out, skill_numb])
        softmax_b = tf.get_variable("softmax_b", [skill_numb])
        logits = tf.matmul(output_RNN, softmax_w) + softmax_b
        # pick up the right one
        self.logits = logits = tf.reshape(logits, [-1])
        self.selected_logits = selected_logits = tf.gather(logits, self.target_id)
        # make prediction
        self._pred = self._pred_values = tf.sigmoid(selected_logits)
        # npc_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(selected_logits, target_correctness))
        npc_loss_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=target_correctness))
        npc_loss_sum = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=target_correctness))

        DEBUG_PRINT(5)

        generNPC_softmax_w = tf.get_variable("generNPC_softmax_w", [size_rnn_out, 1])
        generNPC_softmax_b = tf.get_variable("generNPC_softmax_b", [1])
        generNPC_logits = tf.matmul(output_RNN, generNPC_softmax_w) + generNPC_softmax_b
        generNPC_logits = tf.reshape(generNPC_logits, [-1])
        selected_generNPC__logits = tf.gather(generNPC_logits, self.generNPC_idx)
        self.generNPC_pred = tf.sigmoid(selected_generNPC__logits)
        generNPC_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_generNPC__logits, labels=self.generNPC_list))

        first_action_softmax_w = tf.get_variable("first_action_softmax_w", [size_rnn_out, 1])
        first_action_softmax_b = tf.get_variable("first_action_softmax_b", [1])
        first_action_logits = tf.matmul(output_RNN, first_action_softmax_w) + first_action_softmax_b
        first_action_logits = tf.reshape(first_action_logits, [-1])
        selected_first_action__logits = tf.gather(first_action_logits, self.first_action_idx)
        self.first_action_pred = tf.sigmoid(selected_first_action__logits)
        first_action_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_first_action__logits, labels=self.first_action_list))

        res_correct_softmax_w = tf.get_variable("res_correct_softmax_w", [size_rnn_out, 1])
        res_correct_softmax_b = tf.get_variable("res_correct_softmax_b", [1])
        res_correct_logits = tf.matmul(output_RNN, res_correct_softmax_w) + res_correct_softmax_b
        res_correct_logits = tf.reshape(res_correct_logits, [-1])
        selected_res_correct__logits = tf.gather(res_correct_logits, self.res_correct_idx)
        self.res_correct_pred = tf.sigmoid(selected_res_correct__logits)
        res_correct_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_res_correct__logits, labels=self.res_correct_list))

        wheelspin_softmax_w = tf.get_variable("wheelspin_softmax_w", [size_rnn_out, 1])
        wheelspin_softmax_b = tf.get_variable("wheelspin_softmax_b", [1])
        wheelspin_logits = tf.matmul(output_RNN, wheelspin_softmax_w) + wheelspin_softmax_b
        wheelspin_logits = tf.reshape(wheelspin_logits, [-1])
        selected_wheelspin__logits = tf.gather(wheelspin_logits, self.wheelspin_idx)
        self.wheelspin_pred = tf.sigmoid(selected_wheelspin__logits)
        wheelspin_loss_mean = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_wheelspin__logits, labels=self.wheelspin_list))
        wheelspin_loss_sum = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_wheelspin__logits, labels=self.wheelspin_list))

        bored_softmax_w = tf.get_variable("bored_softmax_w", [size_rnn_out, 1])
        bored_softmax_b = tf.get_variable("bored_softmax_b", [1])
        bored_logits = tf.matmul(output_RNN, bored_softmax_w) + bored_softmax_b
        bored_logits = tf.reshape(bored_logits, [-1])
        selected_bored__logits = tf.gather(bored_logits, self.bored_idx)
        self.bored_pred = tf.sigmoid(selected_bored__logits)
        bored_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_bored__logits, labels=self.bored_list))

        concentrating_softmax_w = tf.get_variable("concentrating_softmax_w", [size_rnn_out, 1])
        concentrating_softmax_b = tf.get_variable("concentrating_softmax_b", [1])
        concentrating_logits = tf.matmul(output_RNN, concentrating_softmax_w) + concentrating_softmax_b
        concentrating_logits = tf.reshape(concentrating_logits, [-1])
        selected_concentrating__logits = tf.gather(concentrating_logits, self.concentrating_idx)
        self.concentrating_pred = tf.sigmoid(selected_concentrating__logits)
        concentrating_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_concentrating__logits, labels=self.concentrating_list))

        confused_softmax_w = tf.get_variable("confused_softmax_w", [size_rnn_out, 1])
        confused_softmax_b = tf.get_variable("confused_softmax_b", [1])
        confused_logits = tf.matmul(output_RNN, confused_softmax_w) + confused_softmax_b
        confused_logits = tf.reshape(confused_logits, [-1])
        selected_confused__logits = tf.gather(confused_logits, self.confused_idx)
        self.confused_pred = tf.sigmoid(selected_confused__logits)
        confused_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_confused__logits, labels=self.confused_list))

        frustrated_softmax_w = tf.get_variable("frustrated_softmax_w", [size_rnn_out, 1])
        frustrated_softmax_b = tf.get_variable("frustrated_softmax_b", [1])
        frustrated_logits = tf.matmul(output_RNN, frustrated_softmax_w) + frustrated_softmax_b
        frustrated_logits = tf.reshape(frustrated_logits, [-1])
        selected_frustrated__logits = tf.gather(frustrated_logits, self.frustrated_idx)
        self.frustrated_pred = tf.sigmoid(selected_frustrated__logits)
        frustrated_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_frustrated__logits, labels=self.frustrated_list))

        # total_loss = first_action_loss + res_correct_loss + wheelspin_loss + bored_loss + concentrating_loss +
        # confused_loss + frustrated_loss

        DEBUG_PRINT(6)

        if code0.INCLUDE_MODEL == 1:
            total_loss = npc_loss_sum
        elif code0.INCLUDE_MODEL == 2:
            total_loss = wheelspin_loss_sum
        elif code0.INCLUDE_MODEL == 3:
            total_loss = first_action_loss + wheelspin_loss_sum
        elif code0.INCLUDE_MODEL == 4:
            total_loss = npc_loss_sum + first_action_loss + res_correct_loss + wheelspin_loss_sum + bored_loss + concentrating_loss + confused_loss + frustrated_loss
        elif code0.INCLUDE_MODEL == 5:
            total_loss = res_correct_loss
        elif code0.INCLUDE_MODEL == 6:
            total_loss = generNPC_loss

        self._cost = total_loss

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), config.max_grad_norm)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.1)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        if (lr_value > self.min_lr):
            session.run(tf.assign(self._lr, lr_value))
        else:
            session.run(tf.assign(self._lr, self.min_lr))

    def getCell(self, is_training, config):
        # code for RNN
        if is_training == True:
            print("==> Construct ", config.cell_type, " graph for training")
        else:
            print("==> Construct ", config.cell_type, " graph for testing")

        if config.cell_type == "LSTM":
            if config.num_layer == 1:
                basicCell = LSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            elif config.num_layer == 2:
                basicCell = LSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
                basicCell_2 = LSTMCell(config.hidden_size_2, forget_bias=0.0, state_is_tuple=True)
            else:
                raise ValueError("config.num_layer should be 1:2 ")
        elif config.cell_type == "RNN":
            if config.num_layer == 1:
                basicCell = BasicRNNCell(config.hidden_size)
            elif config.num_layer == 2:
                basicCell = BasicRNNCell(config.hidden_size)
                basicCell_2 = BasicRNNCell(config.hidden_size_2)
            else:
                raise ValueError("config.num_layer should be [1-3] ")
        elif config.cell_type == "GRU":
            if config.num_layer == 1:
                basicCell = GRUCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            elif config.num_layer == 2:
                basicCell = GRUCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
                basicCell_2 = GRUCell(config.hidden_size_2, forget_bias=0.0, state_is_tuple=True)
            else:
                raise ValueError("only support 1-2 layers ")
        else:
            raise ValueError("cell type should be GRU,LSTM,RNN")

            # add dropout layer between hidden layers
        if is_training and config.keep_prob < 1:
            if config.num_layer == 1:
                basicCell = DropoutWrapper(basicCell, input_keep_prob=config.keep_prob,
                                           output_keep_prob=config.keep_prob)
            elif config.num_layer == 2:
                basicCell = DropoutWrapper(basicCell, input_keep_prob=config.keep_prob,
                                           output_keep_prob=config.keep_prob)
                basicCell_2 = DropoutWrapper(basicCell_2, input_keep_prob=config.keep_prob,
                                             output_keep_prob=config.keep_prob)
            else:
                pass

        if config.num_layer == 1:
            cell = rnn_cell.MultiRNNCell([basicCell], state_is_tuple=True)
        elif config.num_layer == 2:
            cell = rnn_cell.MultiRNNCell([basicCell, basicCell_2], state_is_tuple=True)

        return cell

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def min_lr(self):
        return self._min_lr

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def pred_values(self):
        return self._pred_values

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


if __name__ == "__main__":
    pass
