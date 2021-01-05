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

import os
import sys
import math
import argparse
import copy
import psutil
import threading
import multiprocessing
import numpy as np
import scipy.signal

from time import sleep, time, strftime
from util.helpers import make_dir_if_not_exists, logger, prepare_batch, experience, get_best_checkpoint, coin_flip, Compilation_error_db, done
from util.env import Environment, load_data

import tensorflow as tf
import tf_slim

from network_helpers import new_RNN_cell

tf.compat.v1.disable_eager_execution()

# Adapted from Arthur Juliani's A3C-doom notebook hosted on github:
# https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

# Helper Functions
def update_target_graph(from_scope, to_scope):
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def normalized_columns_initializer(rng, std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = rng.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def make_feed_dict(obj, in_batch, in_len, dropout, actions=None, target_v=None, advantages=None, Train=True):
    feed_dict = {obj.in_batch: in_batch, obj.in_len: in_len}
    if dropout != 0:
        feed_dict.update({obj.keep_prob: (1 - dropout if Train else 0)})
    if target_v is not None:
        feed_dict.update({obj.target_v: target_v})
    if actions is not None:
        feed_dict.update({obj.actions: actions})
    if advantages is not None:
        feed_dict.update({obj.advantages: advantages})
    return feed_dict

def join_str(*args):
    seprator = ' '
    return seprator.join(map(str, args)) + '\n'

# Actor-Critic
class AC_Network():

    def __init__(self, args, vocab_size, num_actions, scope, trainer, seed):
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.vocab_size = vocab_size
        self.embedding_size = args.embedding_dim
        self.num_actions = num_actions

        self.scope = scope

        self.learning_rate = self.args.learning_rate
        
        self.trainer = trainer
        self.rng = np.random.RandomState(seed)

        with tf.compat.v1.variable_scope(scope):

            tf.compat.v1.set_random_seed(args.seed)

            self.in_batch = tf.compat.v1.placeholder(shape=(None, None), dtype=tf.int32)
            self.in_len = tf.compat.v1.placeholder(shape=(None,), dtype=tf.int32)
            if self.dropout != 0:
                self.keep_prob = tf.compat.v1.placeholder(tf.float32)
            else:
                self.keep_prob = None

            self.embedding_matrix = tf.compat.v1.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                dtype=tf.float32)

            self.in_batch_embedded = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=self.in_batch)
            cell = new_RNN_cell(self.hidden_dim, self.num_layers,
                                tf.compat.v1.random_uniform_initializer(-0.1, 0.1), self.dropout, self.keep_prob)
            self.rnn_out, self.rnn_state = tf.compat.v1.nn.dynamic_rnn(
                cell, self.in_batch_embedded, dtype=tf.float32, sequence_length=self.in_len)

            self.embedding = tf.reduce_mean(input_tensor=self.rnn_out, axis=1)

            self.policy = tf_slim.fully_connected(self.embedding, self.num_actions, activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(self.rng, 1.0),
                                               biases_initializer=None)
            self.value = tf_slim.fully_connected(self.embedding, 1, activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(self.rng, 1.0),
                                              biases_initializer=None)

            self.policy = tf.clip_by_value(self.policy, 1e-10, 1.)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
                self.target_v = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(input_tensor=self.policy * self.actions_onehot, axis=[1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(input_tensor=tf.square(self.target_v - tf.reshape(self.value, [-1])))

                self.entropy = - tf.reduce_sum(input_tensor=self.policy * tf.math.log(self.policy))
                self.policy_loss = - tf.reduce_sum(input_tensor=tf.math.log(self.responsible_outputs)* self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(ys=self.loss, xs=local_vars)
                self.var_norms = tf.linalg.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(list(zip(grads, global_vars)))


# keep counters
class Book_Keeper():
    def __init__(self):
        self.lock, self.train_lock = threading.Lock(), threading.Lock()
        self.counters = []
        self.eval_counters = {'all': [], 'fix': [], 'unfix': []}
        self.eval_fixed_progs, self.eval_par_fixed_progs = 0, 0

    def unsafe_reset_eval(self):
        self.eval_counters = {'all': [], 'fix': [], 'unfix': []}
        self.eval_fixed_progs, self.eval_par_fixed_progs = 0, 0

    def update_eval(self, eval_counters, eval_fixed_progs, eval_par_fixed_progs):
        self.lock.acquire()
        for each in eval_counters:
            self.eval_counters[each] += eval_counters[each]
        self.eval_fixed_progs += eval_fixed_progs
        self.eval_par_fixed_progs += eval_par_fixed_progs
        self.lock.release()


    def show_eval_summary(self, global_episodes, which='eval'):

        def show_(tag, err_cnts, episode_lengths, edits, spurious_edits, moves, spurious_moves, fix_cnts, rewards, eval_fixed_progs, eval_par_fixed_progs):
            percent_spurious_edits = 100.0 * np.mean(spurious_edits) / np.mean(
                edits) if np.mean(edits) != 0 and np.mean(edits) != np.nan else 0
            percent_spurious_moves = 100.0 * np.mean(spurious_moves) / np.mean(
                moves) if np.mean(moves) != 0 and np.mean(moves) != np.nan else 0

            print('\n======= SUMMARY_%s (%s) ||' % (which.upper(), tag), 'GE:%d |' % global_episodes, end=' ')
            print('episodes:%-d |' % len(episode_lengths), 'ep_lens:%-4.1f |' % np.mean(episode_lengths), end=' ')
            print('edits:%-4.1f (spuriousE:%-4.1f, %5.1f' % (np.mean(edits), np.mean(spurious_edits), percent_spurious_edits) + '%) |', end=' ')
            print('moves:%-4.1f (spuriousM:%-4.1f, %5.1f' % (np.mean(moves), np.mean(spurious_moves), percent_spurious_moves) + '%) |', end=' ')
            print('ep_rewards:%-5.2f |' % np.mean(rewards), end=' ')
            print('Errs:%-2d |' % np.sum(err_cnts), 'Fixes:%-2d' % np.sum(fix_cnts), end=' ')
            print('(%-5.1f' % (100.0 * np.sum(fix_cnts) / np.sum(err_cnts)) + '%) |', end=' ')
            if tag == 'all':
                print('F_progs:%d |' % eval_fixed_progs, 'PF_progs:%d' % eval_par_fixed_progs)
            else:
                print()

        eval_counters = {}

        self.lock.acquire()
        for each in self.eval_counters:
            eval_counters[each] = np.array(self.eval_counters[each])
        eval_fixed_progs = self.eval_fixed_progs
        eval_par_fixed_progs = self.eval_par_fixed_progs
        self.unsafe_reset_eval()
        self.lock.release()

        for each in eval_counters:
            if len(eval_counters[each]) > 0:
                err_cnts, episode_lengths, edits, spurious_edits, moves, spurious_moves, fix_cnts, rewards = [
                    eval_counters[each][:, i] for i in range(len(eval_counters[each][0]))]
                show_(each, err_cnts, episode_lengths, edits, spurious_edits, moves,
                      spurious_moves, fix_cnts, rewards, eval_fixed_progs, eval_par_fixed_progs)
        print('\n')

    def unsafe_reset(self):
        self.counters = []

    def update(self, counters):
        self.train_lock.acquire()
        self.counters += counters
        self.train_lock.release()

    def show_train_summary(self, global_episodes):
        self.train_lock.acquire()
        counters = np.array(self.counters)
        self.unsafe_reset()
        self.train_lock.release()

        assert len(counters) > 0

        err_cnts, episode_lengths, edits, spurious_edits, moves, spurious_moves, fix_cnts, rewards, values = [counters[:, i] for i in range(len(counters[0]))]
        last = 0

        percent_spurious_edits = 100.0 * np.mean(spurious_edits[last:]) / np.mean(
            edits[last:]) if np.mean(edits[last:]) != 0 and np.mean(edits[last:]) != np.nan else 0
        percent_spurious_moves = 100.0 * np.mean(spurious_moves[last:]) / np.mean(
            moves[last:]) if np.mean(moves[last:]) != 0 and np.mean(moves[last:]) != np.nan else 0

        print('\n======= SUMMARY_TRAIN ||', 'GE:%d |' % global_episodes, end=' ')
        if last != -1:
            print('episodes:%-d |' % len(rewards[last:]), end=' ')
        print('ep_lens:%-4.1f |' % np.mean(episode_lengths[last:]), end=' ')
        print('edits:%-4.1f (spuriousE:%-4.1f, %5.1f' % (np.mean(edits[last:]), np.mean(spurious_edits[last:]), percent_spurious_edits) + '%) |', end=' ')
        print('moves:%-4.1f (spuriousM:%-4.1f, %5.1f' % (np.mean(moves[last:]), np.mean(spurious_moves[last:]), percent_spurious_moves) + '%) |', end=' ')
        print('ep_rewards:%-5.2f |' % np.mean(rewards[last:]), 'mean_Q:%-6.2f |' % np.mean(values[last:]), end=' ')
        print('Errs:%-2d |' % np.sum(err_cnts[last:]), 'Fixes:%-2d' % np.sum(fix_cnts[last:]), end=' ')
        print('(%-5.1f' % (100.0 * np.sum(fix_cnts[last:]) / np.sum(err_cnts[last:])) + '%) |', end=' ')

        fix_percent = 100.0 * np.sum(fix_cnts[last:]) / np.sum(err_cnts[last:])
        return fix_percent



# Worker Agent
class Worker():
    T = 0
    global_episodes = 0
    max_global_episodes = 0

    def __init__(self, env, args, name, vocab_size, trainer, model_path, global_episodes, T, book_keeper, seed):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.dropout = args.dropout
        self.batch_size = args.batch_size
        self.book_keeper = book_keeper
        self.args = args
        self.env = env
        self.embeddings = []
        self.t = 0
        self.rng = np.random.RandomState(seed)
        if self.name == "worker_0":
            Worker.global_episodes = global_episodes
            Worker.T = T
            Worker.max_global_episodes = self.args.epochs * \
                self.env.train_data_size if self.args.evaluate_at == 0 else 1
            
            if self.args.GE_ratio > 0 and self.args.GE_selection_probability > 0:
                Worker.GE_selection_probability = self.args.GE_selection_probability
            else:
                Worker.GE_selection_probability = 0

            print('GE_selection_probability:', Worker.GE_selection_probability)

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(args, vocab_size, env.num_actions, scope=self.name, trainer=self.trainer, seed=seed*2+1)
        self.update_local_ops = update_target_graph('global', self.name)

        self.counters = []
        self.eval_counters = {'all': [], 'fix': [], 'unfix': []}
        self.eval_fixed_progs, self.eval_par_fixed_progs = 0, 0

        self.last_eval_at = Worker.global_episodes
        self.last_checkpointed_at = time()
        self.last_summary_at = time()
        self.last_epoch_time = time()

        if self.name == 'worker_0' and self.args.evaluate_at == 0:
            print('will train for %d max_episodes' % Worker.max_global_episodes)

    def update_eval_books(self):
        self.book_keeper.update_eval(self.eval_counters, self.eval_fixed_progs, self.eval_par_fixed_progs)
        self.reset_eval()

    def update_books(self):
        self.book_keeper.update(self.counters)
        self.reset()

    def reset_eval(self):
        self.eval_fixed_progs, self.eval_par_fixed_progs = 0, 0
        self.eval_counters = {'all': [], 'fix': [], 'unfix': []}

    def reset(self):
        self.counters = []

    def is_it_time_for(self, freq, which='eval'):
        if which == 'eval' and (Worker.global_episodes - self.last_eval_at) > (freq * self.env.train_data_size):
            self.last_eval_at = Worker.global_episodes
            self.last_checkpointed_at = time()
            return True
        elif which == 'ckpt' and (time() - self.last_checkpointed_at) > (freq * 60):
            self.last_checkpointed_at = time()
            return True
        elif which == 'summary' and (time() - self.last_summary_at) > (freq * 60):
            self.last_summary_at = time()
            return True
        else:
            return False

    def show_eval_summary(self, which='eval'):
        assert self.name == 'worker_0', 'name:{}'.format(self.name)
        last = -1

        eval_counters = np.array(self.eval_counters['all'])
        err_cnts, episode_lengths, edits, spurious_edits, moves, spurious_moves, fix_cnts, rewards = [eval_counters[:, i] for i in range(len(eval_counters[0]))]

        percent_spurious_edits = 100.0 * np.mean(spurious_edits[last:]) / np.mean(
            edits[last:]) if np.mean(edits[last:]) != 0 and np.mean(edits[last:]) != np.nan else 0
        percent_spurious_moves = 100.0 * np.mean(spurious_moves[last:]) / np.mean(
            moves[last:]) if np.mean(moves[last:]) != 0 and np.mean(moves[last:]) != np.nan else 0

        print('%s ||' % which.upper(), 'T:%d |' % Worker.T, end=' ')
        print('ep_lens:%-4.1f |' % np.mean(episode_lengths[last:]), end=' ')
        print('edits:%-4.1f (spuriousE:%-4.1f, %5.1f' % (np.mean(edits[last:]), np.mean(spurious_edits[last:]), percent_spurious_edits) + '%) |', end=' ')
        print('moves:%-4.1f (spuriousM:%-4.1f, %5.1f' % (np.mean(moves[last:]), np.mean(spurious_moves[last:]), percent_spurious_moves) + '%) |', end=' ')
        print('ep_rewards:%-5.2f |' % np.mean(rewards[last:]), end=' ')
        print('Errs:%-2d |' % np.sum(err_cnts[last:]), 'Fixes:%-2d' % np.sum(fix_cnts[last:]), end=' ')
        print('(%-5.1f' % (100.0 * np.sum(fix_cnts[last:]) / np.sum(err_cnts[last:])) + '%) |', end=' ')
        print('F_progs:%d |' % self.eval_fixed_progs, 'PF_progs:%d' % self.eval_par_fixed_progs)

    def show_final_eval_summary(self, sess, which):
        self.book_keeper.show_eval_summary(Worker.global_episodes, which)

    def show_final_train_summary(self, sess):
        self.book_keeper.show_train_summary(Worker.global_episodes)

    def show_train_summary(self, counters_for_guided_action=None):
        assert self.name == 'worker_0', 'name:{}'.format(self.name)
        last = -1

        counters = np.array(self.counters) if counters_for_guided_action is None else counters_for_guided_action
        err_cnts, episode_lengths, edits, spurious_edits, moves, spurious_moves, fix_cnts, rewards, values = [
            counters[:, i] for i in range(len(counters[0]))]

        percent_spurious_edits = 100.0 * np.mean(spurious_edits[last:]) / np.mean(
            edits[last:]) if np.mean(edits[last:]) != 0 and np.mean(edits[last:]) != np.nan else 0
        percent_spurious_moves = 100.0 * np.mean(spurious_moves[last:]) / np.mean(
            moves[last:]) if np.mean(moves[last:]) != 0 and np.mean(moves[last:]) != np.nan else 0

        print('TRAIN ||', 'T:%d |' % Worker.T, 'GE:%d |' % Worker.global_episodes, 'epochs:%-4.1f |' % (Worker.global_episodes * 1.0 / self.env.train_data_size), end=' ')
        print('ep_lens:%-4.1f |' % np.mean(episode_lengths[last:]), end=' ')
        print('edits:%-4.1f (spuriousE:%-4.1f, %5.1f' % (np.mean(edits[last:]), np.mean(spurious_edits[last:]), percent_spurious_edits) + '%) |', end=' ')
        print('moves:%-4.1f (spuriousM:%-4.1f, %5.1f' % (np.mean(moves[last:]), np.mean(spurious_moves[last:]), percent_spurious_moves) + '%) |', end=' ')
        print('ep_rewards:%-5.2f |' % np.mean(rewards[last:]), 'mean_Q:%-6.2f |' % np.mean(values[last:]), end=' ')
        print('Errs:%-2d |' % np.sum(err_cnts[last:]), 'Fixes:%-2d' % np.sum(fix_cnts[last:]), end=' ')
        print('(%-5.1f' % (100.0 * np.sum(fix_cnts[last:]) / np.sum(err_cnts[last:])) + '%) |', end=' ')
        # expert demonstration / Teacher forcing
        print('TF:%-3s' % ('No' if counters_for_guided_action is None else 'Yes'))


    def train(self, rollout, sess, gamma, bootstrap_value):
        env = self.env
        programs, actions, updated_programs, rewards, values = rollout.get()
        normalized_ids_programs = []
        for program in programs:
            normalized_ids_programs.append(env.normalize_ids(program))
            env.assert_cursor(normalized_ids_programs[-1])

        prog_batch, prog_batch_len = prepare_batch(normalized_ids_programs)

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        feed_dict = make_feed_dict(self.local_AC, prog_batch, prog_batch_len, self.dropout, actions=actions,
                                   target_v=discounted_rewards, advantages=advantages, Train=True)

        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads
                                            ], feed_dict=feed_dict)

        return v_l / rollout.size, p_l / rollout.size, e_l / rollout.size, g_n, v_n


    def evaluate(self, max_episode_length, sess, coord, which, start, end):
        env = self.env
        args = self.args
        use_compiler = True

        print("Starting evaluator thread %-2d with a batch of %d (%d-%d) episodes" % (self.number, end - start, start, end))
        reject_edits = False if args.do_not_reject_edits_during_eval else None

        with sess.as_default(), sess.graph.as_default():
            with coord.stop_on_exception():
                while not coord.should_stop() and start < end:
                    try:
                        program_id, program, original_program, org_err_cnt, \
                                suggested_actions = env.new_indexed_episode(start, \
                                use_compiler=use_compiler, which=which)

                        name_dict, name_seq = env.get_identifiers(program_id)

                        assert suggested_actions is None, 'dataset:{}, type(suggested_actions): {}'.format(which, type(suggested_actions))
                    except ValueError:
                        raise
                    finally:
                        start += 1

                    sess.run(self.update_local_ops)

                    episode_reward, episode_step_count = 0, 0
                    edit_actions, move_actions = 0, 0
                    spurious_edits, spurious_moves = 0, 0
                    actions_rejected_for_this_state = 0
                    halt = False
                    step = 0
                    output = ''

                    this_err_cnts_min = this_err_cnts = org_err_cnt

                    if self.name == "worker_0" and args.verbose:
                        current_output = ''
                        if not use_compiler:
                            current_output += join_str(
                             'original_program'.upper(), 'ID:', program_id)
                        
                            current_output += join_str(env.show(
                             program_id, original_program, name_dict, name_seq, original_program, use_compiler=False), '\n')
                        current_output += join_str('initial_program'.upper(),
                                                   'ID:', program_id, 'STEP:', step)
                        current_output += join_str(env.show(program_id, program, name_dict, name_seq, original_program, 
                                                    use_compiler=use_compiler), '\n')

                        if args.show_failures:
                            output += current_output
                        else:
                            print(current_output)
                            sleep(args.verbose_sleep_time)


                    while episode_step_count < max_episode_length:
                        step += 1
                        normalized_ids_program = env.normalize_ids(program)
                        env.assert_cursor(normalized_ids_program)
                        prog_batch, prog_batch_len = prepare_batch([normalized_ids_program])
                        feed_dict = make_feed_dict(self.local_AC, prog_batch, prog_batch_len, self.dropout)
                        action_dist, state_value = sess.run([self.local_AC.policy, self.local_AC.value], feed_dict=feed_dict)
                        state_value = state_value[0][0]

                        action_dist = np.squeeze(action_dist)
                        action_indices_from_higher_to_lower_prob = action_dist.argsort()[::-1]
                        if actions_rejected_for_this_state < len(action_dist):
                            action = action_indices_from_higher_to_lower_prob[
                                actions_rejected_for_this_state]

                        else:
                            halt = True
                            break

                        action_name = env.actions[action]
                        updated_program, reward, halt, updated_err_cnts = env.update(program_id, program, name_dict, 
                                                                                     name_seq, action, original_program,
                                                                                     use_compiler, reject_edits)
                        this_err_cnts = this_err_cnts if updated_err_cnts is None else updated_err_cnts
                        this_err_cnts_min = min(this_err_cnts_min, this_err_cnts)

                        edit_actions += (1 if 'move' not in action_name else 0)
                        move_actions += (1 if 'move' in action_name else 0)
                        if updated_program == program:
                            if 'move' in action_name:
                                spurious_moves += 1
                            else:
                                spurious_edits += 1
                            actions_rejected_for_this_state += 1
                        else:
                            actions_rejected_for_this_state = 0

                        if self.name == "worker_0" and args.verbose:
                            current_output = join_str('updated_program'.upper(), 'STEP:', step, ', ACTION:', action_name,
                                                      'REWARD:', reward, 'ID:', program_id, 'spurious_edits:', spurious_edits,
                                                      'spurious_moves:', spurious_moves, 'state_value:', state_value)
                            current_output += join_str(env.show(program_id, updated_program, name_dict, name_seq, \
                                                                original_program, use_compiler=use_compiler), '\n')

                            if args.show_failures:
                                output += current_output
                            else:
                                print(current_output)
                                sleep(args.verbose_sleep_time)

                        episode_reward += reward
                        program = copy.deepcopy(updated_program)
                        episode_step_count += 1
                        if halt:
                            break


                    this_fix_cnt = org_err_cnt - this_err_cnts_min
                    assert this_fix_cnt >= 0 and this_fix_cnt <= org_err_cnt

                    values_to_log = [org_err_cnt, episode_step_count, edit_actions,
                                     spurious_edits, move_actions, spurious_moves, this_fix_cnt, episode_reward]

                    if this_fix_cnt == org_err_cnt:
                        self.eval_counters['fix'].append(values_to_log)
                    else:
                        self.eval_counters['unfix'].append(values_to_log)

                    self.eval_counters['all'].append(values_to_log)

                    self.eval_fixed_progs += 1 if this_fix_cnt == org_err_cnt else 0
                    self.eval_par_fixed_progs += 1 if (this_fix_cnt < org_err_cnt and this_fix_cnt > 0) else 0

                    if self.name == 'worker_0':
                        if args.show_failures and this_fix_cnt != org_err_cnt:
                            print(output)
                            output = ''
                        self.show_eval_summary(which)
                        if args.verbose:
                            sleep(args.verbose_sleep_time)

                self.update_eval_books()

    def save_model_and_start_eval(self, sess, saver):
        save_model_at = str(Worker.T)
        saver.save(sess, os.path.join(self.model_path, 'model-' + save_model_at))
        saver2.save(sess, os.path.join(self.model_path,
                                       'epoch-ckpts', 'model-' + save_model_at))
        print("Saved Model:", save_model_at)
        self.show_final_train_summary(sess)
        done()
        time_taken = time() - self.last_epoch_time
        self.last_epoch_time = time()
        print('time_taken:', time_taken)
        do_eval(sess, self.env, self.args, 'test')

    def work(self, max_episode_length, gamma, sess, coord, saver):
        env = self.env
        use_compiler = self.args.use_compiler

        print("Starting worker: %d, T: %d, GE: %f, labelled_data: %d" % (self.number, Worker.T, Worker.global_episodes, env.guided_train_data_size))

        with sess.as_default(), sess.graph.as_default():
            with coord.stop_on_exception():
                while not coord.should_stop() and Worker.global_episodes <= Worker.max_global_episodes:
                    sess.run(self.update_local_ops)

                    episode_buffer = experience()
                    episode_values = []
                    episode_reward = 0
                    episode_step_count = 0
                    halt = False
                    edit_actions, move_actions = 0, 0
                    spurious_edits, spurious_moves = 0, 0

                    program_id, program, original_program, org_err_cnt, suggested_actions = env.new_random_episode(
                        use_compiler, 'train', self.GE_selection_probability)

                    name_dict, name_seq = env.get_identifiers(program_id)

                    if args.verbose:
                        print('original_program'.upper(), 'STEP:', len(episode_values), 'ID:', program_id)
                        print(env.show(program_id, original_program, name_dict, name_seq, original_program, use_compiler), '\n')
                        print('initial_program'.upper(), 'STEP:', len(episode_values), 'ID:', program_id)
                        print(env.show(program_id, program, name_dict, name_seq, original_program, use_compiler), '\n')
                        if suggested_actions is not None:
                            print('suggested_actions:', [env.actions[x] for x in suggested_actions])
                        sleep(args.verbose_sleep_time)

                    this_err_cnts_min = this_err_cnts = org_err_cnt

                    while episode_step_count < max_episode_length:

                        normalized_ids_program = env.normalize_ids(program)
                        env.assert_cursor(normalized_ids_program)
                        prog_batch, prog_batch_len = prepare_batch([normalized_ids_program])
                        feed_dict = make_feed_dict(self.local_AC, prog_batch, prog_batch_len, self.dropout)

                        # Take an action using probabilities from policy network output.
                        action_dist, v = sess.run([self.local_AC.policy, self.local_AC.value], feed_dict=feed_dict)

                        # Guided exploration
                        if suggested_actions is not None and len(suggested_actions) > 0:
                            action = suggested_actions.pop(0)
                        else:
                            action = self.rng.choice(action_dist[0], p=action_dist[0])
                            action = np.argmax(action_dist == action)

                        action_name = env.actions[action]

                        updated_program, reward, halt, updated_err_cnts = env.update(program_id, program, name_dict, \
                                                                            name_seq, action, original_program, use_compiler)
                        this_err_cnts = this_err_cnts if updated_err_cnts is None else updated_err_cnts
                        this_err_cnts_min = min(this_err_cnts_min, this_err_cnts)

                        episode_step_count += 1
                        edit_actions += (1 if 'move' not in action_name else 0)
                        move_actions += (1 if 'move' in action_name else 0)
                        if updated_program == program:
                            if 'move' in action_name:
                                spurious_moves += 1
                            else:
                                spurious_edits += 1
                        episode_values.append(v[0, 0])
                        episode_reward += reward

                        self.t += 1
                        Worker.T += 1

                        episode_buffer.add((program, action, updated_program, reward, v[0, 0]))
                        program = copy.deepcopy(updated_program)

                        if args.verbose:
                            print('TF:Yes,' if suggested_actions is not None else 'TF:No ,', end=' ')
                            print('updated_program'.upper(), 'STEP:', len(episode_values), ', ACTION:', action_name, end=' ')
                            print('REWARD:', reward, 'ID:', program_id)
                            print(env.show(program_id, updated_program, name_dict, name_seq, original_program, use_compiler), '\n')
                            sleep(args.verbose_sleep_time)

                        if episode_buffer.size == self.batch_size and halt != True and episode_step_count != max_episode_length - 1:
                            feed_dict = make_feed_dict(self.local_AC, prog_batch, prog_batch_len, self.dropout, Train=True)
                            v1 = sess.run(self.local_AC.value, feed_dict=feed_dict)[0, 0]

                            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)

                            episode_buffer.reset()
                            sess.run(self.update_local_ops)

                        if halt:
                            break

                    this_fix_cnt = org_err_cnt - this_err_cnts_min
                    assert this_fix_cnt >= 0 and this_fix_cnt <= org_err_cnt

                    episode_mean_Q = np.mean(episode_values) if len(episode_values) > 0 else 0

                    values_to_log = [org_err_cnt, episode_step_count, edit_actions, spurious_edits, move_actions, spurious_moves, this_fix_cnt,
                                     episode_reward, episode_mean_Q]

                    if suggested_actions is None:
                        self.counters.append(values_to_log)

                    Worker.global_episodes += 1

                    # Update the network using the experience buffer at the end of the episode.
                    if episode_buffer.size != 0:
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                    if self.name == 'worker_0':
                        if suggested_actions is None:
                            self.show_train_summary()
                        else:
                            temp_counters = np.array([values_to_log])
                            self.show_train_summary(temp_counters)
                        if args.verbose:
                            sleep(args.verbose_sleep_time)

                    if self.t % self.args.workers == 0:
                        self.update_books()

                    if self.name == 'worker_0' and self.is_it_time_for(args.checkpoint_interval, which='ckpt'):
                        save_model_at = str(Worker.T)
                        saver.save(sess, os.path.join(
                            self.model_path, 'model-' + save_model_at))
                        print("Saved Model:", save_model_at)

                    if self.name == 'worker_0' and self.is_it_time_for(args.evaluation_frequency, which='eval'):
                        self.save_model_and_start_eval(sess, saver)

                self.update_books()


def get_env(dataset, step_penalty, args, action_names, verbose, seed, train_data_size=None, top_down_movement=True,
            single_delete=True, GE_code_ids=None):
    global compilation_error_db
    reject_spurious_edits = not args.do_not_reject_edits
    GE_ratio = args.GE_ratio if GE_code_ids is None else None
    test_program = None
    if args.evaluate_single_program:
        with open(args.evaluate_single_program, 'r') as f:
            test_program = f.read()

    return Environment(dataset, step_penalty=step_penalty, seed=seed, actions=action_names,
                     top_down_movement=top_down_movement, reject_spurious_edits=reject_spurious_edits,
                     compilation_error_store=compilation_error_db, train_data_size=train_data_size,
                     valid_data_size=0, test_data_size=args.eval_size, single_delete=single_delete, verbose=verbose,
                     GE_ratio=GE_ratio, GE_code_ids=GE_code_ids, single_program=test_program,
                     sparse_rewards=args.sparse_rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a staged fault-localization and repair seq2seq RNN.')

    parser.add_argument('data_directory', help='Data directory')
    parser.add_argument('checkpoints_directory', help='Checkpoints directory')
    parser.add_argument("--seed", type=int, help="random seed for reproducible results", default=1189)

    parser.add_argument('-b', "--batch_size", type=int,help="batch size", default=24)
    parser.add_argument('-ed', "--embedding_dim", type=int, help="embedding_dim", default=24)
    parser.add_argument('-hd', "--hidden_dim", type=int, help="memory_dim", default=128)
    parser.add_argument('-n', "--num_layers", type=int, help="num_layers", default=2)
    parser.add_argument('-d', '--dropout', help='Probability to use for dropout', type=float, default=0)
    parser.add_argument('-r', "--resume", 
                        help="resume training by loading previous checkpoint", action="store_true")
    parser.add_argument('-g', "--global_episodes", type=int,
                        help="resuming at this global episode count", default=0)

    parser.add_argument('-sr', '--sparse_rewards',
                        help='use sparse rewards', action="store_true")
    parser.add_argument('-y', "--discount_factor", type=float,
                        help="Discount factor on the target Q-values", default=0.99)
    parser.add_argument('-e', "--max_epLength", type=int,
                        help="The max allowed length of each episode.", default=100)
    parser.add_argument('-w', "--workers", type=int,
                        help="Number of parallel workers.", default=32)
    parser.add_argument('-sc', "--spare_cores", type=int,
                        help="Spare these many cpu cores.", default=-1)
    parser.add_argument('-ge', "--GE_ratio", type=float,
                        help="use this fraction of dataset for guided training", default=0.1)
    parser.add_argument('-lr', "--learning_rate",
                        type=float, help="learning_rate", default=0.0001)

    parser.add_argument('-tds', "--train_data_size", type=int,
                        help="Use only this many programs for training.", default=0)
    parser.add_argument("-ep", "--epochs", type=int,
                        help="run training for this many epochs", default=10)
    parser.add_argument('-c', "--checkpoint_interval", type=int,
                        help="Checkpoint after this many minutes", default=90)

    parser.add_argument('-evs', "--eval_size", type=int,
                        help="Use maximum this many programs for evaluation.", default=None)
    parser.add_argument('-evf', "--evaluation_frequency",
                        type=int, help="Eval after this many epochs", default=1)
    parser.add_argument("-v", "--verbose", help="set to verbose", action="store_true")
    parser.add_argument('-vt', "--verbose_sleep_time", type=float,
                        help="sleep for this many seconds after each step", default=1.5)

    parser.add_argument('-uc', "--use_compiler",
                        help="use compiler for training", action="store_true")
    parser.add_argument('-dnr', "--do_not_reject_edits",
                        help="Turn off edit rejection", action="store_true")
    parser.add_argument('-sf', '--show_failures',
                        help='verbose for failure cases', action="store_true")
    parser.add_argument('-eval', "--evaluate_at", type=int,
                        help="Evaluate the specified saved model", default=0)
    parser.add_argument("-wh", "--eval_which", help="evaluate saved model",
                        choices=['real', 'seeded', 'test'], default='real')
    parser.add_argument('-dnr_eval', "--do_not_reject_edits_during_eval",
                        help="Turn off edit rejection during evaluation", action="store_true")
    parser.add_argument('-esp', '--evaluate_single_program',
                        help='provide path to the file containing a program to fix', default=None)

        
    parser.add_argument('-gesp', "--GE_selection_probability", type=float,
                        help="use guided exploration with this probability", default=1)

    args = parser.parse_args()

    assert 'network_inputs' not in args.checkpoints_directory, 'move your checkpoints directory to the designated folder!'
    assert args.data_directory != args.checkpoints_directory, 'data and checkpoints directories should be different!'

    action_names = None
    if args.GE_ratio <= 0:
        args.GE_selection_probability, args.GE_till_epoch = 0, 0

    gamma = args.discount_factor

    max_episode_length = args.max_epLength
    compilation_error_db = Compilation_error_db() if args.use_compiler else None

    def rename_ckpt_dir(ckpt_dir):
        ckpt_dir_name_list = ckpt_dir.split('/')
        new_list = []
        for each in ckpt_dir_name_list:
            if 'longshot' in each or 'RLA' in each or 'rla' in each or 'final' in each:
                suffix = ''
                if args.do_not_reject_edits:
                    suffix += '-dnr' if '-dnr' not in each else ''
                if args.GE_ratio >= 0:
                    ge_str = '-ge_%s' % str(args.GE_ratio).replace('.', '')
                    suffix += ge_str if ge_str not in each else ''
                if args.sparse_rewards:
                    suffix += '-sr' if '-sr' not in each else ''
                new_list += [each + suffix]
            else:
                new_list += [each]
        return '/'.join(new_list)

    args.checkpoints_directory = rename_ckpt_dir(args.checkpoints_directory)
    ckpt_dir = args.checkpoints_directory 

    if args.evaluate_at > 0:
        args.GE_ratio = 0
        args.GE_selection_probability = 0
        if args.eval_size is None: args.eval_size = 0
    else:
        if args.eval_size is None: args.eval_size = 500

    if args.evaluate_single_program:
        args.workers = 1
        args.eval_which = 'single'
        args.train_data_size = 1
        args.verbose = True
        args.verbose_sleep_time = 0

    dataset = load_data(args.data_directory, load_real_test_data=(args.evaluate_at > 0 and args.eval_which == 'real'),
                        load_seeded_test_data=(args.evaluate_at > 0 and args.eval_which == 'seeded'),
                        load_only_dicts=(args.evaluate_single_program is not None),
                        allow_pickle=True)
    args.train_data_size = args.train_data_size
    train_data_size = dataset.data_size[0] if args.train_data_size == 0 else args.train_data_size
    args.train_data_size = train_data_size

    temp_env = get_env(dataset, -1.0 / args.max_epLength, args, action_names, False, args.seed, train_data_size)
    guided_exploration_code_ids = temp_env.code_ids['GE_train'] if args.evaluate_at == 0 else []
    vocab_size = len(temp_env.normalized_ids_tl_dict)
    num_actions = len(temp_env.actions)

    dataset_name = '_'.join(args.checkpoints_directory.split('/')[2:]) if 'bin_' in args.checkpoints_directory \
        else '_'.join(args.checkpoints_directory.split('/')[1:])
    dataset_name = dataset_name[:-1]

    log = logger(dataset_name + '-' + args.eval_which +
                 ('' if args.evaluate_at == 0 else '-eval_at-%d' % args.evaluate_at) +
                 ('-single' if args.evaluate_single_program else '') +
                 ('-dnr' if args.do_not_reject_edits_during_eval else ''))

    print('\nlogging into {}'.format(log.log_file))
    sys.stdout = log

    print('COMMAND:', ' '.join(sys.argv))
    print('normalized vocab size:', vocab_size, 'num_actions:', num_actions)

    nproc = psutil.cpu_count()
    if args.spare_cores == -1:
        spare_cores = 4 if nproc >= 32 else 1
    else:
        spare_cores = min(nproc-1, max(0, args.spare_cores))
    p = psutil.Process()
    p.cpu_affinity(list(range(nproc-spare_cores)))
    print('Left %d spare cores, running training only on the following cores:' % spare_cores, p.cpu_affinity(), '\n')

    if not args.evaluate_single_program:
        print('\nOriginal-Dataset-size- | TRAIN:%d | VALID:%d | TEST:%d' % (dataset.data_size))
        print('Controlled-Dataset-size- | TRAIN:%d | REAL-TEST:%d | TEST:%d | GE_TRAIN:%d\n' % (temp_env.train_data_size,
                                                                                                temp_env.real_test_data_size,
                                                                                                temp_env.test_data_size,
                                                                                                temp_env.guided_train_data_size))
    print('ckpt_dir:', ckpt_dir)
    make_dir_if_not_exists(ckpt_dir)
    make_dir_if_not_exists(os.path.join(ckpt_dir, 'epoch-ckpts'))

    configuration = {}
    configuration["args"] = args
    configuration["log"] = log.log_file
    np.save(os.path.join(ckpt_dir, 'experiment-configuration-%s-%s.npy' % (strftime("%d_%m"), strftime("%H_%M"))), configuration)

    def do_eval(sess, env, args, which='test'):
        try:
            eval_coord = tf.train.Coordinator()

            eval_data_size = min(args.eval_size, env.data_sizes[which]) \
                if args.eval_size > 0 else env.data_sizes[which]
            if eval_data_size == 0:
                raise ValueError('%s dataset has zero programs for evaluation, skipping!' % which)
            num_eval_workers = min(len(workers), max(int(math.ceil(eval_data_size / 50.0)), 1))
            eval_batch_size = int(eval_data_size / num_eval_workers)

            print('%s eval_data_size:%d, num_eval_workers:%d, eval_batch_size:%d' % (which, eval_data_size, num_eval_workers, eval_batch_size))

            worker_threads = []
            for i, worker in enumerate(workers):
                if i == num_eval_workers:
                    break
                start = i * eval_batch_size
                end = (i + 1) * eval_batch_size if i < num_eval_workers-1 else eval_data_size
                def worker_work(): return worker.evaluate(args.max_epLength, sess, eval_coord, which, start, end)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            eval_coord.join(worker_threads)
        except KeyboardInterrupt:
            print('\nKeyboard interrupt: stopping evaluation!')
        finally:
            workers[0].show_final_eval_summary(sess, which)


    if args.resume:
        best_ckpt = get_best_checkpoint(ckpt_dir)
        print('Loaded model:', best_ckpt)
    else:
        best_ckpt = 0

    # Training
    with tf.device("/cpu:0"):
        global_episodes = args.global_episodes
        assert not args.resume or global_episodes > 0
        trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)

        book_keeper = Book_Keeper()
        master_network = AC_Network(args, vocab_size, num_actions, scope='global', trainer=None, seed=args.seed*2)

        # Set workers ot number of available CPU threads
        num_workers = multiprocessing.cpu_count() if args.workers == 0 else args.workers
        print('num_workers:', num_workers)

        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(get_env(dataset, -1.0 / args.max_epLength, args, action_names, i == 0, args.seed + i + 1, train_data_size,
                                          GE_code_ids=guided_exploration_code_ids), args, i, vocab_size, trainer, ckpt_dir,
                                  global_episodes, best_ckpt, book_keeper, args.seed + i + 100))

        params_to_save = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='global')
        saver = tf.compat.v1.train.Saver(params_to_save, max_to_keep=10)
        saver2 = tf.compat.v1.train.Saver(params_to_save, max_to_keep=10)

    with tf.compat.v1.Session() as sess:
        if args.evaluate_at == 0:
            try:
                coord = tf.train.Coordinator()
                sess.run(tf.compat.v1.global_variables_initializer())
                if args.resume:
                    print('restoring at:', best_ckpt)
                    saver.restore(sess, os.path.join(ckpt_dir, 'model-%d' % best_ckpt))

            # Start the "work" process for each worker in a separate thread.
                worker_threads = []
                for worker in workers:
                    def worker_work(): return worker.work(max_episode_length, gamma, sess, coord, saver)
                    t = threading.Thread(target=(worker_work))
                    t.start()
                    sleep(0.5)
                    worker_threads.append(t)
                coord.join(worker_threads)
                print('\n\nDone training! Initializing last evaluation - \n')
                workers[0].save_model_and_start_eval(sess, saver)
            except KeyboardInterrupt:
                if compilation_error_db is not None:
                    compilation_error_db.close()
                print('\nKeyboard interrupt: stopping training!')

        else:           # just evalaute the given model
            print('restoring at:', args.evaluate_at)
            try:
                saver.restore(sess, os.path.join(ckpt_dir, 'model-%d' % args.evaluate_at))
            except:
                saver.restore(sess, os.path.join(ckpt_dir, 'epoch-ckpts', 'model-%d' % args.evaluate_at))
            if args.do_not_reject_edits_during_eval:
                print('----- evaluating WITHOUT edit rejection ------')
            else:
                print('----- evaluating WITH edit rejection ------')

            do_eval(sess, temp_env, args, which=args.eval_which)

            if compilation_error_db is not None:
                compilation_error_db.close()