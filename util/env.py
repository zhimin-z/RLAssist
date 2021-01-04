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

from util.helpers import get_rev_dict, split_list, tokens_to_source, make_dir_if_not_exists, prepend_line_numbers, prepare_batch, compilation_errors, coin_flip
from util.c_tokenizer import C_Tokenizer
import numpy as np
import os
import sys
import copy
from functools import partial
from data_processing.typo_mutator import Typo_Mutate
from collections import namedtuple


def load_dictionaries(destination, allow_pickle, name='all_dicts.npy'):
    tl_dict, rev_tl_dict = np.load(os.path.join(destination, name), allow_pickle=allow_pickle)
    return tl_dict, rev_tl_dict


def save_dictionaries(destination, all_dicts):
    np.save(os.path.join(destination, 'all_dicts.npy'), all_dicts)


class Env_engine:

    def get_normalized_ids_dict(self, tl_dict):
        normalized_ids_tl_dict = {}
        normalized_ids_tl_dict['-new-line-'] = self.new_line    # 2
        normalized_ids_tl_dict['_pad_'] = self.pad              # 0
        normalized_ids_tl_dict['_eos_'] = self.cursor           # 1
        id_token_vecs = []
        for key, value in sorted(tl_dict.items()):
            if '_<id>_' not in key:
                if key not in normalized_ids_tl_dict:
                    normalized_ids_tl_dict[key] = len(normalized_ids_tl_dict)
            else:
                id_token_vecs.append(value)
        normalized_ids_tl_dict['_<id>_@'] = len(normalized_ids_tl_dict)
        assert len(normalized_ids_tl_dict) + \
            len(id_token_vecs) - 1 == len(tl_dict)
        return normalized_ids_tl_dict, id_token_vecs


    def __init__(self, tl_dict, seed, correct_fix_reward=1.0, step_penalty=-0.01,
                 top_down_movement=False, reject_spurious_edits=False, compilation_error_store=None,
                 single_delete=True, actions=None, sparse_rewards=True):
        self.rng = np.random.RandomState(seed)
        self.action_tokens = list('''{}();,.''')
        self.actions = ([] if top_down_movement else [
                        'move_up', 'move_left']) + ['move_down', 'move_right']
        for each in self.action_tokens:
            self.actions += ['insert' + each] if each != '.' else []
            self.actions += ['delete' + each] if not single_delete else []
        self.actions += ['delete'] if single_delete else []
        self.actions += ['replace;with,', 'replace,with;',
                         'replace.with;', 'replace;)with);']

        self.replacement_action_to_action_sequence_map = {
            'replace;)with);': ['delete;)', 'move_right', 'insert;'],
            'replace;with,': ['delete;', 'insert,'],
            'replace,with;': ['delete,', 'insert;'],
            'replace.with;': ['delete.', 'insert;']
        }
        self.action_sequence_to_replacement_action_map = {}
        for replacement_action, action_seq in list(self.replacement_action_to_action_sequence_map.items()):
            self.action_sequence_to_replacement_action_map[''.join(
                action_seq)] = replacement_action

        if actions is not None:
            raise NotImplementedError()

        self.tl_dict = tl_dict
        self.rev_tl_dict = get_rev_dict(tl_dict)
        self.new_line = tl_dict['-new-line-']
        self.pad = tl_dict['_pad_']
        self.cursor = tl_dict['_eos_']
        self.normalized_ids_tl_dict, self.id_token_vecs = self.get_normalized_ids_dict(tl_dict)
        assert self.cursor == self.normalized_ids_tl_dict['_eos_']
        assert self.new_line == self.normalized_ids_tl_dict['-new-line-']
        assert self.pad == self.normalized_ids_tl_dict['_pad_']

        self.mutables = [self.tl_dict['_<op>_' + each]
                         for each in self.action_tokens]

        self.top_down_movement = top_down_movement
        self.reject_spurious_edits = reject_spurious_edits
        self.compilation_error_store = compilation_error_store
        self.single_delete = single_delete

        self.correct_fix_reward = correct_fix_reward
        if sparse_rewards:
            self.step_penalty = step_penalty
            self.edit_penalty = 0.0
            self.error_resolution_reward = 2 * abs(step_penalty) - (abs(step_penalty)/10)
        else:
            self.step_penalty = step_penalty / 2
            self.edit_penalty = self.step_penalty * 5
            self.error_resolution_reward = 2 * (abs(self.edit_penalty) - abs(self.step_penalty))

    @property
    def num_actions(self):
        return len(self.actions)

    def normalize_ids(self, prog_vector):
        normalized_id = self.normalized_ids_tl_dict['_<id>_@']
        normalized_prog_vector = []
        for token in prog_vector:
            if token in self.id_token_vecs:
                normalized_prog_vector += [normalized_id]
            else:
                normalized_prog_vector += [
                    self.normalized_ids_tl_dict[self.rev_tl_dict[token]]]
        return normalized_prog_vector

    def normalize_id(self, token):
        if token in self.id_token_vecs:
            return self.normalized_ids_tl_dict['_<id>_@']
        return self.normalized_ids_tl_dict[self.rev_tl_dict[token]]

    def get_action(self, action_name):
        assert type(action_name) == str
        for action, name in enumerate(self.actions):
            if action_name == name:
                return action
        raise ValueError('%s is not in the action list' % action_name)

    def get_line_number(self, program, cursor):
        '''returns line_number of cursor'''
        assert len(program) >= cursor
        return program[:cursor].count(self.new_line)

    def get_line(self, program, line_number):
        return split_list(program, self.new_line)[line_number]

    def get_line_count(self, program):
        return program.count(self.new_line) - 1

    def cursor_of_line(self, program, line_number, position='init'):
        program_line_count = self.get_line_count(program)
        assert line_number >= 0
        assert line_number <= program_line_count
        if line_number == 0 and position == 'init':   # this is the first line of the program
            return 0
        if line_number == program_line_count and position == 'end':     # this is the last line of the program
            cursor = len(program) - 1
            assert program[cursor] == self.new_line
            return cursor
        lines = split_list(program, self.new_line)
        current_cursor = 0
        for current_line_number, line in enumerate(lines):
            if current_line_number < line_number:
                current_cursor += len(line)
            elif current_line_number == line_number:
                if position == 'init':
                    return current_cursor
                elif position == 'end':
                    return current_cursor + len(line) - 1
                else:
                    raise ValueError(
                        'position could be either "init" or "end",  not "%s"' % position)


    def act(self, program, cursor, action):
        assert type(action) == str, 'action:{}, type:{}'.format(action, type(action))

        if action == 'move_up':
            current_line_number = self.get_line_number(program, cursor)
            assert current_line_number >= 0
            if current_line_number == 0:
                return program, cursor
            else:
                return program, self.cursor_of_line(program, current_line_number - 1)

        if action == 'move_down':
            current_line_number = self.get_line_number(program, cursor)
            max_line_number = self.get_line_count(program)

            assert current_line_number <= max_line_number

            if current_line_number == max_line_number:
                return program, cursor
            else:
                return program, self.cursor_of_line(program, current_line_number + 1)

        if action == 'move_left':
            current_line_number = self.get_line_number(program, cursor)
            line_init_cursor = self.cursor_of_line(
                program, current_line_number)

            assert line_init_cursor <= cursor, 'line_init_cursor: %d, cursor: %d' % (
                line_init_cursor, cursor)

            if line_init_cursor == cursor:
                return program, cursor
            else:
                return program, cursor - 1

        if action == 'move_right':
            current_line_number = self.get_line_number(program, cursor)
            line_end_cursor = self.cursor_of_line(
                program, current_line_number, 'end')

            assert line_end_cursor >= cursor, 'line_end_cursor: %d, cursor: %d' % (
                line_end_cursor, cursor)

            if line_end_cursor == cursor:
                return program, cursor
            else:
                return program, cursor + 1

        # cursor can be moved upto 1 location after the program (say upto an invisible EOF)
        # insert means insert_before(at)_cursor, delete means delete at cursor. e.g. (abc, b) -insert x-> (axbc, x) | (abc, b) -delete x-> (ac, c)
        # insert before init of a line will be done on the same line
        # delete before init of a line will be performed on the previous line

        if 'insert' in action:
            action_token = '_<op>_' + action[-1]
            to_insert = self.tl_dict[action_token]
            new_program = program[:cursor] + [to_insert] + program[cursor:]
            assert len(new_program) == len(
                program) + 1, 'cursor:{}, len(new_prog):{}, len(prog):{}'.format(cursor, len(new_program), len(program))
            return new_program, cursor

        if 'delete' in action:
            if action == 'delete':  # self.single_delete: # for supporting replacement actions
                if program[cursor] not in self.mutables:
                    return program, cursor
                new_program = program[:cursor] + program[cursor + 1:]
                assert len(new_program) == len(
                    program) - 1, 'cursor:{}, len(new_prog):{}, len(prog):{}'.format(cursor, len(new_program), len(program))
                return new_program, cursor
            else:           # this case is reachable for replacement deletes even with single_delete configuration
                action_token = '_<op>_' + action[-1]
                to_delete = self.tl_dict[action_token]
                if program[cursor] == to_delete:
                    new_program = program[:cursor] + program[cursor + 1:]
                    assert len(new_program) == len(
                        program) - 1, 'cursor:{}, len(new_prog):{}, len(prog):{}'.format(cursor, len(new_program), len(program))
                    return new_program, cursor
                else:
                    return program, cursor

        # delete + insert = replace  BUT   insert + delete != replace   for this implementation!
        if 'replace' in action:
            if 'replace;)with);' in action:
                try:
                    if self.rev_tl_dict[program[cursor]] == '_<op>_;' and self.rev_tl_dict[program[cursor + 1]] == '_<op>_)':
                        program, cursor = self.act(program, cursor, 'delete;')
                        program, cursor = self.act(
                            program, cursor, 'move_right')
                        return self.act(program, cursor, 'insert;')
                except:
                    print("replace);with;) action failed!")
                    print(program, '\n', cursor, '\n')
                    raise
            else:
                if self.rev_tl_dict[program[cursor]] == '_<op>_' + action[7]:
                    program, cursor = self.act(
                        program, cursor, 'delete' + action[7])
                    return self.act(program, cursor, 'insert' + action[-1])
            return program, cursor

    def assert_cursor(self, prog_vector):
        assert prog_vector.count(self.cursor) == 1, 'count:{}, vector:{}'.format(prog_vector.count(self.cursor), ', '.join(prog_vector))

    def gui_set_cursor(self, prog_vector, cursor=None):
        ''' input: the position of the character, the cursor should overlap '''
        if prog_vector.count(self.cursor) > 0:
            prog_vector = copy.deepcopy(prog_vector)
            prog_vector = self.gui_remove_cursor(prog_vector)
        if not cursor:
            if self.top_down_movement:
                cursor = 0
            else:
                cursor = self.rng.choice(list(range(len(prog_vector))))
        position = cursor + 1
        prog_vector.insert(position, self.cursor)
        return prog_vector

    def gui_find_cursor_position(self, prog_vector):
        cursor_position = prog_vector.index(self.cursor)
        assert cursor_position > 0
        cursor = cursor_position - 1
        return cursor

    def gui_remove_cursor(self, prog_vector):
        try:
            cursor_position = prog_vector.index(self.cursor)
        except:
            return prog_vector
        return prog_vector[:cursor_position] + prog_vector[cursor_position + 1:]

    def gui_pop_cursor(self, prog_vector):
        assert prog_vector.count(self.cursor) == 1, 'cursor: {}, prog_vector:\n{}'.format(self.cursor, prog_vector)
        cursor_position = prog_vector.index(self.cursor)
        assert cursor_position > 0
        cursor = cursor_position - 1
        del prog_vector[cursor_position]
        return prog_vector, cursor

    def devectorize(self, program, keep_cursor=False):
        tokens_to_be_removed = [self.pad] + ([] if keep_cursor else [self.cursor])
        devectorized_program = []
        for token in program:
            if token not in tokens_to_be_removed:
                devectorized_program.append(self.rev_tl_dict[token])
        return ' '.join(devectorized_program)

    def vectorize(self, program):
        program_vector = []
        for token in program.split():
            program_vector.append(self.tl_dict[token])
        return program_vector

    def filter_program_vector(self, program_vector):
        tokens_to_be_removed = [self.pad, self.cursor]
        return [token for token in program_vector if token not in tokens_to_be_removed]

    def is_exact_match(self, vector_program1, vector_program2):
        tokens_to_be_removed = [self.new_line, self.pad, self.cursor]
        vector_program1 = [token for token in vector_program1 if token not in tokens_to_be_removed]
        vector_program2 = [token for token in vector_program2 if token not in tokens_to_be_removed]
        return cmp(vector_program1, vector_program2) == 0

    def get_program_source_from_vector(self, program_vector, name_dict, name_seq, keep_cursor=False, clang_format=False, get_tokens=False):
        return tokens_to_source(self.devectorize(program_vector, keep_cursor), name_dict, clang_format, name_seq,
                                                                                            cursor=('_eos_' if keep_cursor else None), 
                                                                                            get_tokens=get_tokens)


    def show(self, program_id, program_vector, name_dict, name_seq, original_program_vector, use_compiler):
        assert type(program_vector) == list
        temp_program_source = self.get_program_source_from_vector(
            program_vector, name_dict, name_seq, keep_cursor=True, clang_format=False, get_tokens=False)
        output = prepend_line_numbers(temp_program_source) + '\n'
        err_list = self.get_compiler_errors(program_id, program_vector, name_dict, name_seq) if use_compiler else self.get_errors_from_ground_truth(
            program_vector, original_program_vector)
        err_list = err_list if use_compiler else [x + 1 for x in err_list]
        output += ('**NO ERRORS**' if len(err_list) == 0 else 'ERRORS:\n' + '\n'.join(map(str, err_list))) + '\n'
        return output

    def calculate_fixes(self, old_err_list, err_list, old_prog_vector=None, prog_vector=None):
        diff = len(old_err_list) - len(err_list)
        return diff

    def format_program_vector(self, program_vector):
        lines = split_list(self.filter_program_vector(
            program_vector), delimiter=self.new_line, keep_delimiter=True)
        output = []
        for i, line in enumerate(lines):
            output += ['[%-2d, %-3d]' % (i, len(line)) + ', '.join(map(str, line))]
        return '\n'.join(output)

    def fix_line(self, mutated_line, original_line, recursive=False):
        assert recursive or mutated_line != original_line
        edit_count = 0
        while mutated_line != original_line:
            action, token = None, None
            for i, (a, b) in enumerate(zip(mutated_line, original_line)):
                if a != b:
                    if a not in self.mutables and b in self.mutables:  # b has been deleted from original line
                        action, token = 'insert', b
                    elif a in self.mutables and b not in self.mutables:  # a is a duplicate token
                        action = 'delete'
                    else:
                        action = 'check_min_of_both'
                    edit_count += 1
                    break
            assert action is not None, '\nmut:{}\norg:{}'.format(
                mutated_line, original_line)
            if action == 'delete':
                del mutated_line[i]
            elif action == 'insert':
                mutated_line.insert(i, token)
            else:
                try:
                    del_cost = self.fix_line(
                        mutated_line[i + 1:], original_line[i:], recursive=True)
                except:
                    print('original_line:', original_line, '\n', 'mutated_line:', mutated_line)
                    raise
                ins_cost = self.fix_line(
                    mutated_line[i:], original_line[i + 1:], recursive=True)
                return edit_count + min(del_cost, ins_cost)
        assert mutated_line == original_line, '{}\n{}'.format(
            mutated_line, original_line)
        return edit_count

    def get_errors_from_ground_truth(self, program_vector, original_program_vector):
        programA_lines = split_list(self.filter_program_vector(
            program_vector), delimiter=self.new_line, keep_delimiter=True)
        programB_lines = split_list(self.filter_program_vector(
            original_program_vector), delimiter=self.new_line, keep_delimiter=True)
        if len(programA_lines) != len(programB_lines):
            raise ValueError('Program are of different lengths: %d and %d!' % (
                len(programA_lines), len(programB_lines)))
        faulty_lines = []
        for index, (lineA, lineB) in enumerate(zip(programA_lines, programB_lines)):
            if lineA != lineB:
                edit_count = self.fix_line(
                    copy.deepcopy(lineA), copy.deepcopy(lineB))
                for _ in range(edit_count):
                    faulty_lines.append(index)
        return faulty_lines

    def get_compiler_errors(self, program_id, program, name_dict, name_seq):
        program_source = self.get_program_source_from_vector(program, name_dict, name_seq) 
        if self.compilation_error_store is not None:
            err_list = self.compilation_error_store.get_errors(program_id, program_source)
        else:
            err_list, _ = compilation_errors(program_source)
        return err_list

    def get_compiler_errors_from_source(self, program_id, program_source):
        if self.compilation_error_store is not None:
            err_list = self.compilation_error_store.get_errors(program_id, program_source)
        else:
            err_list, _ = compilation_errors(program_source)
        return err_list

    def update(self, program_id, program_vector, name_dict, name_seq, action, \
                known_vectorized_correct_program, use_compiler, reject_edits=None):
        assert known_vectorized_correct_program is not None
        program_vector, cursor = self.gui_pop_cursor(copy.deepcopy(program_vector))
        assert program_vector[-1] == self.new_line
        if reject_edits is None:
            reject_edits = self.reject_spurious_edits
        action_name = self.actions[action] if type(action) != str else action
        reward = self.step_penalty
        halt = False
        compilation_reward = 0
        err_list = None

        if self.top_down_movement:
            # moving further from end-of-file
            if len(program_vector) - 1 == cursor and 'move_' in action_name:
                assert program_vector[cursor] == self.new_line
                halt = True
                return self.gui_set_cursor(program_vector, cursor), reward, halt, err_list

        updated_program_vector, updated_cursor = self.act(copy.deepcopy(program_vector), cursor, action_name)

        if 'move_' not in action_name:
            reward += self.edit_penalty
            err_list = self.get_compiler_errors(program_id, updated_program_vector, name_dict, \
                name_seq) if use_compiler else self.get_errors_from_ground_truth(
                    updated_program_vector, known_vectorized_correct_program)
            if len(err_list) == 0:
                halt = True
                reward += self.correct_fix_reward
            else:
                old_err_list = self.get_compiler_errors(program_id, program_vector, name_dict, name_seq) if \
                    use_compiler else self.get_errors_from_ground_truth(
                        program_vector, known_vectorized_correct_program)
                corrections = len(old_err_list) - len(err_list)
                if corrections > 0:
                    reward += self.error_resolution_reward
                if reject_edits and corrections <= 0:
                    updated_program_vector = program_vector
                    updated_cursor = cursor
                    err_list = old_err_list
                    halt = False

        # checks
        if __debug__:
            error_string = 'org:{}\nupdated:{}\nzipped:{}\ncursor:{}, action:{}, len(org):{}, len(updated):{}'.format(program_vector,
                                                                                                                      updated_program_vector, list(zip(
                                                                                                                          program_vector, updated_program_vector)), cursor, action_name,
                                                                                                                      len(program_vector), len(updated_program_vector))
        assert 'move' not in action_name or program_vector == updated_program_vector, error_string
        assert not ('insert' in action_name and compilation_reward > 0) or len(program_vector) + 1 == len(updated_program_vector), error_string
        assert 'delete' not in action_name or len(program_vector) - 1 == len(updated_program_vector) or len(program_vector) == len(updated_program_vector), error_string

        return self.gui_set_cursor(updated_program_vector, updated_cursor), reward, halt, (None if err_list is None else len(err_list))

    def localize_error(self, programA, programB):
        '''programA is the mutated program while programB is the original one!'''
        assert type(programA) == list and type(programB) == list, 'types:{}, {}, programs:\n{}\n{}'.format(type(programA), type(programB), programA, programB)
        assert type(self.new_line) == int
        programA = self.filter_program_vector(programA)
        programB = self.filter_program_vector(programB)
        programA_lines = split_list(programA, delimiter=self.new_line, keep_delimiter=True)
        programB_lines = split_list(programB, delimiter=self.new_line, keep_delimiter=True)

        if len(programA_lines) != len(programB_lines):
            raise ValueError('Program are of different lengths: %d and %d!' % (
                len(programA_lines), len(programB_lines)))

        different_line_num = None
        for j, (A_line, B_line) in enumerate(zip(programA_lines, programB_lines)):
            if A_line != B_line:
                different_line_num = j
                break

        error_at = None
        if different_line_num is not None:
            error_at = sum([len(line) for line in programA_lines[:different_line_num]])
            for i, (a, b) in enumerate(zip(A_line, B_line)):
                if a != b:
                    break
            error_at += i
        assert error_at is not None
        assert programA[error_at] == a, '{},{},{},{}'.format(self.format_program_vector(programA), '\na:', a, programA[error_at])
        assert programB[error_at] == b, '{},{},{},{}'.format(self.format_program_vector(programB), '\nb:', b, programB[error_at])

        fix_action = None
        if a in self.mutables and b in self.mutables:
            del_cost = self.fix_line(A_line[i + 1:], B_line[i:], recursive=True)
            ins_cost = self.fix_line(A_line[i:], B_line[i + 1:], recursive=True)
            fix_action = 'delete' if del_cost <= ins_cost else 'insert'

        # b has been deleted from original line
        if fix_action == 'insert' or (a not in self.mutables and b in self.mutables):
            fix_action = 'insert' + self.rev_tl_dict[b][-1]
        # a is a duplicated token
        elif fix_action == 'delete' or (a in self.mutables and b not in self.mutables):
            fix_action = 'delete' + self.rev_tl_dict[a][-1]
        else:
            err_str = 'Should not happen:' + self.devectorize(A_line) + '\n' + self.devectorize(B_line)
            assert False, err_str

        return different_line_num, error_at, fix_action


    def new_random_episode_with_right_actions(self, program_id, cursor_program, name_dict, name_seq, original_program, org_err_cnt, use_compiler, toy=False):

        def get_replacement_actions(action_names, new_action_name):
            action_names.append(new_action_name)
            if new_action_name in ['insert;', 'insert,']:
                act_seq_to_rep_act_map = self.action_sequence_to_replacement_action_map

                if len(action_names) >= 2:
                    action_seq = ''.join(action_names[-2:])
                    if action_seq in act_seq_to_rep_act_map:
                        action_names = action_names[:-2] + [act_seq_to_rep_act_map[action_seq]]
                if len(action_names) >= 3:
                    action_seq = ''.join(action_names[-3:])
                    if action_seq in act_seq_to_rep_act_map:
                        action_names = action_names[:-3] + [act_seq_to_rep_act_map[action_seq]]
            return action_names

        def act(program, cursor, action_name):
            program, cursor = self.act(program, cursor, action_name)
            return program, cursor, self.get_line_number(program, cursor)

        action_names = []
        error_line, error_cursor, fix_action_name = self.localize_error(cursor_program, original_program)
        current_program, current_cursor = self.gui_pop_cursor(copy.deepcopy(cursor_program))
        current_cursor_line = self.get_line_number(current_program, current_cursor)

        iterations = 0
        while True:
            if error_line < current_cursor_line:
                current_program, current_cursor, current_cursor_line = act(current_program, current_cursor, 'move_up')
                action_names = get_replacement_actions(action_names, 'move_up')
            elif error_line > current_cursor_line:
                current_program, current_cursor, current_cursor_line = act(current_program, current_cursor, 'move_down')
                action_names = get_replacement_actions(
                    action_names, 'move_down')
            elif error_cursor < current_cursor:
                current_program, current_cursor, current_cursor_line = act(current_program, current_cursor, 'move_left')
                action_names = get_replacement_actions(action_names, 'move_left')
            elif error_cursor > current_cursor:
                current_program, current_cursor, current_cursor_line = act(current_program, current_cursor, 'move_right')
                action_names = get_replacement_actions(action_names, 'move_right')
            else:
                assert error_line == current_cursor_line and error_cursor == current_cursor
                current_program, current_cursor, current_cursor_line = act(current_program, current_cursor, fix_action_name)
                action_names = get_replacement_actions(action_names, fix_action_name)
                diffs = self.get_errors_from_ground_truth(current_program, original_program)
                if len(diffs) == 0:
                    break
                else:
                    error_line, error_cursor, fix_action_name = self.localize_error(
                        current_program, original_program)

            iterations += 1
            if iterations > 1000:
                print('\ncurrent_program:')
                print(self.format_program_vector(self.filter_program_vector(current_program)))

                print('\noriginal_program:')
                print(self.format_program_vector(self.filter_program_vector(original_program)))

                raise ValueError('Went into a possible infinite loop, exiting!')

        if self.single_delete:
            action_names = [('delete' if 'delete' in each else each) for each in action_names]

        actions = [self.get_action(x) for x in action_names]

        final_errs = self.get_compiler_errors(program_id, current_program, name_dict, name_seq) if not toy and use_compiler else \
            self.get_errors_from_ground_truth(current_program, original_program)
        if len(final_errs) == 0 and not toy and use_compiler:
            init_errs = len(self.get_compiler_errors(program_id, cursor_program, name_dict, name_seq))
            assert init_errs > 0, 'init_errs:%d, final_errs:%d' % (init_errs, len(final_errs))
            assert len(final_errs) == 0, 'init_errs:%d, final_errs:%d' % (init_errs, len(final_errs))
        return program_id, cursor_program, original_program, org_err_cnt, actions


class Environment(Env_engine):
    def __init__(self, dataset, step_penalty, seed, GE_ratio=None, top_down_movement=True, single_delete=True,
                 reject_spurious_edits=True, compilation_error_store=None, train_data_size=0, valid_data_size=0, test_data_size=0,
                 GE_code_ids=None, actions=None, verbose=False, single_program=None, sparse_rewards=True):

        assert (GE_ratio is None and GE_code_ids is not None) or (GE_ratio is not None and GE_code_ids is None)

        Env_engine.__init__(self, dataset.get_tl_dict(), seed, step_penalty=step_penalty, top_down_movement=top_down_movement,
                         reject_spurious_edits=reject_spurious_edits, compilation_error_store=compilation_error_store,
                         single_delete=single_delete, actions=actions, sparse_rewards=sparse_rewards)

        if single_program is not None:
            td = self.tl_dict
            tokenized_program, name_dict, name_sequence = C_Tokenizer().tokenize_single_program(single_program)
            single_ex_dataset = namedtuple('single_ex_dataset', ['single_ex', 'name_dict_store'], verbose=True)
            self.dataset = single_ex_dataset(single_ex = {'single': (self.vectorize(tokenized_program),
                                                [td['_eos_'], td['-new-line-'], td['_pad_']])},
                                            name_dict_store = {'single': (name_dict, name_sequence)})
            self.data_sizes = {'single': 1}
            self.code_ids = {'single': ['single']}
        else:
            self.verbose = verbose
            self.dataset = dataset
            train_data_size = dataset.data_size[0] if train_data_size == 0 else min(train_data_size, dataset.data_size[0])
            valid_data_size = dataset.data_size[1] if valid_data_size == 0 else min(valid_data_size, dataset.data_size[1])
            test_data_size = dataset.data_size[2] if test_data_size == 0 else min(test_data_size, dataset.data_size[2])

            train_code_ids = list(self.dataset.train_ex.keys())[:train_data_size]

            guided_train_data_size = int(
                GE_ratio * train_data_size) if GE_code_ids is None else len(GE_code_ids)
            if GE_code_ids is None:
                guided_train_code_ids = set(self.rng.choice(train_code_ids, guided_train_data_size, replace=False))
            else:
                guided_train_code_ids = GE_code_ids

            # raw test dataset
            real_data_size = 0
            real_data_keys = []
            try:
                self.real_test_data = self.dataset.real_test_data
            except AttributeError:
                pass
            else:
                real_data_size = len(self.real_test_data)
                real_data_keys = list(self.real_test_data.keys())

            # seeded test dataset
            seeded_data_size = 0
            seeded_data_keys = []
            try:
                self.seeded_test_data = self.dataset.seeded_test_data
            except AttributeError:
                pass
            else:
                seeded_data_size = len(self.seeded_test_data)
                seeded_data_keys = list(self.seeded_test_data.keys())

            self.data_sizes = {'train': train_data_size, 'valid': valid_data_size, 'test': test_data_size,
                               'real': real_data_size, 'seeded': seeded_data_size, 'GE_train': guided_train_data_size}
            self.code_ids = {'train': train_code_ids, 'GE_train': guided_train_code_ids, 'valid': list(self.dataset.valid_ex.keys()),
                             'test': list(self.dataset.test_ex.keys()), 'real': real_data_keys, 'seeded': seeded_data_keys}
            self.rng.shuffle(self.code_ids['train'])

    @property
    def train_data_size(self):
        return self.data_sizes['train']

    @property
    def valid_data_size(self):
        return self.data_sizes['valid']

    @property
    def test_data_size(self):
        return self.data_sizes['test']

    @property
    def real_test_data_size(self):
        return self.data_sizes['real']

    @property
    def seeded_test_data_size(self):
        return self.data_sizes['seeded']

    @property
    def guided_train_data_size(self):
        return self.data_sizes['GE_train']

    def get_data(self, which):
        assert which != 'GE_train', which

        if which == 'train':
            data = self.dataset.train_ex
        elif which == 'test':
            data = self.dataset.test_ex
        elif which == 'real':
            data = self.real_test_data
        elif which == 'seeded':
            data = self.seeded_test_data
        elif which == 'valid':
            data = self.dataset.valid_ex
        elif which == 'single':
            data = self.dataset.single_ex
        else:
            raise ValueError('%s is not a supported dataset' % which)

        return data, self.data_sizes[which]

    def get_identifiers(self, code_id):
        try:
            if '_v' in code_id:
                code_id = code_id.split('_')[0]
            name_dict, name_seq = self.dataset.name_dict_store[code_id]
        except AttributeError:      # in case of single program evaluation
            name_dict, name_seq = {}, None
        return name_dict, name_seq

    def new_indexed_episode(self, index, use_compiler, which='train', code_id=None):
        data, size = self.get_data(which)
        if code_id is None:
            assert index < size, 'index:%d >= %s-data-size:%d' % (
                index, which, size)
            code_id = self.code_ids[which][index]
        name_dict, name_seq = self.get_identifiers(code_id)
        program, original_program = copy.deepcopy(data[code_id])
        error_count = len(self.get_compiler_errors(code_id, program, name_dict, name_seq) if use_compiler else self.get_errors_from_ground_truth(program, original_program))
        if 'GE_train' in self.code_ids and code_id in self.code_ids['GE_train']:
            return self.new_random_episode_with_right_actions(code_id, self.gui_set_cursor(program), name_dict, name_seq, original_program, error_count, use_compiler)
        return code_id, self.gui_set_cursor(program), original_program, error_count, None

    
    def new_random_episode_with_right_actions(self, code_id, cursor_program, name_dict, name_seq, original_program, error_count, use_compiler):
        return Env_engine.new_random_episode_with_right_actions(self, code_id, cursor_program, name_dict, name_seq, original_program, error_count, use_compiler)

    def new_random_episode(self, use_compiler, which, GE_selection_probability):
        data, size = self.get_data(which)
        index = self.rng.randint(size)
        code_id = self.code_ids[which][index]
        name_dict, name_seq = self.get_identifiers(code_id)
        program, original_program = copy.deepcopy(data[code_id])
        error_count = len(self.get_compiler_errors(
            code_id, program, name_dict, name_seq) if use_compiler else \
            self.get_errors_from_ground_truth(program, original_program))
        if code_id in self.code_ids['GE_train'] and coin_flip(self.rng, GE_selection_probability):
            return self.new_random_episode_with_right_actions(code_id, self.gui_set_cursor(program), name_dict, name_seq, original_program, error_count, use_compiler)
        return code_id, self.gui_set_cursor(program), original_program, error_count, None


class load_data:
    def _deserialize(self, data_folder, allow_pickle):
        train_ex = np.load(os.path.join(data_folder, 'examples-train.npy'), allow_pickle=allow_pickle).item()
        valid_ex = np.load(os.path.join(data_folder, 'examples-validation.npy'), allow_pickle=allow_pickle).item()
        test_ex = np.load(os.path.join(data_folder, 'examples-test.npy'), allow_pickle=allow_pickle).item()
        assert train_ex is not None and valid_ex is not None and test_ex is not None
        return train_ex, valid_ex, test_ex

    def __init__(self, data_folder, load_real_test_data=False, load_seeded_test_data=False, load_only_dicts=False, shuffle=False, seed=1189, allow_pickle=True):
        self.rng = np.random.RandomState(seed)
        self.tl_dict, self.rev_tl_dict = load_dictionaries(data_folder, allow_pickle)
        assert self.tl_dict is not None and self.rev_tl_dict is not None
        assert self.tl_dict['-new-line-'] == 2
        if load_only_dicts:
            return
        if load_real_test_data:
            try:
                self.real_test_data = np.load(os.path.join(data_folder, 'test_real_raw.npy'), allow_pickle=allow_pickle).item()
            except:
                self.real_test_data = np.load(os.path.join(data_folder, 'test_raw.npy'), allow_pickle=allow_pickle).item()
        if load_seeded_test_data:
            try:
                self.seeded_test_data = np.load(os.path.join(data_folder, 'test_real_seeded.npy'), allow_pickle=allow_pickle).item()
            except:
                self.seeded_test_data = np.load(os.path.join(data_folder, 'test_seeded.npy'), allow_pickle=allow_pickle).item()
        try:
            self.name_dict_store = np.load(os.path.join(data_folder, 'name_dict_store.npy'), allow_pickle=allow_pickle).item()
        except:
            print('init name_dict_store with {}')
            self.name_dict_store = {}
        if not shuffle:
            # Load originals
            self.train_ex, self.valid_ex, self.test_ex = self._deserialize(data_folder, allow_pickle)
            print("Successfully loaded data.")
        else:
            try:  # to load pre-generated shuffled data
                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(os.path.join(data_folder, 'shuffled'), allow_pickle)
                print("Successfully loaded shuffled data.")
            # or generate it
            except IOError:
                print("Generating shuffled data...")
                sys.stdout.flush()
                # Load originals
                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(data_folder, allow_pickle)
                # Shuffle
                self.rng.shuffle(self.train_ex)
                self.rng.shuffle(self.valid_ex)
                self.rng.shuffle(self.test_ex)
                # Save for later
                make_dir_if_not_exists(os.path.join(data_folder, 'shuffled'))
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-train.npy'), self.train_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-validation.npy'), self.valid_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-test.npy'), self.test_ex)

    def get_raw_data(self):
        return self.train_ex, self.valid_ex, self.test_ex

    def get_batch(self, start, end, which='train'):
        if which == 'train':
            X, Y = list(zip(*self.train_ex[start:end]))
        elif which == 'valid':
            X, Y = list(zip(*self.valid_ex[start:end]))
        elif which == 'test':
            X, Y = list(zip(*self.test_ex[start:end]))
        else:
            raise ValueError('choose one of train/valid/test for which')
        return tuple(prepare_batch(X) + [Y])

    def get_tl_dict(self):
        return self.tl_dict

    def get_rev_tl_dict(self):
        return self.rev_tl_dict

    @property
    def data_size(self):
        return len(self.train_ex), len(self.valid_ex), len(self.test_ex)

    @property
    def vocab_size(self):
        return len(self.tl_dict)
