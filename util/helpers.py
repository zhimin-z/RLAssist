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

import os, tempfile, time, sys, sqlite3, random, copy, threading, subprocess
import numpy as np
from unqlite import UnQLite


def prepend_line_numbers(program):
    return '\n'.join(['[%-2d] ' % (line_number+1) + line for line_number, line in enumerate(program.split('\n')) if line.strip() != ''])


def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])


def coin_flip(rng, probability):
    return rng.random_sample() < probability


def remove_empty_new_lines(program):
    new_program = program.replace('-new-line- -new-line-', '-new-line-')
    while program != new_program:
        program = new_program
        new_program = program.replace('-new-line- -new-line-', '-new-line-')
    return program


def remove_imports(program_text):
    count = program_text.count('#include')
    lines = program_text.split('\n')
    line_count = len(lines)
    lines = [line for line in lines if '#include' not in line]
    err_str = 'input_program:\n' + program_text + '\nnew_program:\n' + '\n'.join(lines)
    assert len(lines) > 0, err_str
    if line_count - len(lines) == count:
        new_program = '\n'. join(lines)
    else:
        raise ValueError('Could not remove includes\n' + err_str)
    return new_program


def get_error_list(error_message):
    error_list = []
    for line in error_message.splitlines():
        if 'error:' in line:
            error_list.append(line)
    return error_list


def clang_format(source_file):
    shell_string = 'clang-format %s' % source_file.name
    clang_output = subprocess.check_output(shell_string, timeout=30, shell=True)
    return clang_output.strip()


def compilation_errors(string, which='gcc'):
    name1 = int(time.time() * 10**6)
    name2 = np.random.random_integers(0, 1000)
    filename = '/tmp/tempfile_%d_%d.c' % (name1, name2)
    outfile = '/tmp/temp.out'
    with open(filename, 'w+') as f:
        f.write(string)

    if which == 'clang':
        shell_string = "clang -w -std=c99 -pedantic %s -lm -o %s" % (filename, outfile)
        # shell_string = "clang -w -std=c99 -pedantic -fsyntax-only -fno-caret-diagnostics %s" % filename
    else:
        shell_string = "gcc -w -std=c99 -pedantic %s -lm -o %s" % (filename, outfile)
        # shell_string = "gcc -w -std=c99 -pedantic -fsyntax-only %s" % filename

    attempts = 0
    while True:
        attempts += 1
        try:
            result = subprocess.check_output(shell_string, timeout=30, shell=True, stderr=subprocess.STDOUT)
            break
        except subprocess.CalledProcessError as e:
            result = e.output
            break
        except subprocess.TimeoutExpired:
            print('~~~~~~~~ WARNING: got a clang timeout on the attempt: %d, retrying!' % attempts)
            if attempts > 2:
                raise

    os.unlink('%s' % (filename,))

    result = remove_non_ascii(result)

    error_list = get_error_list(result)
    # for line in result.splitlines():
    #     if 'error:' in line:
    #         error_set.append(line)

    return error_list, result

# not used
def compilation_errors_2(filename, which='clang'):
    if which == 'gcc':
        shell_string = "clang -w -std=c99 -pedantic %s -lm -o %s" % (filename, outfile)
        # shell_string = "clang -w -std=c99 -pedantic -fsyntax-only -fno-caret-diagnostics %s" % filename
    else:
        shell_string = "gcc -w -std=c99 -pedantic %s -lm -o %s" % (filename, outfile)
        # shell_string = "gcc -w -std=c99 -pedantic -fsyntax-only %s" % filename

    try:
        result = subprocess.check_output(shell_string, timeout=30, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        result = e.output

    result = remove_non_ascii(result)
    error_list = get_error_list(result)
    return error_list, result


def isolate_line(program_string, char_index):
    begin = program_string[:char_index].rfind('~') - 2
    while begin - 2 > 0 and program_string[begin - 2] in [str(i) for i in range(10)]:
        begin -= 2
    if program_string[char_index:].find('~') == -1:
        end = len(program_string)
    else:
        end = char_index + program_string[char_index:].find('~') - 2
        while end - 2 > 0 and program_string[end - 2] in [str(i) for i in range(10)]:
            end -= 2
        end -= 1
    return program_string[begin:end]


def extract_line_number(line):
    number = 0
    never_entered = True
    for token in line.split('~')[0].split():
        never_entered = False
        number *= 10
        try:
            number += int(token) - int('0')
        except ValueError:
            raise
    if never_entered:
        raise FailedToGetLineNumberException(line)
    return number


def get_lines(program_string):
    tokens = program_string.split()
    ignore_tokens = ['~'] + [chr(n + ord('0')) for n in range(10)]
    lines = []
    for token in tokens:
        if token in ignore_tokens and token == '~':
            if len(lines) > 0:
                lines[-1] = lines[-1].rstrip(' ')
            lines.append('')
        elif token not in ignore_tokens:
            lines[-1] += token + ' '
    return lines


def recompose_program(lines):
    recomposed_program = ''
    for i, line in enumerate(lines):
        for digit in str(i):
            recomposed_program += digit + ' '
        recomposed_program += '~ '
        recomposed_program += line + ' '
    return recomposed_program


def fetch_line(program_string, line_number, include_line_number=True):
    result = ''
    if include_line_number:
        for digit in str(line_number):
            result += digit + ' '
        result += '~ '
    result += get_lines(program_string)[line_number]
    return result


def tokens_to_source(tokens, name_dict, clang_format=False, name_seq=None, cursor=None, get_tokens=False):
    result_list = []
    result = ''
    type_ = None

    reverse_name_dict = get_rev_dict(name_dict)
    name_count = 0

    for token in tokens.split():
        try:
            prev_type_was_op = (type_ == 'op')

            type_, content = token.split('>_')
            type_ = type_.lstrip('_<')

            if type_ == 'id':
                if name_seq is not None:
                    content = name_seq[name_count]
                    name_count += 1
                else:
                    try:
                        content = reverse_name_dict[content.rstrip('@')]
                    except KeyError:
                        content = 'new_id_' + content.rstrip('@')
            elif type_ == 'number':
                content = content.rstrip('#')

            if type_ == 'directive' or type_ == 'include' or type_ == 'op' or type_ == 'type' or type_ == 'keyword' or type_ == 'APIcall':
                if type_ == 'op' and prev_type_was_op:
                    result = result[:-1] + content + ' '
                else:
                    result += content + ' '
                result_list.append(content)
            elif type_ == 'id':
                result += content + ' '
                result_list.append(content)
            elif type_ == 'number':
                result += '0 '
                result_list.append('0')
            elif type_ == 'string':
                result += '"String" '
                result_list.append('"String"')
            elif type_ == 'char':
                result += "'c' "
                result_list.append("'c'")
        except ValueError:
            type_ = None
            if token == '-new-line-' or token == '~':
                result += '\n'
                result_list.append("\n")
            elif cursor is not None and token == cursor:
                result += '/*CURSOR*/ '
            #else:
            #    result += '\n'

    if get_tokens:
        return result_list

    if not clang_format:
        return result.strip()

    source_file = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
    source_file.write(result)
    source_file.close()

    shell_string = 'clang-format %s' % source_file.name
    clang_output = subprocess.check_output(shell_string, timeout=30, shell=True)
    os.unlink(source_file.name)

    return clang_output.strip()


def done(msg=''):
    if msg == '':
        print('done at', time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S"))
    else:
        print(msg, ',done at', time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S"))


def make_dir_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass


def get_curr_time_string():
    return time.strftime("%b %d %Y %H:%M:%S")


class logger():
    def _open(self):
        if not self.open:
            try:
                self.handle = open(self.log_file, 'a+')
                self.open = True
            except Exception as e:
                print(os.getcwd())
                raise e
        else:
            raise RuntimeError('ERROR: Trying to open already opened log-file!')

    def close(self):
        if self.open:
            self.handle.close()
            self.open = False
        else:
            raise RuntimeError('ERROR: Trying to close already closed log-file!')

    def __init__(self, log_file, move_to_logs_dir=True):
        self.log_file = log_file + '.txt' if '.txt' not in log_file else log_file
        if move_to_logs_dir and not self.log_file.startswith('logs/'):
            self.log_file = os.path.join('logs', self.log_file)
        self.open = False
        self.handle = None
        self._open()

        self.terminal = sys.stdout

        self.log('\n\n-----------------------| Started logging at: {} |----------------------- \n'.format(get_curr_time_string()))

    # for backward compatibility
    def log(self, *msg_list):

        msg_list = list(map(str, msg_list))
        msg = ' '.join(msg_list)

        if not self.open:
            self._open()

        self.handle.write(msg + '\n')
        self.handle.flush()

        print(msg)
        self.terminal.flush()

    # set ** sys.stdout = logger(filename) ** and then simply use print call
    def write(self, message):
        if not self.open:
            self._open()

        self.handle.write(message)
        self.terminal.write(message)

        self.flush()

    @property
    def terminal(self):
        return self._terminal

    @terminal.setter  
    def terminal(self, value):  
        self._terminal = value

    def flush(self):
        self.terminal.flush()
        self.handle.flush()


def get_rev_dict(dict_):
    assert len(dict_) > 0, 'passed dict has size zero'
    rev_dict_ = {}
    for key, value in list(dict_.items()):
        rev_dict_[value] = key

    return rev_dict_


def get_best_checkpoint(checkpoint_directory, name=None):

    def get_best_checkpoint_in_dir(checkpoint_dir):
        best_checkpoint = None
        for checkpoint_name in os.listdir(checkpoint_dir):
            if 'meta' in checkpoint_name and ( name is None or name in checkpoint_name):
                if 'shot' in checkpoint_dir:
                    this_checkpoint = int(checkpoint_name.split('-')[-1].split('.')[0])
                else:
                    #this_checkpoint = int(checkpoint_name[17:].split('.')[0])
                    this_checkpoint = int(checkpoint_name[6:].split('.')[0])

                if best_checkpoint is None or this_checkpoint > best_checkpoint:
                    best_checkpoint = this_checkpoint

        return best_checkpoint

    try:
        bc = get_best_checkpoint_in_dir(os.path.join(checkpoint_directory, 'best'))
    except:
        bc = None
    if bc is None:
        bc = get_best_checkpoint_in_dir(checkpoint_directory)
    if bc is None:
        raise ValueError('No checkpoints found!')
    return bc


def prepare_batch(sequences, msg=False):
    sequence_lengths = [len(seq) for seq in sequences]
    batch_size = len(sequences)
    max_sequence_length = max(sequence_lengths)

    if msg:
        print('max_sequence_length', max_sequence_length)

    inputs = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # initialize with _pad_ = 0

    for i, seq in enumerate(sequences):
        for j, element in enumerate(seq):
            inputs[i, j] = element

    return [inputs, np.array(sequence_lengths)]


def split_list(a_list, delimiter, keep_delimiter=True):
    output = []
    temp = []
    for each in a_list:
        if each == delimiter:
            if keep_delimiter:
                temp.append(delimiter)
            output.append(temp)
            temp = []
        else:
            temp.append(each)
    if len(temp) > 0:
        output.append(temp)
    return output


class experience():
    # programs, actions, updated_programs, rewards, halts
    def __init__(self, buffer_size = 10000):
        self.buffer_size = buffer_size
        self.programs, self.actions, self.updated_programs, \
            self.rewards, self.halts = [experience_buffer(buffer_size) for _ in range(5)]
        self.state = [self.programs, self.actions, self.updated_programs, self.rewards, self.halts]

    def add(self, event):
        old_size = self.size

        for state, value in zip(self.state, event):
            state.add(value)

        assert old_size + 1 >= self.buffer_size or old_size + 1 == self.size

    def extend(self, expr, last=None):
        old_size = self.size

        for old, new in zip(self.state, expr.state):
            old.extend(new, last)

        assert last is not None or (old_size + expr.size >= self.buffer_size or old_size + expr.size == self.size)

    @property
    def size(self):
        return self.actions.size

    def sample(self, sample_size):
        try:
            indices = random.sample(list(range(self.size)), sample_size)
        except:
            print('sample_size:', sample_size, 'self.size:', self.size)
            raise
        programs, actions, updated_programs, rewards, halts = [each.index(indices) for each in self.state]

        return programs, np.array(actions), updated_programs, np.array(rewards), np.array(halts)

    def get(self):
        programs, actions, updated_programs, rewards, halts = [each.buffer for each in self.state]
        return programs, np.array(actions), updated_programs, np.array(rewards), np.array(halts)

    def reset(self):
        for each in self.state:
            each.reset()

        assert self.size == 0
        

class experience_buffer():
    def __init__(self, buffer_size = 10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) == self.buffer_size:
            self.buffer[0 : 1] = []
        self.buffer.append(copy.deepcopy(experience))

    def extend(self, experience, last=None):
        if last is not None:
            last = min(last, experience.size)
            exp_buffer = experience.buffer[-last:]
        else:
            exp_buffer = experience.buffer

        if len(self.buffer) + len(exp_buffer) >= self.buffer_size:
            self.buffer[0 : len(self.buffer) + len(exp_buffer) - self.buffer_size] = []

        for each in exp_buffer:
            self.add(each)

    @property
    def size(self):
        return len(self.buffer)

    def index(self, indices):
        if type(indices) != list:
            index = indices
            try:
                return self.buffer[index]
            except:
                print('type:', type(index), 'index:', index)
                raise
        else:
            return [self.buffer[index] for index in indices]

    def reset(self):
        self.buffer = []


class Compilation_error_db:
    def __init__(self, db_path=''):
        self.db_path = db_path
        self.store = UnQLite()  #loading db from databases freezes the process!!
        self.hits = 0
        self.misses = 0
        self.uncommited_recs = 0

    # keeping prog_id for backward compatibility
    def get_errors(self, prog_id, prog):
        if prog in self.store:
            err_msg = self.store[prog]
            errs = get_error_list(err_msg)
            self.hits += 1
        else:
            errs, err_msg = compilation_errors(prog)
            self.store[prog] = err_msg
            self.misses += 1
            self.uncommited_recs += 1

            if self.uncommited_recs > 0 and self.uncommited_recs % 250 == 0:
                self.commit()
        return errs

    def close(self):
        self.store.close()

    def commit(self, ):
        cnt = self.uncommited_recs
        self.uncommited_recs = 0
        self.store.commit()

    def __len__(self):
        return len(self.store)
