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
import argparse
import sqlite3
import numpy as np

from data_processing.training_data_generator import load_dictionaries
from util.helpers import remove_empty_new_lines
from util.c_tokenizer import C_Tokenizer

deepfix_base_dir = 'data/deepfix-test-data/' 
RLAssist_base_dir = 'data/network_inputs/RLAssist-seed-1189/'
iitk_db_path = 'data/iitk-dataset/dataset.db'
max_program_len = 450

dummy_correct_program = '_eos_ -new-line- _pad_'

tokenize = C_Tokenizer().tokenize
convert_to_new_line_format = C_Tokenizer().convert_to_new_line_format
convert_to_rla_format = lambda x: remove_empty_new_lines(convert_to_new_line_format(x))

raw_test_data = {}
seeded_test_data = {}

def vectorize(tokens, tldict, max_vector_length=max_program_len):
    vec_tokens = []
    for token in tokens.split():
        try:
            vec_tokens.append(tldict[token])
        except Exception:
            return None

    if len(vec_tokens) > max_vector_length:
        return None

    return vec_tokens

# convert the df test data into rla format
print('iitk_db_path:', iitk_db_path)
with sqlite3.connect(iitk_db_path) as conn:
    cursor = conn.cursor()
    for bin_id in range(5):
        raw_test_data[bin_id] = {}
        seeded_test_data[bin_id] = {}

        bin_raw_test_data = np.load(os.path.join(deepfix_base_dir, 'bin_%d' % bin_id, 'test_raw_bin_%d.npy' % bin_id), allow_pickle=True).item()
        bin_seeded_test_data = np.load(os.path.join(deepfix_base_dir, 'bin_%d' % bin_id, 'test_seeded-typo_bin_%d.npy' % bin_id), allow_pickle=True).item()

        for problem_id, test_programs_ in list(bin_raw_test_data.items()):
            raw_test_data[bin_id][problem_id] = []
            for incorrect_program, name_dict, name_sequence, user_id, code_id in test_programs_:
                raw_test_data[bin_id][problem_id].append((code_id, convert_to_rla_format(incorrect_program), dummy_correct_program))

        for problem_id, test_programs_ in list(bin_seeded_test_data.items()):
            seeded_test_data[bin_id][problem_id] = []
            for incorrect_program, name_dict, name_sequence, user_id, code_id in test_programs_:
                for row in cursor.execute('SELECT tokenized_code from Code where code_id=?;', (code_id,)):
                    correct_program = str(row[0])
                seeded_test_data[bin_id][problem_id].append((code_id, convert_to_rla_format(incorrect_program), convert_to_rla_format(correct_program)))

skipped = 0
for bin_id in range(5):
    print('bin_%d' % bin_id, end=' ')
    target_bin_dir = os.path.join(RLAssist_base_dir, 'bin_%d' % bin_id)
    tl_dict, _ = load_dictionaries(target_bin_dir)

    for which, test_data in [('raw', raw_test_data), ('seeded', seeded_test_data)]:
        test_data_this_fold = {}
        for problem_id in test_data[bin_id]:
            for code_id, inc_tokens, cor_tokens in test_data[bin_id][problem_id]:
                inc_vector = vectorize(inc_tokens, tl_dict)
                corr_vector = vectorize(cor_tokens, tl_dict)
                if inc_vector is None or corr_vector is None:
                    skipped += 1
                    continue
                test_data_this_fold[code_id] = (inc_vector, corr_vector)
        print(which, len(test_data_this_fold), end=' ')
        np.save(os.path.join(target_bin_dir, 'test_%s.npy' % which), test_data_this_fold)
    print()
