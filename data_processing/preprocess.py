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

import os, sys
import sqlite3
import json
from util.c_tokenizer import C_Tokenizer
tokenize = C_Tokenizer().tokenize

db_path = 'data/iitk-dataset/dataset.db'

def AddColumn(connection, command):
    try:
        connection.execute(command)
    except sqlite3.OperationalError:
        pass #print("Oops!", sys.exc_info()[0], "occurred.")

with sqlite3.connect(db_path) as conn:
    AddColumn(conn, '''ALTER TABLE Code ADD tokenized_code text;''')
    AddColumn(conn, '''ALTER TABLE Code ADD name_dict;''')
    AddColumn(conn, '''ALTER TABLE Code ADD name_seq;''')
    AddColumn(conn, '''ALTER TABLE Code ADD codelength integer;''')

tuples = []
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    for row in cursor.execute("SELECT code_id, code FROM Code;"):
        code_id = str(row[0])
        code = row[1]#.encode('utf-8')
        #print(code[:10])
        tokenized_code, name_dict, name_seq = tokenize(code)
        codelength = len(tokenized_code.split())
        tuples.append((tokenized_code, json.dumps(name_dict),
                       json.dumps(name_seq), codelength, code_id))

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE Code SET tokenized_code=?, name_dict=?, name_seq=?, codelength=? WHERE code_id=?;", tuples)
    conn.commit()
