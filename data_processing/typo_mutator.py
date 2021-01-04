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

import re, numpy as np
from util.c_tokenizer import C_Tokenizer
from util.helpers import isolate_line, fetch_line, extract_line_number, get_lines, recompose_program

class FailedToMutateException(Exception):
    pass

class LoopCountThresholdExceededException(Exception):
    pass

class TypeInferenceFailedException(Exception):
    pass


def do_fix_at_line(corrupted_prog, line, fix):
    try:
        lines = get_lines(corrupted_prog)
    except Exception:
        print(corrupted_prog)
        raise
    if '~' in fix:
        try:
            fix = fix.split(' ~ ')[1]
            fix = fix.strip()
        except:
            print(fix, fix.split(' ~ '))
            raise
    try:
        lines[line] = fix
    except IndexError:
        raise
    return recompose_program(lines)


class Typo_Mutate:
    __action_pattern_map = {
           'delete(' : ("_<op>_\(", ""),
           'delete)' : ("_<op>_\)", ""),
           'delete,' : ("_<op>_,", ""),
           'delete;' : ("_<op>_;", ""),
           'delete{' : ("_<op>_\{", ""),
           'delete}' : ("_<op>_\}", ""),
           'duplicate(' : ("_<op>_\(", "_<op>_( _<op>_("),
           'duplicate)' : ("_<op>_\)", "_<op>_) _<op>_)"),
           'duplicate,' : ("_<op>_,", "_<op>_, _<op>_,"),
           'duplicate{' : ("_<op>_\{", "_<op>_{ _<op>_{"),
           'duplicate}' : ("_<op>_\}", "_<op>_} _<op>_}"),           
           'replace;with,' : ("_<op>_;", "_<op>_,"),
           'replace,with;' : ("_<op>_,", "_<op>_;"),
           'replace;with.' : ("_<op>_;", "_<op>_."),
           'replace);with;)' : ("_<op>_\) _<op>_;", "_<op>_; _<op>_)"),
          }

    __actions = [
           'delete(' ,
           'delete)' ,
           'delete,' ,
           'delete;' ,
           'delete{' ,
           'delete}' ,
           'duplicate(' ,
           'duplicate)' ,
           'duplicate,' ,
           'duplicate{' ,
           'duplicate}' ,
           'replace;with,' ,
           'replace,with;' ,
           'replace;with.' ,
           'replace);with;)',
          ]

    @classmethod
    def get_actions(self):
        return self.__actions


    __mutation_distribution = None    
    __pmf = None
    

    def find_and_replace(self, org_prog, corrupted_prog, regex, replacement, mutation_name):
                       
        # special handling for pointer mutate
        if regex == '[^)@,#\]] (_<op>_\*)(?! _<number>_)':
            positions = [m.span(1) for m in re.finditer(regex, corrupted_prog)]
        else:
            positions = [m.span() for m in re.finditer(regex, corrupted_prog)]
                
        if len(positions) > 1:
            to_corrupt = self.rng.randint(len(positions))
        elif len(positions) == 1:
            to_corrupt = 0
        else:
            return corrupted_prog, None, mutation_name       
                
        line_number = extract_line_number(isolate_line(corrupted_prog, positions[to_corrupt][0]))

        corrupted_prog = corrupted_prog[:positions[to_corrupt][0]] + replacement + corrupted_prog[positions[to_corrupt][1]:]
                
        return corrupted_prog, line_number, mutation_name


    def __update_pmf(self):                
        _dist = self.__mutation_distribution
        assert self.__pmf != None, 'self.__pmf is None in __update_pmf'        
        denominator = 0        
        _max = max(_dist.values()) + 1
        
        # reset pmf
        self.__pmf = []
        pmf = self.__pmf
    
        for action in self.__actions:
            new_val = _max - _dist[action]
            pmf.append(new_val)
            denominator += new_val            

        for i in range(len(pmf)):
            pmf[i] = float(pmf[i])/float(denominator)

    
    update_pmf = __update_pmf
    
    def __init__(self, rng):
        self.rng = rng

        if self.__mutation_distribution == None:
            self.__mutation_distribution = {action:1 for action in self.__actions}
            self.__flag_one_Extra_count = True
        assert self.__mutation_distribution != None, '_mutation_distribution is None'
        
        if self.__pmf == None:
            self.__pmf = []
            self.__update_pmf()
        assert self.__pmf != None, 'pmf is None'

    def easy_mutate(self, org_prog, corrupted_prog, include_kind=False):
        action_map = self.__action_pattern_map
        action = self.rng.choice(self.__actions, p=self.__pmf)
        return self.find_and_replace(org_prog, corrupted_prog, regex=action_map[action][0], replacement=action_map[action][1], mutation_name=action)
    
    def update_mutation_distribution(self, list_of_applied_mutations):
        for each in list_of_applied_mutations:
            self.__mutation_distribution[each] += 1
    
    def get_mutation_distribution(self):
        if self.__flag_one_Extra_count:
            for each in self.__mutation_distribution:
                self.__mutation_distribution[each] = self.__mutation_distribution[each] - 1
            self.__flag_one_Extra_count = False
        return self.__mutation_distribution


def typo_mutate(mutator_obj, prog, max_num_mutations, num_mutated_progs):

    assert len(prog) > 10 and max_num_mutations > 0 and num_mutated_progs > 0, "Invalid argument(s) supplied to the function token_mutate_series_network2"
    corrupt_fix_pair = set()
    
    for _ in range(num_mutated_progs):
        num_mutations = mutator_obj.rng.choice(list(range(max_num_mutations))) + 1 if max_num_mutations > 1 else 1
        this_corrupted = prog
        lines = set()
        mutation_count = 0
        loop_counter = 0
        loop_count_threshold = 50
        mutations = {}
        
        while(mutation_count < num_mutations):
            loop_counter += 1
            if loop_counter == loop_count_threshold:
                print("mutation_count", mutation_count)                
                raise LoopCountThresholdExceededException
            line = None

            this_corrupted, line, mutation_name = mutator_obj.easy_mutate(prog, this_corrupted)     # line is line_number here!

            if line is not None:
                fix = fetch_line(prog, line)
                corrupt_line = fetch_line(this_corrupted, line)

                if fix != corrupt_line:
                    lines.add(line)
                    mutation_count += 1
                    if line not in mutations:
                        mutations[line] = [mutation_name]
                    else:
                        mutations[line].append(mutation_name)
    
        assert len(lines) > 0, "Could not mutate!"
        
        flag_empty_line_in_corrupted = False
        for _line_ in get_lines(this_corrupted):
            if _line_.strip() == '':
                flag_empty_line_in_corrupted = True
                break
                
        if flag_empty_line_in_corrupted:
            continue

        final_corrupt_program = this_corrupted
                            
        sorted_lines = sorted(lines)

        for line in sorted_lines:
            fix = fetch_line(prog, line)
            corrupt_line = fetch_line(this_corrupted, line)
            assert len(fetch_line(prog, line, include_line_number=False).strip()) != 0, "empty fix" 
            assert len(fetch_line(this_corrupted, line, include_line_number=False).strip()) != 0, "empty corrupted line"
            if fix != corrupt_line:
                corrupt_fix_pair.add((this_corrupted, fix))
                mutator_obj.update_mutation_distribution(mutations[line])
                try:
                    this_corrupted = do_fix_at_line(this_corrupted, line, fetch_line(prog, line, include_line_number=False))
                except IndexError:
                    raise
              
        if len(corrupt_fix_pair) > 0:
            mutator_obj.update_pmf()
    
    return list(corrupt_fix_pair)