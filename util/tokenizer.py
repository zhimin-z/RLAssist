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

from abc import abstractmethod

class UnexpectedTokenException(Exception):
    pass

class EmptyProgramException(Exception):
    '''In fn tokenizer:get_lines(), positions are empty, most probably the input program \
       is without any newline characters or has a special character such as ^A'''
    pass
    
class FailedTokenizationException(Exception):
    '''Failed to create line-wise id_sequence or literal_sequence or both'''
    pass

class Tokenizer:

    @abstractmethod
    def tokenize(self, code, keep_format_specifiers=False, keep_names=True, \
                 keep_literals=False):
        return NotImplemented
    
    def enforce_restrictions(self, result):    
        return result
