"""
FunctionBank represents a collections of functions.

In turn every function can either be:
- from Allamanis tokenisation (Allamanis 2016 paper)
- from iCoDL tokenization (JavaParser internal SOLA tool)

Every function is identified by:
    - project_name
    - file_name
    - function_name
    - tokens_list

At a Token Level:

Allamanis specificities:
    - token_group (tokens within the same identifier)
    #TODO fix the names "jniCollision GC set( )" using the group info to add "_"

iCoDL specificities:
    - prefix for every token
      "STD:)", "SP: ", "STD:throws", "SP: ", "ID:IOException", "SP: ",
    - including space in the encoding

"""
import json
import logging
import os
import re
from tqdm import tqdm
import abc
from abc import ABCMeta
import pandas as pd
import numpy as np
import random
from copy import deepcopy


logging.basicConfig()  # required
logger = logging.getLogger('token-bank')
logger.setLevel(logging.INFO)


class TokenInCode(object):

    def __init__(self, index_id,
                 token_group,
                 text, start_char, line):
        if (index_id == ""):
            index_id = -1
        self.index_id = int(index_id)
        self.token_group = int(token_group)
        self.text = text
        self.start_char = int(start_char)
        self.line = int(line)

    def __repr__(self):
        return json.dumps(self.__dict__.__repr__())


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TokenInCode):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


class Function(object):

    def __init__(self, project_name, file_name, function_name):
        self.project_name = project_name
        self.file_name = file_name
        self.function_name = function_name

    def set_nlp_function_name(self, function_name_tokens):
        self.function_name_tokens = function_name_tokens

    def get_nlp_function_name(self):
        return function_name_tokens

    def set_source_code_string(self, source_code_string):
        """Set a view of code where the tokens are all together."""
        self.source_code_string = source_code_string

    def get_source_code_string(self):
        return self.source_code_string

    def set_tokens(self, tokens_list):
        self.tokens_list = tokens_list

    def get_tokens(self):
        return self.tokens_list

    def set_tokens_categories(self, tokens_list_category):
        self.tokens_list_category = tokens_list_category

    def get_tokens_categories(self):
        return self.tokens_list_category

    def set_formatted_lines(self, lines):
        self.formatted_lines = lines

    def get_formatted_lines(self):
        return self.formatted_lines

    def set_tokens_in_code(self, tokens_in_code):
        self.tokens_in_code = tokens_in_code

    def get_tokens_in_code(self):
        return self.tokens_in_code

    def __repr__(self):
        if len(self.tokens_list) > 10:
            first_tokens = "-".join(self.tokens_list[:10])
        else:
            first_tokens = "-".join(self.tokens_list)
        return f'{self.project_name} > {self.file_name} > {self.function_name} :\n{first_tokens}'


class FunctionsBank(object):

    def __init__(self, data, project_name, functions=None):
        """Save data."""
        self.data = data
        self.project_name = project_name
        self.functions = functions
        self.df_functions_shuffled = None

    @classmethod
    def from_single_file(cls, json_file, project_name):
        """Load the json file with tokens."""
        logger.info(f" Loaded file = {json_file}")
        with open(json_file, 'r') as in_file:
            data = json.load(in_file)
            return cls(data, project_name)
        n_datapoints = len(data)
        logger.info(f" Unique datapoints: {n_datapoints}")

    @classmethod
    def from_folder_of_json(cls, folder_name, project_name):
        """Load all the json in the folder."""
        files = os.listdir(folder_name)
        logger.info(f" Loaded Folder = {folder_name}")
        data = []
        for f in tqdm(files):
            icodl_json_path = os.path.join(folder_name, f)
            logger.debug(f" File = {icodl_json_path}")
            with open(icodl_json_path, 'r') as json_file:
                data_chunk = json.load(json_file)
            data += data_chunk
        return cls(data, project_name)

    @classmethod
    def from_dataframe(cls, df):
        """Load from a Pandas Dataframe of functions.

        Required columns:
        - project_name: string
        - file_name: string
        - function_name: string
        - source_code: string
        - token_list: List[string]
        - formatted_lines: List[string]
        """
        # listOfReading= [(Reading(row.HourOfDay,row.Percentage)) for index, row in df.iterrows() ]
        functions = []
        logger.info(f" Loading from Dataframe")
        project_name = None
        available_columns = set(df.columns)
        accepted_columns = set([
            'function_name_tokens', 'id_same_identifier_list',
            'cap_original_tokens', 'filtered_original_tokens',
            'formatted_lines', 'tokens_in_code',
            'option_correct', 'options_tfidf', 'options_random',
            'options', 'options_nlp', 'id_body_hash',
            'uuid', 'token_category'
        ])
        for index, row in tqdm(df.iterrows()):
            if project_name is not None:
                project_name = row.project_name
            read_function = \
                Function(
                    project_name=row.project_name,
                    file_name=row.file_name,
                    function_name=row.function_name)
            read_function.set_source_code_string(row.source_code_string)
            read_function.set_tokens(row.tokens_list)
            for col in available_columns.intersection(accepted_columns):
                setattr(read_function, col, getattr(row, col))
            #if 'formatted_lines' in available_columns:
            #    read_function.set_formatted_lines(row.formatted_lines)
            functions.append(read_function)

        return cls(data=None, project_name=project_name, functions=functions)

    @classmethod
    def load(cls, path):
        """Load collection form DataFrame as json pandas file."""
        df_read = pd.read_json(path, orient='records')
        df_read = df_read[sorted(df_read.columns)]
        return cls.from_dataframe(df_read)

    def parse_info(self):
        """Parse all info in the data (functions and tokens)."""
        self.functions = []
        for raw_function in self.data:
            parsed_function = \
                self.parse_function(raw_function,
                                    project_name=self.project_name)
            tokens = []
            tokens_category = []
            for raw_token in parsed_function.get_tokens():
                parsed_token = self.parse_token(raw_token)
                parse_category = self.parse_category(raw_token)
                tokens.append(parsed_token)
                tokens_category.append(parse_category)
            parsed_function.set_tokens(tokens)
            parsed_function.set_tokens_categories(tokens_category)
            # prepare additional info for the source code reconstruction
            self.enrich_function(parsed_function)
            source_code = self.build_source_code(parsed_function)
            parsed_function.set_source_code_string(source_code)
            self.functions.append(parsed_function)
        n_total_functions = len(self.functions)
        logger.info(f"This Bank contains {n_total_functions}")

    def get_source_files_list(self):
        """Get the original file names of all the functions in the bank."""
        if self.functions is None:
            logger.info("Initialize the bank with some functions.")
            raise AttributeError("No functions have been parsed.")
        filenames = []
        for f in self.functions:
            filenames.append(f.file_name)
        return list(set(filenames))

    @abc.abstractmethod
    def parse_token(self, raw_token):
        """Parse a token obect."""
        raise NotImplementedError()

    @abc.abstractmethod
    def parse_category(self, raw_token):
        """Parse the token category."""
        raise NotImplementedError()

    @abc.abstractmethod
    def parse_function(self, raw_function, project_name):
        """Parse a function obect."""
        raise NotImplementedError()

    @abc.abstractmethod
    def enrich_function(self, function_object):
        """Enrich the function object with addinal info (bank type dependent).

        For example the group of tokens, belonging to the same identifier."""
        raise NotImplementedError()

    @abc.abstractmethod
    def build_source_code(self, function_object):
        """Build the source code from the parsed function object."""
        raise NotImplementedError()

    def __iter__(self):
        if self.functions is not None:
            return self.functions.__iter__()
        return self.data.__iter__()

    def __getitem__(self, item):
        if self.functions is not None:
            return self.functions[item]
        return self.data[item]

    def _clean_slash(self, path):
        return path.replace("/", "")

    def to_dataframe(self):
        """Create a Dataframe from the function collection."""
        rows = []
        if (self.functions is None):
            raise ValueError("Your Bank doesn't contains any function.")
        for f in self.functions:
            rows.append(f.__dict__)
        df = pd.DataFrame(rows)
        df = df[sorted(df.columns)]
        return df

    def save(self, path):
        """Save the collection as a DataFrame in json format."""
        df_self = self.to_dataframe()
        df_self.to_json(path, orient='records')

    def pre_shuffle(self, shuffle_seed=42):
        """Shuffle the collection of functions."""
        random.seed(shuffle_seed)
        # self.shuffled_functions = deepcopy(self.functions)
        self.shuffle_seed = shuffle_seed
        logger.info('Shuffle performed ' + f'(seed {self.shuffle_seed})')
        # random.shuffle(self.shuffled_functions)
        # rows = []
        # for f in self.shuffled_functions:
        #     rows.append(f.__dict__)
        # self.df_functions_shuffled = pd.DataFrame(rows)
        df_self = deepcopy(self.to_dataframe())
        df_self = \
            df_self.sample(
                frac=1,
                random_state=shuffle_seed).reset_index(drop=True)
        self.df_functions_shuffled = df_self

    def save_json(self, path, indices=None, from_suffled=False):
        """Save the collection as JSON file for experiment."""
        if (self.functions is None):
            raise ValueError("Your Bank dowsn't contains any function.")
        if (indices is None or len(indices) == 0):
            raise ValueError("Empty list of indices to save.")
        if from_suffled and self.df_functions_shuffled is None:
            raise ValueError("Shuffle before if you ask for the shuffle version.")

        if from_suffled:
            df = self.df_functions_shuffled
            logger.info(' Sampling from SHUFFLED version ' +
                        f'(seed {self.shuffle_seed})')
        else:
            df = self.to_dataframe()
            logger.info(f' Sampling from ORDERED version')

        # create dataframe
        idx = np.array(indices)
        df.iloc[idx].to_json(path, orient='records')


class AllamanisBank(FunctionsBank):

    def parse_token(self, raw_token):
        """Parse a token object."""
        return raw_token

    def parse_category(self, raw_token):
        """Parse the token category."""
        return "UNKNOWN"

    def parse_function(self, raw_function, project_name):
        """Parse a function obect."""
        only_filename = \
            self._clean_slash(raw_function['filename'].split(':')[0])
        function_name = raw_function['filename'].split(':')[-1]
        raw_tokens = raw_function['tokens']
        new_function = \
            Function(project_name=project_name,
                     file_name=only_filename,
                     function_name=function_name)
        new_function.set_tokens(raw_tokens)
        new_function.set_nlp_function_name(raw_function['name'])
        return new_function

    def enrich_function(self, function_object):
        """Enrich the function object with group of tokens same identifier."""
        self._get_id_cap_filter_tokens(function_object)

    def build_source_code(self, function_object):
        """Build the source code from the parsed function object."""
        cap_original_tokens = function_object.cap_original_tokens
        # remove sentence start and end
        cap_original_tokens = cap_original_tokens[1:-1]
        return self._beautify_allamanis_tokens(cap_original_tokens)

    def _get_id_cap_filter_tokens(self, function_object):
        """Get group id for same identifier, capitalized and filtered tokens."""
        original_tokens = function_object.get_tokens()

        cap_original_tokens = []
        filtered_original_tokens = []
        id_same_identifier_list = []
        id_identifier = 0
        block_increment = False
        capitalize_next_token = False

        old_id_identifier = -1

        for t in original_tokens:
            if (t == '<id>'):
                block_increment = True
                capitalize_next_token = False
            elif (t == '</id>'):
                block_increment = False
                id_identifier += 1
                capitalize_next_token = True
            else:
                id_same_identifier_list.append(id_identifier)
                if not block_increment:
                    id_identifier += 1
                if capitalize_next_token:
                    t = t.capitalize()
                    capitalize_next_token = False
                if id_identifier == old_id_identifier:
                    capitalize_next_token = True
                filtered_original_tokens.append(t)
            old_id_identifier = id_identifier
            # always add (sometimes it will be cap)
            cap_original_tokens.append(t)
        function_object.id_same_identifier_list = id_same_identifier_list
        function_object.cap_original_tokens = cap_original_tokens
        function_object.filtered_original_tokens = filtered_original_tokens[1:-1]

    def _beautify_allamanis_tokens(self, cap_original_tokens, verbose=False):
        """Give Allamanis tokens the shape of a function."""
        awful_source_code = "".join(cap_original_tokens)
        out_text_regex = awful_source_code
        if verbose:
            print('AFTER: ', '"".join\n', out_text_regex)
        out_text_regex = re.sub('(?<=[a-zA-Z])<id>', ' ', out_text_regex)
        if verbose:
            print('AFTER: ', '(?<=[a-zA-Z])<id>\n', out_text_regex)
        out_text_regex = re.sub('(?<=[^a-zA-Z])<id>', '', out_text_regex)
        if verbose:
            print('AFTER: ', '(?<=[^a-zA-Z])<id>\n', out_text_regex)
        out_text_regex = re.sub('<\/id>(?=[a-zA-Z])', ' ', out_text_regex)
        if verbose:
            print('AFTER: ', '<\/id>(?=[a-zA-Z])\n', out_text_regex)
        out_text_regex = re.sub('<\/id>(?=[^a-zA-Z])', '', out_text_regex)
        if verbose:
            print('AFTER: ', '<\/id>(?=[^a-zA-Z])\n', out_text_regex)
        awful_source_code = out_text_regex
        # AUGMENT WITH DUMMY FUNCTION DEFINITION
        function_like_code = 'int f() ' + out_text_regex + '\n'
        return function_like_code


class CodlBank(FunctionsBank):

    def parse_token(self, raw_token):
        """Parse a token object."""
        return raw_token[raw_token.find(":") + 1:]

    def parse_category(self, raw_token):
        """Parse the token category."""
        return raw_token[:raw_token.find(":")]

    def parse_function(self, raw_function, project_name):
        """Parse a function obect."""
        abs_path = raw_function['metadata']['file']
        only_filename = abs_path.split('/')[-1]
        function_name = raw_function['metadata']['function']
        raw_tokens = raw_function['data']
        new_function = \
            Function(project_name=project_name,
                     file_name=only_filename,
                     function_name=function_name)
        new_function.set_tokens(raw_tokens)
        return new_function

    def enrich_function(self, function_object):
        """Enrich the function object with addinal info."""
        # No enrichment required for this Bank type.
        return

    def build_source_code(self, function_object):
        """Build the source code from the parsed function object."""
        tokens = function_object.get_tokens()
        return "".join(tokens)
