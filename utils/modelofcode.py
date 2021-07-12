"""
It represent a model of code that stores or ceates attention weights.
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
logger = logging.getLogger('model-of-code')
logger.setLevel(logging.INFO)


class ModelOfCode(object):

    def __init__(self, historical_data=None,
                 function_level_data=None):
        self.historical_data = historical_data
        self.function_level_data = function_level_data

    @classmethod
    def from_precomputed_prediction(cls, path):
        """Load prediction form json file."""
        # load model attention predictions
        with open(path, "r") as read_file:
            all_data_model_attention = json.load(read_file)
        return cls(all_data_model_attention)

    @classmethod
    def from_precomputed_prediction_multi_projects(
            cls, path_list=[],
            project_name_list=[]):
        """Load prediction form a list of json file."""
        # load model attention predictions
        all_data_model_attention = []
        for i, (path, project_name) \
                in enumerate(zip(path_list, project_name_list)):
            with open(path, "r") as read_file:
                all_data_this_project = json.load(read_file)
            # append project_name
            all_data_this_project_with_project_name = \
                [{**item, 'project_name': project_name}
                 for item in all_data_this_project]
            all_data_model_attention += all_data_this_project_with_project_name
        return cls(all_data_model_attention)

    def to_dataframe(self):
        """Convert the model predictions into rows of a Pandas DataFrame."""
        rows = []
        if (self.function_level_data is None):
            raise ValueError("Your model doesn't contains any historical." +
                             "function-level prediction. " +
                             "Check to have run aggregate_attention().")
        for historical_prediction in self.function_level_data:
            rows.append(historical_prediction)
        df = pd.DataFrame(rows)
        return df

    def query_history(self, list_function_names, list_tokens_list,
                      list_att_vector=None):
        """Extract the functions queried based on name and body string.

        Note that the query name can be Camelcase and with underscore.
        e.g. isImage
        Whereas the machine data are NLP-tokenized with a comma.
        e.g. ['is,image']

        The function makes them the save using lowercase and removing the
        underscores for the query and removing commas from the nlp-tokenized.
        """

        query_name = \
            [f_name.lower().replace('_', '') for f_name in list_function_names]

        query_body = []
        for token_list in list_tokens_list[:]:
            only_text_tokens = [t['text'].lower() for t in token_list]
            body_string = "".join(only_text_tokens[:])
            query_body.append(body_string)

        data = {'original_name': query_name, 'body_string': query_body}

        if list_att_vector is not None:
            data['att_vector_human'] = list_att_vector

        df_query = \
            pd.DataFrame(data)  # noqa

        n_query_functions = len(df_query)
        logger.info(f'{n_query_functions} query functions')

        df_machine_data = deepcopy(self.to_dataframe())
        df_machine_data['original_name'] = \
            df_machine_data['original_name'].apply(lambda x: x.replace(',', ''))  # noqa

        n_machine_functions = len(df_machine_data)
        logger.info(f'{n_machine_functions} machine functions')

        df_merged = \
            df_machine_data.merge(df_query, how='inner', on=['original_name', 'body_string'])  # noqa
        # remove duplicates
        df_no_duplicates = \
            df_merged.loc[df_merged.astype(str).drop_duplicates(
                subset=['original_name', 'body_string'],
                keep='first').index]
        df_no_duplicates.reset_index(inplace=True, drop=True)

        n_filtered_functions = len(df_no_duplicates)
        logger.info(f'{n_filtered_functions} functions retireved')
        return df_no_duplicates

    def _key_with_max_val(self, dictionary):
        """ a) create a list of the dict's keys and values;
            b) return the key with the max value"""
        v = list(dictionary.values())
        k = list(dictionary.keys())
        return k[v.index(max(v))]

    def aggregate_attention(self, how='avg'):
        """Aggregate attention for every function."""
        # aggregate attention from different token prediction with MAX
        # assumption: prediction of the same function are sequential
        if (self.historical_data is None):
            logger.info('Empty log bank. No human log to aggregate.')
            return

        all_machine_attention = []
        attention_vectors = []
        copy_attention_vectors = []
        prediction = []
        copy_probs = []

        old_function_name = ""
        current_function_name = ""
        old_project_name = ""
        current_project_name = ""

        old_function_len = 0
        current_function_len = 0

        for i, obj in tqdm(enumerate(self.historical_data)):
            current_function_name = obj["original_name"]
            current_project_name = obj["project_name"]
            current_attention = obj["att_vector"]
            current_copy_attention = obj["copy_vector"]
            current_function_len = len(obj["tokens"])
            current_predicted_token = \
                self._key_with_max_val(obj["suggestions"])
            current_copy_prob = obj["copy_prob"]
            logger.debug(str(i) + " - " + current_function_name +
                         " - " + current_project_name +
                         " " + str(current_function_len))

            if ((current_function_name != old_function_name) or
                (current_project_name != old_project_name) or
                (current_function_len != old_function_len)):
                # add vector
                if len(attention_vectors) > 0:
                    new_obj = {}
                    new_obj["original_name"] = old_function_name
                    new_obj["project_name"] = old_project_name
                    new_obj["tokens"] = old_tokens
                    new_obj["body_string"] = "".join(old_tokens[1:-1])
                    new_obj["predicted_tokens"] = prediction
                    new_obj["att_vector_max"] = \
                        self._aggregate_function(attention_vectors, how='max')
                    new_obj["att_vector_avg"] = \
                        self._aggregate_function(attention_vectors, how='avg')
                    new_obj["copy_att_vector_max"] = \
                        self._aggregate_function(copy_attention_vectors, how='max')
                    new_obj["copy_att_vector_avg"] = \
                        self._aggregate_function(copy_attention_vectors, how='avg')
                    new_obj["copy_prob_max"] = np.max(copy_probs)
                    new_obj["copy_prob_avg"] = np.mean(copy_probs)
                    all_machine_attention.append(new_obj)
                # reset
                attention_vectors = []
                copy_attention_vectors = []
                prediction = []

            attention_vectors.append(current_attention)
            copy_attention_vectors.append(current_copy_attention)
            prediction.append(current_predicted_token)
            copy_probs.append(current_copy_prob)
            # DEBUG print(current_function_name)
            old_function_name = current_function_name
            old_project_name = current_project_name
            old_function_len = current_function_len
            old_tokens = obj["tokens"]

        # flush the remaining vectors, aka add the last function
        if len(attention_vectors) > 0:
            new_obj = {}
            new_obj["original_name"] = current_function_name
            new_obj["project_name"] = current_project_name
            new_obj["tokens"] = old_tokens
            new_obj["body_string"] = "".join(old_tokens[1:-1])
            new_obj["att_vector"] = self._aggregate_function(attention_vectors)
            all_machine_attention.append(new_obj)

        n_functions = len(all_machine_attention)
        logger.info(f"{n_functions} unique functions found.")
        all_names = set([d['original_name'] for d in all_machine_attention])
        n_unique_name = len(all_names)
        logger.info(f"{n_unique_name} unique function name found.")
        self.function_level_data = all_machine_attention

    def _aggregate_function(self, vectors, how='avg'):
        """Aggregate attention vectors."""
        np_vectors = [np.array(v) for v in vectors]
        if how == 'avg':
            res = list(np.mean(np_vectors, axis=0))
        elif how == 'max':
            res = list(np.amax(np.vstack(np_vectors), axis=0))
        return res

    def save(self, path):
        """Save the collection as a DataFrame in json format."""
        df_self = self.to_dataframe()
        df_self.to_json(path, orient='records')

    @classmethod
    def load(cls, path):
        """Load collection form DataFrame as json pandas file."""
        with open(path, "r") as read_file:
            all_data_model_attention = json.load(read_file)
        return cls(function_level_data=all_data_model_attention)
