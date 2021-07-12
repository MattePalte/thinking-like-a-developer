"""
It compares human and machine attention weights.
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

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import os
import json
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
from matplotlib.pyplot import imshow

from copy import copy
import numpy as np

import yaml
import pymongo

from pprint import pprint
from datetime import datetime
import argparse

from scipy.stats import binom
from scipy.stats import poisson


from scipy import stats

from pathlib import Path


logging.basicConfig()  # required
logger = logging.getLogger('attention-comparer')
logger.setLevel(logging.INFO)


class VisualToken(object):

    def __init__(self, index_id, text, x, y, width, height, clicked):
        self.text = text
        if (index_id == ""):
            index_id = -1
        self.index_id = int(index_id)
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.attention = 0
        self.clicked = clicked

    def draw_PIL(self, drw, global_attention=0,
                 guessed_right=False,
                 human=True,
                 almost_correct=False):
        """Draw the patch on the plot."""
        alpha = 0.1
        if global_attention != 0:
            alpha = int((float(self.attention) / global_attention) * 255)
        if self.attention == 0:
            alpha = 0
        if human:
            # human
            if (almost_correct):
                color = (255, 127, 80, alpha)  # orange)
            else:
                if (guessed_right):
                    color = (26, 255, 26, alpha)  # green
                else:
                    color = (255, 102, 102, alpha)  # red
        else:
            # Machine
            color = (0, 191, 255, alpha)  # blue
        border = None
        if self.clicked:
            border = 'red'
        rect = \
            drw.rectangle([
                self.x,
                self.y,
                self.x + self.width,
                self.y + self.height],
                outline=border,
                width=2,
                fill=color)

    def add_attention(self, attention):
        self.attention = attention

    def __repr__(self):
        return 'x:' + str(self.x).zfill(3) \
                + ' - y:' + str(self.y).zfill(3) \
                + ' - width:' + str(self.width).zfill(4) \
                + ' - height:' + str(self.height).zfill(4) \
                + ' - |' + self.text + '|'


class Comparer(object):

    def __init__(self, df_human, df_machine):
        self.df_human = deepcopy(df_human)
        self._normalize_human_df()
        print(sorted(self.df_human.columns))
        print("Unique uuid human: ", len(self.df_human['uuid'].unique()))
        self.df_machine = deepcopy(df_machine)
        self._normalize_machine_df()
        print(sorted(self.df_machine.columns))
        if 'uuid' in self.df_human.columns and \
                'file_name' in self.df_human.columns and \
                'id_body_hash' in self.df_human.columns:
            merging_columns = ['uuid', 'original_name', 'body_string', 'file_name', 'id_body_hash', 'project_name']
        else:
            merging_columns = ['original_name', 'body_string', 'project_name']
        if (('uuid' in self.df_human.columns) and
                ('uuid' in self.df_machine.columns)):
            # keep only the merging columns that are in both human and machine
            intersection_cols = list(
                set(self.df_machine.columns).intersection(
                    set(self.df_human.columns)
            ))
            print('intersection_cols:', intersection_cols)
            # remove all the duplicate columns already available in the human
            # side, otherwise the merge with columns that are lists will be
            # unsuccessful
            machine_columns_to_merge = \
                (set(self.df_machine.columns).difference(
                    set(intersection_cols)
                )).union(set(['uuid']))

            print('machine_columns_to_merge:', machine_columns_to_merge)
            merging_columns = ['uuid']
        else:
            machine_columns_to_merge = self.df_machine.columns
        self.df_intersection = \
            pd.merge(
                left=self.df_machine[machine_columns_to_merge],
                right=self.df_human,
                how='inner',
                on=merging_columns)
        print('AFTER merge: ', len(self.df_intersection))
        self.df_intersection_no_dup = self.df_intersection
        # remove duplicates
        if 'uuid' not in merging_columns:
            self.df_intersection_no_dup = \
                self.df_intersection.loc[
                    self.df_intersection.astype(str).drop_duplicates(
                        subset=['file_name', 'project_name', 'original_name', 'body_string', 'randomcode'],
                        keep='first').index]
        else:
            self.df_intersection_no_dup = \
                self.df_intersection.drop_duplicates(
                    subset=['randomcode', 'uuid']
                )
        self.df_intersection_no_dup.reset_index(inplace=True, drop=True)
        print('AFTER de-duplication: ', len(self.df_intersection_no_dup))

    def get_compare_df(self, only_min_n_annotators=5):
        logger.info(f' {len(self.df_intersection_no_dup)} records in the intersection.')
        logger.info(f' Each of them with at least {only_min_n_annotators} annotators.')
        df_grouped = \
            self.df_intersection_no_dup.groupby(by='uuid').count()['randomcode']
        df_grouped = df_grouped[df_grouped >= only_min_n_annotators]
        n_unique_functions = len(df_grouped)
        fully_annotated_function_uuid = list(df_grouped.index)
        return deepcopy(self.df_intersection_no_dup[
            self.df_intersection_no_dup['uuid'].isin(fully_annotated_function_uuid)
        ])

    def _normalize_human_df(self):
        """Remove underscores in the name and create body_string to match."""
        self.df_human.rename(
            columns={'option_correct': 'original_name',
                     'att_vector': 'att_vector_human'}, inplace=True)

        self.df_human['original_name'] = \
            self.df_human['original_name'].apply(
                lambda x: x.lower().replace('_', ''))

        self.df_human['functionnamebyuser'] = \
            self.df_human['functionnamebyuser'].apply(
                lambda x: x.lower().replace('_', ''))

        self.df_human['body_string'] = \
            self.df_human['tokens_in_code'].apply(
                lambda token_list:
                    "".join([t['text'].lower() for t in token_list]))

    def _normalize_machine_df(self):
        """Join original tokens in the name and create body_string to match."""
        if 'att_vector' in self.df_machine.columns:
            self.df_machine.rename(
                columns={'att_vector': 'att_vector_machine'}, inplace=True)

        if 'original_name' in self.df_machine.columns:
            self.df_machine['original_name'] = \
                self.df_machine['original_name'].apply(
                    lambda x: x.replace(',', ''))

        if 'body_string' in self.df_machine.columns:
            self.df_machine['body_string'] = \
                self.df_machine['body_string'].apply(
                    lambda x: x.lower())

        n_machine_functions = len(self.df_machine)
        logger.info(f'{n_machine_functions} machine functions')

    def _old_get_attention(self, index_token, df_interaction):
        """Compute the attention (total second) for the given token.

        Compute attention for every token aka. count the second each
        token was over."""
        # no dataframe of interactions in case the user subitted without looking
        # at the code
        if len(df_interaction) == 0:
            return 0
        single_token = \
            df_interaction[df_interaction["position"] == index_token]
        # remove clicked (to handle them separately)
        single_token = single_token[single_token["clicked"] == 0]
        # compute fixation on each token
        single_token['fixation'] = single_token['t'].diff()
        # count the time from the event where the mouse leave the token
        single_token = single_token[single_token["over"] == 0]
        attention = np.sum(single_token["fixation"])
        return attention

    def allow_max_length(self, input_list, max_len=150):
        """Allow only vector/list shorter than max, if longer trim it."""
        if (len(input_list) > max_len):
            return input_list[:max_len]
        return input_list

    def compute_correlations(
            self,
            human_attention_col, machine_attention_cols,
            ranking_measures=True,
            overlapping_measures=True,
            also_per_token_kind_restricted=True,
            consider_only_first_n_tokens=None,
            remove_start_end=True
            ):
        """Compute all combinations of correlations metrics and machine."""

        if consider_only_first_n_tokens is not None:
            self.df_intersection_no_dup['att_vector_w_click'] = \
                self.df_intersection_no_dup.apply(
                    lambda row: self.allow_max_length(
                        row['att_vector_w_click'],
                        consider_only_first_n_tokens
                    ),  # APPLY THE CUT
                    axis=1
                )

        for machine_col_name in machine_attention_cols:
            prefix = self._create_column_prefix(machine_col_name)

            if ranking_measures:
                self.df_intersection_no_dup = \
                    self._compute_correlation(
                        df=self.df_intersection_no_dup,
                        human_col=human_attention_col,
                        machine_col=machine_col_name,
                        prefix=prefix,
                        remove_start_end=remove_start_end)

            if overlapping_measures:
                self.df_intersection_no_dup = \
                    self._compute_overlapping(
                        df=self.df_intersection_no_dup,
                        human_col=human_attention_col,
                        machine_col=machine_col_name,
                        prefix=prefix)

            if also_per_token_kind_restricted:
                self.df_intersection_no_dup = \
                    self._compute_correlation_category(
                        df=self.df_intersection_no_dup,
                        human_col=human_attention_col,
                        machine_col=machine_col_name,
                        prefix=prefix)

    def _create_column_prefix(self, machine_col):
        """Derive the prefix based on the name."""
        prefix = ""
        if 'copy' in machine_col.lower():
            prefix += "COPY_"
        if 'regular' in machine_col.lower():
            prefix += "REGULAR_"
        for i in range(8):
            if f'_{i}' in machine_col.lower():
                prefix += f'{i}_'
        if 'avg' in machine_col.lower():
            prefix += "AVG_"
        if 'max' in machine_col.lower():
            prefix += "MAX_"
        if 'transformers' in machine_col.lower():
            prefix += "TRANSF_"
        return prefix

    def _remove_parenthesis(self, scores, tokens):
        return [s for (t, s) in zip(tokens, scores)
                if t['text'] != "{" and ['text'] != "}"]

    def _compute_correlation(self, df, human_col, machine_col,
                             prefix, remove_start_end):
        """Compute the correlation metrics between att weights in 2 columns."""
        df = deepcopy(df)

        def clip_start_end(array, remove_start_end):
            if remove_start_end:
                return array[1:-1]
            return array

        # p at the end stands for p-value
        df[prefix + 'spearman_no_parenthesis'], df[prefix + 'spearman_no_parenthesis_p'] = \
            zip(*df.apply(
                    lambda row:
                    stats.spearmanr(
                        self._remove_parenthesis(
                            np.array(row[human_col]),  # minus to have large weights = rank 1
                            row['tokens_in_code']),
                        self._remove_parenthesis(
                            np.array(clip_start_end(
                                array=row[machine_col],
                                remove_start_end=remove_start_end)),  # minus to have large weights = rank 1
                            row['tokens_in_code'])
                    ),
                    axis=1))

        df[prefix + 'spearman'], df[prefix + 'spearman_p'] = \
            zip(*df.apply(
                    lambda row:
                    stats.spearmanr(
                        np.array(row[human_col]),  # minus to have large weights = rank 1
                        np.array(clip_start_end(
                            array=row[machine_col],
                            remove_start_end=remove_start_end))  # minus to have large weights = rank 1
                    ),
                    axis=1))

        df[prefix + 'kendalltau'], df[prefix + 'kendalltau_p'] = \
            zip(*df.apply(
                    lambda row:
                    stats.kendalltau(
                        np.array(row[human_col]),  # minus to have large weights = rank 1
                        np.array(clip_start_end(
                            array=row[machine_col],
                            remove_start_end=remove_start_end))  # minus to have large weights = rank 1
                    ),
                    axis=1))

        df[prefix + 'pearson'], df[prefix + 'pearson_p'] = \
            zip(*df.apply(
                    lambda row:
                    stats.pearsonr(
                        np.array(row[human_col]),  # minus to have large weights = rank 1
                        np.array(clip_start_end(
                            array=row[machine_col],
                            remove_start_end=remove_start_end))  # minus to have large weights = rank 1
                    ),
                    axis=1))

        #weighting_dictionary = \
        #    {
        #        'Annotation': 1,
        #        'BasicType': 1,
        #        'Boolean': 1,
        #        'DecimalFloatingPoint': 1,
        #        'DecimalInteger': 1,
        #        'Identifier': 10,
        #        'Keyword': 1,
        #        'Modifier': 1,
        #        'Null': 1,
        #        'Operator': 1,
        #        'Separator': 1,
        #        'String': 1,
        #        np.nan: 0,
        #    }

        #df[prefix + 'weightedtau'], df[prefix + 'weightedtau_p'] = \
        #    zip(*df.apply(
        #            lambda row:
        #            stats.weightedtau(
        #                np.array(row[human_col]),  # minus to have large weights = rank 1
        #                np.array(row[machine_col][1:-1]),  # minus to have large weights = rank 1
        #                # kendalltau not weighted
        #                # rank=np.ones(len(row[human_col])).astype('int')
        #                rank=self._get_category_weights(
        #                    row['tokens_in_code'],
        #                    weighting_dictionary)
        #            ),
        #            axis=1))

        # top-k overlapping (intersection over union)

        return df

    def _compute_overlapping(self, df, human_col, machine_col, prefix):
        """Compute the correlation metrics between att weights in 2 columns."""
        df = deepcopy(df)

        df[prefix + 'overlap_perc_top_10'] = \
            df.apply(
                lambda row:
                self._top_k_overlap_perc(
                    row[human_col],
                    row[machine_col][1:-1],
                    k=10),
                axis=1)

        df[prefix + 'overlap_perc_top_10_p'] = \
            df.apply(
                lambda row:
                self._compute_pvalue_for_topk(
                    real_overlapping=row[prefix + 'overlap_perc_top_10'],
                    k=10, n_tokens=row['n_tokens'],
                    expected_avg_overlapping=2.9,
                    comparison_among=2, same_token_in=2),
                axis=1)

        df[prefix + 'overlap_perc_top_5'] = \
            df.apply(
                lambda row:
                self._top_k_overlap_perc(
                    row[human_col],
                    row[machine_col][1:-1],
                    k=5),
                axis=1)

        df[prefix + 'overlap_perc_top_5_p'] = \
            df.apply(
                lambda row:
                self._compute_pvalue_for_topk(
                    real_overlapping=row[prefix + 'overlap_perc_top_5'],
                    k=5, n_tokens=row['n_tokens'],
                    expected_avg_overlapping=1.5,
                    comparison_among=2, same_token_in=2),
                axis=1)

        df[prefix + 'overlap_perc_top_3'] = \
            df.apply(
                lambda row:
                self._top_k_overlap_perc(
                    row[human_col],
                    row[machine_col][1:-1],
                    k=3),
                axis=1)

        df[prefix + 'overlap_perc_top_3_p'] = \
            df.apply(
                lambda row:
                self._compute_pvalue_for_topk(
                    real_overlapping=row[prefix + 'overlap_perc_top_3'],
                    k=3, n_tokens=row['n_tokens'],
                    expected_avg_overlapping=0.8,
                    comparison_among=2, same_token_in=2),
                axis=1)

        df[prefix + 'IoU_top_10'] = \
            df.apply(
                lambda row:
                self._top_k_overlap_over_union(
                    row[human_col],
                    row[machine_col][1:-1],
                    k=10),
                axis=1)

        df[prefix + 'IoU_top_5'] = \
            df.apply(
                lambda row:
                self._top_k_overlap_over_union(
                    row[human_col],
                    row[machine_col][1:-1],
                    k=5),
                axis=1)

        df[prefix + 'IoU_top_3'] = \
            df.apply(
                lambda row:
                self._top_k_overlap_over_union(
                    row[human_col],
                    row[machine_col][1:-1],
                    k=3),
                axis=1)
        return df

    def _top_k_overlap_over_union(self, list_a, list_b, k,
                                  tokens_text=None, content_based=False):
        """Compute the no. of tokens that overlap in the top-k most attended."""
        if (len(list_a) <= k or (len(list_b) <= k)):
            print(f'Less than {k} tokens, we use the the dafult 1 value.')
            return 1

        top_k_idx_a = np.argsort(list_a)[-k:]
        top_k_idx_b = np.argsort(list_b)[-k:]
        #print(top_k_idx_a)
        #print(top_k_idx_b)

        if content_based:
            # compare token content
            # e.g. two occurrences of "function" token will be threated as the same
            if tokens_text is None:
                print('No token for this function.')
                return 0
            top_k_tokens_a = set([tokens_text[i] for i in top_k_idx_a])
            top_k_tokens_b = set([tokens_text[i] for i in top_k_idx_b])
            intersection = len(top_k_tokens_a.intersection(top_k_tokens_a))
            union = len(top_k_tokens_a.union(top_k_tokens_a))
            IoU = float(intersection) / union
        else:
            # compare token positions
            # e.g. two occurrences of "function" token will be threated as separated entities
            top_k_idx_a = set(list(top_k_idx_a))
            top_k_idx_b = set(list(top_k_idx_b))
            intersection = len(top_k_idx_a.intersection(top_k_idx_b))
            union = len(top_k_idx_a.union(top_k_idx_b))
            IoU = float(intersection) / union
        return IoU

    def _top_k_overlap_perc(self, list_a, list_b, k,
                            tokens_text=None, content_based=False):
        """Compute the no. of tokens that overlap in the top-k most attended."""
        if (len(list_a) <= k or (len(list_b) <= k)):
            print(f'Less than {k} tokens, we use the the dafult 1 value.')
            return 1

        top_k_idx_a = np.argsort(list_a)[-k:]
        top_k_idx_b = np.argsort(list_b)[-k:]
        # print(top_k_idx_a)
        # print(top_k_idx_b)

        if content_based:
            # compare token content
            # e.g. two occurrences of "function" token will be threated as the same
            if tokens_text is None:
                print('No token for this function.')
                return 0
            top_k_tokens_a = set([tokens_text[i] for i in top_k_idx_a])
            top_k_tokens_b = set([tokens_text[i] for i in top_k_idx_b])
            overlap = len(top_k_tokens_a.intersection(top_k_tokens_a))
            overlap_perc = float(overlap) / k
        else:
            # compare token positions
            # e.g. two occurrences of "function" token will be threated as separated entities
            top_k_idx_a = set(list(top_k_idx_a))
            top_k_idx_b = set(list(top_k_idx_b))
            elements_intersection = top_k_idx_a.intersection(top_k_idx_b)
            #print(elements_intersection)
            overlap = len(elements_intersection)
            overlap_perc = float(overlap)/ k
        return overlap_perc

    def _get_category_weights(self, tokens, categories_weights):
        """Get the map of weights depending on the category of token.

        For weighted kendal tau."""
        rank_weights =\
            [categories_weights[t['kind']]
             if not (str(t['kind']) == 'nan') else 0
             for t in tokens]
        return rank_weights

    def _keep_category(self, scores, tokens, category):
        #print(scores)
        #print(tokens)
        filter_list = \
            [s for (t, s) in zip(tokens, scores)
            if t['kind'] == category]
        #print(filter_list)
        return filter_list

    def get_token_categories(self):
        return [
            'Annotation',
            'BasicType',
            'Boolean',
            'DecimalFloatingPoint',
            'DecimalInteger',
            'Identifier',
            'Keyword',
            'Modifier',
            'Null',
            'Operator',
            'Separator',
            'String'
        ]

    def _compute_correlation_category(self, df, human_col, machine_col, prefix):
        """Compute the correlation metrics between att weights in 2 columns."""
        df = deepcopy(df)
        # p at the end stands for p-value
        categories = [
            'Annotation',
            'BasicType',
            'Boolean',
            'DecimalFloatingPoint',
            'DecimalInteger',
            'Identifier',
            'Keyword',
            'Modifier',
            'Null',
            'Operator',
            'Separator',
            'String'
        ]

        for category in categories:
            df[prefix + 'spearman_only_' + category.lower()], \
                df[prefix + 'spearman_only_' + category.lower() + '_p'] = \
                zip(*df.apply(
                        lambda row:
                        stats.spearmanr(self._keep_category(row[human_col], row['tokens_in_code'], category),
                                        self._keep_category(row[machine_col][1:-1], row['tokens_in_code'], category)),
                        axis=1))

            df[prefix + 'overlap_perc_top_5_only_' + category.lower()] = \
                df.apply(
                    lambda row:
                    self._top_k_overlap_perc(
                        self._keep_category(row[human_col], row['tokens_in_code'], category),
                        self._keep_category(row[machine_col][1:-1], row['tokens_in_code'], category),
                        k=5),
                    axis=1)

        return df

    def _compute_pvalue_for_topk(
            self,
            real_overlapping,
            k=10, n_tokens=100,
            expected_avg_overlapping=2.9,
            comparison_among=2, same_token_in=2):
        """Compute the p-value for the top-k comparison/correlation."""
        # check if assumptions are met
        if n_tokens < (k * 10):
            # if the tokens are not at least 10 times the ranking
            # therefore T >> r is not satisfied
            return 1  # unreliable data
        if k < (expected_avg_overlapping * 3):
            # if the tokens are not at least 10 times the ranking
            # therefore T >> r is not satisfied
            return 1  # unreliable data

        T = n_tokens  # no. total genes (no.tokens)
        r = k  # k ot the top-k we compare
        p_0 = r / T  # probability of getting a random gene in the ranking top-k
        N = 2  # total studies (total attention maps to compare)
        n = 2  # at least two studies (at least 2 participants should have the same token in top-k)

        # probability that at least two studies have the same gene
        # probability that at least two attention top-k have the same important token
        P_0_2 = 1 - binom.cdf (n-1, N, p_0)

        # number of genes captured in the intersection of 2 or more studies
        # mean of the poisson distribution
        # on average this is the number of overlapping genes we should have among the studies
        # on average this is the number of overlappin tokens we should have among the participants
        test_statistic = P_0_2 * T

        # compute p-value:
        empirical_value = real_overlapping

        p_value_one_side = 1 - poisson.cdf(k=empirical_value,
                                        mu=test_statistic)
        return p_value_one_side

    # model performance

    def precision(self, predicted_tokens, real_tokens):
        predicted_set = set(predicted_tokens)
        real_set = set(real_tokens)
        hits = predicted_set.intersection(real_set)
        return float(len(hits)) / len(predicted_set)

    def recall(self, predicted_tokens, real_tokens):
        predicted_set = set(predicted_tokens)
        real_set = set(real_tokens)
        hits = predicted_set.intersection(real_set)
        return float(len(hits)) / len(real_set)

    def f1(self, precision, recall):
        if (precision + recall) == 0:
            return 0
        return (2 * precision * recall) / (precision + recall)

    def perfect_match(self, predicted_tokens, real_tokens):
        predicted_set = set(predicted_tokens)
        if '%END%' in predicted_set:
            predicted_set.remove('%END%')
        return set(predicted_tokens) == set(real_tokens)

    def compute_model_performance(self, prediction_column_name='predicted_tokens'):
        """Comptute precision, recall, f1-score."""
        df = self.df_intersection_no_dup

        df['perfect_prediction'] = \
            df.apply(
                lambda row: self.perfect_match(row[prediction_column_name], row['function_name_tokens']), axis=1)

        df['prediction_precision'] = \
            df.apply(
                lambda row: self.precision(row[prediction_column_name], row['function_name_tokens']), axis=1)

        df['prediction_recall'] = \
            df.apply(
                lambda row: self.recall(row[prediction_column_name], row['function_name_tokens']), axis=1)

        df['prediction_f1'] = \
            df.apply(
                lambda row: self.f1(row['prediction_precision'], row['prediction_recall']), axis=1)

        bins = [-np.inf, .2, .4, .6, .8, np.inf]
        names = ['(0-0.2]', '[0.2-0.4)',
                 '[0.4-0.6)', '[0.6-0.8)', '[0.8-1]']

        df['f1_category'] = \
            pd.cut(df['prediction_f1'],
                   bins, labels=names, right=True)

        self.df_intersection_no_dup = df


    # VISUALIZATION

    def plot_token_heatmap_side_by_side(self, machine_col,
                                        limit=None, only_functions=None,
                                        only_uuids=None, only_users=None,
                                        columns_to_observe=None,
                                        save_first=0,
                                        human_col='att_vector_human',
                                        out_folder='saved_token_maps'):
        """Plot Human and Machine heatmaps on token side by side."""

        df_comparison = self.df_intersection_no_dup
        if only_functions is not None:
            df_comparison = \
                df_comparison[df_comparison['original_name'].isin(only_functions)]

        if only_uuids is not None:
            print('filter uuid')
            print(only_uuids)
            df_comparison = \
                df_comparison[df_comparison['uuid'].isin(only_uuids)]

        if only_users is not None:
            print('filter randomcode')
            print(only_users)
            #print(df_comparison.columns)
            df_comparison = \
                df_comparison[df_comparison['randomcode'].isin(only_users)]
            df_comparison.drop_duplicates(
                subset='randomcode', keep='first', inplace=True)
            #print(df_comparison[['randomcode', 'time']].astype('string'))

        counter = 0
        old_function = ""
        saved_pictures = 0
        for row in df_comparison.iterrows():
            counter += 1
            if limit is not None and counter > limit:
                break
            idx = row[0]
            record = row[1]
            correct_answered = \
                record['original_name'] == record['functionnamebyuser']

            almost_correct = \
                record['functionnamebyuser'] in [
                    x.lower().replace('_', '')
                    for x in record['options_tfidf']]

            print('*' * 50, ' CODE: ', record['uuid'])

            # plot the machine only at the beginning
            if record['original_name'] != old_function:
                print(f"{record['original_name']} -> {record['predicted_tokens']}")
                [print(f' - {o}') for o in sorted(record['options_nlp'])]
                # machine
                img = self.process_single(tokens=record['tokens_in_code'],
                                    attention=record[machine_col],
                                    human=False,
                                    formattedcode=record['formattedcode'],
                                    formatted_lines=record['formatted_lines'],
                                    almost_correct=None,
                                    correct_answered=correct_answered)

                if saved_pictures < save_first:
                    # create folder if it is not there
                    Path(out_folder).mkdir(parents=True, exist_ok=True)
                    out_img_name = out_folder + '/' + str(record['uuid']) + '_' + str(saved_pictures) + '.png'
                    img.save(out_img_name, 'PNG', compress_level=0)
                    saved_pictures += 1
                plt.show()
            # human
            nickname = record['nickname']
            user_decision = record['functionnamebyuser']
            print(f'{nickname} > {user_decision}')
            if columns_to_observe is not None:
                for c in columns_to_observe:
                    print(f"{c}: {record[c]}")

            img = \
                self.process_single(
                    tokens=record['tokens_in_code'],
                    attention=record[human_col],
                    human=True,
                    formattedcode=record['formattedcode'],
                    formatted_lines=record['formatted_lines'],
                    correct_answered=correct_answered,
                    almost_correct=almost_correct,
                    final_clicked_tokens=record['finalclickedtokens'])
            if saved_pictures < save_first:
                # create folder if it is not there
                Path(out_folder).mkdir(parents=True, exist_ok=True)
                out_img_name = out_folder + '/' + str(record['uuid']) + '_' + str(saved_pictures) + '.png'
                img.save(out_img_name, 'PNG', compress_level=0)
                saved_pictures += 1
            plt.show()
            old_function = record['original_name']

    def process_single(self, tokens, attention, human,
                       formattedcode,
                       formatted_lines,
                       correct_answered, almost_correct,
                       final_clicked_tokens=None):
        """Display attention of the given function."""
        # PREPARE IMAGE
        path_font_file = '../public/FreeMono.ttf'
        surce_code_content = formattedcode #'\n'.join(formatted_lines)
        #img_name = folder + data['id'] + data['rawdictionarykey'][1:] + '.png'

        ratio = (8.4/14)
        char_height = 20
        char_width = char_height * ratio

        # compute max width required
        lines = surce_code_content.splitlines()
        offset = \
            int(tokens[0]['start_char'])
        lines[0] = " " * offset + lines[0]
        #print(lines)
        lines_len = [len(line) for line in lines]
        max_width = int(max(lines_len) * char_width)
        max_height = int(char_height * len(lines))

        img = Image.new('RGB', (max_width, max_height), color=(255, 255, 255))
        fnt = ImageFont.truetype(path_font_file, char_height)
        drw = ImageDraw.Draw(img, 'RGBA')
        drw.text((0, 0), " " * offset + surce_code_content, font=fnt, fill=(0, 0, 0))
        # CAN BE DELAYED AT AFTER TOKEN DRAWING img.save(img_name)

        # check clicked tokens to draw squares around them
        if final_clicked_tokens is not None:
            clicked_tokens = np.array(final_clicked_tokens)
            clicked_tokens_indices = np.where(clicked_tokens == 1)[0].tolist()
        else:
            clicked_tokens_indices = []

        # INSTANTIATE TOKENS
        # get the positon form the metadata of tokens
        viz_tokens = []
        # DEBUG print(tokens)
        # DEBUG print(formattedcode)
        for i, t in enumerate(tokens):
            # print(t)
            new_token = \
                VisualToken(
                    index_id=t['index_id'],
                    text=t['text'],
                    x=char_width * int(t['start_char']),
                    y=char_height * int(t['line']),
                    width=char_width * len(t['text']),
                    height=char_height,
                    clicked=(i in clicked_tokens_indices))
            viz_tokens.append(new_token)

        # COMPUTE ATTENTION
        global_attention = 1
        # compute attention

        for att, viz_token in zip(attention, viz_tokens):
            viz_token.add_attention(att)

        # COMPUTE REFERENCE ATTENTION TO RESCALE
        # sum all the attention received by the tokens
        global_attention = 0
        attentions = []
        for viz_token in viz_tokens:
            attentions.append(viz_token.attention)
        global_attention = max(attentions) * 1.33

        # check user was right to decide the color of the tokens (red vs green)
        # correct_answered decides the color
        for viz_token in viz_tokens:
            # print(token)
            viz_token.draw_PIL(drw, global_attention, correct_answered, human, almost_correct)

        #img.save(img_name)
        #return img_name
        print(type(img))
        imshow(np.asarray(img))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        if human:
            plt.title('Human')
        else:
            plt.title('Machine')
        return img
