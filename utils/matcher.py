"""
It stores a CodeBank collection and matched with an AllamanisBank.

This focuses only on telling if there is a correspondence in the CoDL dataset,
it does not compose the two.
"""
import re
import math
import pandas as pd
import logging
import regex
from .functionbank import AllamanisBank

logging.basicConfig()  # required
logger = logging.getLogger('matcher')
logger.setLevel(logging.INFO)


class Matcher(object):

    def __init__(self, max_tolerance=0):
        """Match the original source code with allamanis tokenization.
        max_tolerance: int
            nr of character that can be substituted between source and target
            in the same token. Fuzzy matching from regex library.
            To disable it leave the default 0 value, aka only precise tokens
            will be matched.
        """
        self.max_tolerance = max_tolerance
        pass

    def fit(self, original_code_bank):
        self.df_original = original_code_bank.to_dataframe()
        self.df_original = \
            self.df_original[['project_name', 'file_name',
                              'function_name', 'source_code_string']]
        self.df_original.drop_duplicates(inplace=True)
        n_functions = len(self.df_original)
        logger.info(f'The input bank contained {n_functions} ' +
                    'original function bodies')

    def transform(self, functions_code_bank):
        self.df_functions = functions_code_bank.to_dataframe()
        n_functions_to_match = len(self.df_functions)
        df_merged = \
            self.df_functions.merge(
                self.df_original,
                how='left', on=['project_name', 'file_name', 'function_name'])
        df_merged['compatible'] = \
            df_merged.apply(lambda row: self._check_compatibility(
                filtered_tokens=row.filtered_original_tokens,
                original_source_code=row.source_code_string_y), axis=1)
        df_only_compatible = df_merged[df_merged['compatible']].copy()
        df_only_compatible.rename(
            columns={'source_code_string_y': 'source_code_string',
                     'tokens_list': 'to_drop_tokens_list',
                     'filtered_original_tokens': 'tokens_list'},
            inplace=True)
        n_functions_matched = len(df_only_compatible)
        logger.info(f'Total of {n_functions_matched}/{n_functions_to_match} ' +
                    'functions matched')
        logger.warning('Handle the ramaining functions with no original code')
        return AllamanisBank.from_dataframe(df_only_compatible)

    def _check_compatibility(self, filtered_tokens, original_source_code):
        """Check that the original source code contains all allamanis tokens."""
        current_char = 0
        # make everything lowecase
        original_source_code = str(original_source_code).lower()
        # handle corner case '\n'
        original_source_code = \
            re.sub("(?<=[^\"'])\\\\n(?=[^\"'])", ' ', original_source_code)
        for t in filtered_tokens:
            query_token = t.lower()
            # next_char = original_source_code[current_char:].find(query_token)
            string_query = \
                '(%s){s<=%i}' % (re.escape(query_token), self.max_tolerance)
            # print('string_query: ', string_query)
            # print('search in: ', original_source_code[current_char:])
            res = \
                regex.search(r''+string_query,
                             original_source_code[current_char:])
            if res is not None:
                next_char = res.span()[0]
            else:
                next_char = -1
            if next_char == -1:  # token not found
                return False
            current_char = current_char + next_char + len(query_token)
        return True

