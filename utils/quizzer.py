"""
This class is responsible for creting the question to the user.
Given a group of functions as FunctionsBank objects
It prepares:
- question (optional and encoded as an integer code
            #TODO create a file with questions and code
            # e.g. select the correct name among the alternatives?
            # e.g. can this token be part of the function name?)
- list of options
- correct answer
- metdata about the options
  (e.g. which one was from knn tf-idf and which random)
  This part is quite arbitrary
"""

import logging
from tqdm import tqdm
from .functionbank import AllamanisBank

import abc
from abc import ABCMeta
# SKLEARN
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import numpy as np
from annoy import AnnoyIndex
import random
from copy import deepcopy

logging.basicConfig()  # required
logger = logging.getLogger('aligner')
logger.setLevel(logging.INFO)


class Quizzer(metaclass=ABCMeta):

    def __init__(self):
        pass

    def fit(self, functions_code_bank):
        self.df_functions = functions_code_bank.to_dataframe()
        n_functions = len(self.df_functions)
        self.prepare_internals()

    def transform(self, functions_code_bank):
        """Add options to the functions."""
        self.df_input = functions_code_bank.to_dataframe()
        df_output = self.prepare_questions(self.df_input)
        return AllamanisBank.from_dataframe(df_output)

    @abc.abstractmethod
    def prepare_internals(self):
        """Prepare internal models."""
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_questions(self, df_input):
        """Create a question with option for every function."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_single_question(self, single_function):
        """Create a single question."""
        raise NotImplementedError


class AlternativeNamesQuizzer(Quizzer):

    def __init__(self, random_options=3, tf_idf_options=3, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.random_options = int(random_options)
        self.tf_idf_options = int(tf_idf_options)

    def prepare_internals(self):
        """Prepare KNN model of function names similarity."""
        # DROP DUPLICATES NAMES
        self.df_functions = \
            self.df_functions.drop_duplicates(
                subset='function_name',
                keep="first")
        nr_unique_function_names = len(self.df_functions)
        logger.info(f"Nr unique function names: {nr_unique_function_names}")
        # NLP PREPROCESSING
        text_corpus = list(self.df_functions['function_name_tokens'])
        text_corpus = \
            [" ".join(nlp_tokens) for nlp_tokens in text_corpus]
        original_corpus = list(self.df_functions['function_name'])
        self.text_corpus = deepcopy(text_corpus)
        self.original_corpus = deepcopy(original_corpus)
        self.mapping_to_nlp = \
            {k: v for k, v in zip(self.original_corpus, self.text_corpus)}
        # PREPARE BAG OF WORDS TF-IDF
        pipe = Pipeline([('count', CountVectorizer()),
                        ('tfid', TfidfTransformer())]).fit(text_corpus)
        pipe['count'].transform(text_corpus).toarray()
        tf_idf_matrix = pipe.transform(text_corpus)
        # PREPARE ANNOY MODEL
        functions_matrix = tf_idf_matrix.toarray()
        nr_items = functions_matrix.shape[0]
        dimension = functions_matrix.shape[1]
        self.model = AnnoyIndex(dimension, 'angular')
        # Length of item vector that will be indexed
        for i in range(nr_items):
            v = functions_matrix[i]
            self.model.add_item(i, v)
        self.model.build(10)  # 10 trees

    def prepare_questions(self, df_input):
        """Create a question with option for every function."""
        df = df_input
        # self.df_functions['options'] = \
        #     self.df_functions.progress_apply(
        #         lambda row:
        #         self._format_java_code(
        #             unformatted_java_code=row.source_code_string), axis=1)

        # df_a['dashy'], df_a['slashy'] = \
        #   zip(*df_a.apply(lambda row: elaborate(ident=row.ident, name=row.name), axis=1))

        df['option_correct'], df["options_random"], df["options_tfidf"], \
            df["options"] = \
            zip(*df.apply(
                lambda row:
                self.generate_single_question(
                    query_function_original=row.function_name,
                    query_function_nlp_tokens=row.function_name_tokens),
                    axis=1))
        df["options_nlp"] = \
            df["options"].apply(
                lambda options: self._get_nlp_version(options=options))
        # df['textcol'].apply(lambda s: pd.Series({'feature1':s+1, 'feature2':s-1}))
        # https://stackoverflow.com/a/16242202

        return df

    def _get_nlp_version(self, options):
        """Get the npl tokenized version of the function names."""
        return [self.mapping_to_nlp[o] for o in options]

    def generate_single_question(self,
                                 query_function_original,
                                 query_function_nlp_tokens):
        """Create a single question."""
        correct_option = query_function_original
        current_function_name = " ".join(query_function_nlp_tokens)
        # TFIDF
        extracted = \
            self.get_similar_names(
                query_function_name=current_function_name,
                model=self.model,
                text_corpus=self.text_corpus,
                original_corpus=self.original_corpus,
                nr_neighbors=self.tf_idf_options + 1)
        # keep only neighbors that are not the equal to the correct option
        knn_minus_right = list(set(extracted) - set([correct_option]))
        random.shuffle(knn_minus_right)
        otpions_tfidf = list(knn_minus_right[:self.tf_idf_options])
        # RANDOM
        different_names = deepcopy(self.original_corpus)
        different_names.remove(correct_option)
        for opt in otpions_tfidf:
            different_names.remove(opt)
        extracted = \
            random.sample(different_names,
                          self.random_options)
        # remove overlapping with knn neighbors
        not_overlapping_extracted = \
            list(set(extracted) - set(otpions_tfidf) - set([correct_option]))
        options_random = not_overlapping_extracted[:self.random_options]
        # ALL OPTIONS
        all_options = \
            list(set([correct_option]) | set(otpions_tfidf) |
                 set(options_random))
        return correct_option, options_random, otpions_tfidf, all_options

    def get_similar_names(self, query_function_name, model,
                          text_corpus, original_corpus, nr_neighbors):
        """Get the nearest neighbors of the function passed."""
        position_in_corpus = text_corpus.index(query_function_name)
        neighbors = \
            self.model.get_nns_by_item(position_in_corpus, nr_neighbors)
        return list(np.take(np.array(original_corpus), neighbors))
