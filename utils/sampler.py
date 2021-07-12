"""
It extracts groups of functions to be evaualted by users.
It makes sure that the functions are seen by 5 developers as soon as possible.
"""

import json
import pandas as pd
import numpy as np
import os
import re
import random
import logging
from tqdm import tqdm
from pathlib import Path


logging.basicConfig()  # required
logger = logging.getLogger('sampler')
logger.setLevel(logging.INFO)


class Sampler(object):

    def __init__(self, per_experiment,
                 n_warmup_functions,
                 min_people_per_function,
                 functions_code_bank,
                 seed=42):
        self.seed = seed
        self.per_experiment = per_experiment
        self.n_warmup_functions = n_warmup_functions
        self.min_people_per_function = min_people_per_function
        self.functions_code_bank = functions_code_bank
        self.n_total_functions = len(self.functions_code_bank[:])

    def produce(self, n_experiments, out_folder=None):
        """Produce JSON fles, one for each set of functions.
        Note that there are no duplicate functions in the same set."""

        # CREATE BIG LIST
        input_list = list(range(self.n_total_functions))
        counters = np.zeros(len(input_list))

        # HOW MUCH WE WANT TO FAVOR FUNCTIONS EARLY IN THE LIST
        # THE CLOSER TO 1 THE MORE IT TENDS TO PRESENT SAMPLE
        # ALREADY SEEN BY OTHERS
        DECAY = 0.5

        np.random.seed(self.seed)
        random.seed(self.seed)
        bag_of_experiments = []
        for i in range(n_experiments):
            logger.debug('*' * 50)
            logger.debug(f'USER {i}')
            extracted_indices, counters = \
                self._extract(
                    n_elements=self.n_total_functions,
                    n_warmup_functions=self.n_warmup_functions,
                    counters=counters,
                    threshold=self.min_people_per_function,
                    n_extractions=self.per_experiment,
                    decay=DECAY)
            logger.debug('EXTRACTION RESULT INDICES')
            logger.debug(extracted_indices)
            logger.debug('CONVERT INDICES INTO REAL OBJECT...')
            logger.debug('UPDATED COUNTERS')
            logger.debug(counters)
            bag_of_experiments.append(extracted_indices)
        # convert arrays into list
        bag_of_experiments = \
            [list(exp) for exp in bag_of_experiments]

        if (out_folder is None):
            logger.info("No json file saves, " +
                        "because out_folder param is empty.")
        else:
            logger.info(f'Making sure that folder {out_folder} exists.')
            out_folder_path = os.path.join(os.getcwd(), out_folder)
            Path(out_folder_path).mkdir(parents=True, exist_ok=True)
            logger.info('Saving experiment set files in folder ' +
                        f'at path {out_folder_path}')
            self.functions_code_bank.pre_shuffle(shuffle_seed=self.seed)
            for i, exp_set_indices in enumerate(bag_of_experiments):
                exp_set_filename = 'experiment_' + str(i).zfill(7) + '.json'
                exp_set_path = os.path.join(out_folder_path, exp_set_filename)
                self.functions_code_bank.save_json(
                    exp_set_path,
                    indices=exp_set_indices,
                    from_suffled=True)
        return bag_of_experiments

    def _extract(self, n_elements, n_warmup_functions,
                 counters, threshold, n_extractions, decay):
        """Extract a set of indices with a given probability."""
        input_list = np.arange(n_elements)

        pre_exponentiation_list = np.arange(n_elements)
        # zero the already extracted values
        inidices_not_to_extract = tuple(np.where(counters >= threshold))
        pre_exponentiation_list[inidices_not_to_extract] = 0
        #print('pre_exponentiation_list after zero: ', pre_exponentiation_list[:40])
        # rescale values so that the smallest is 1
        # get the smallest (non-zero!) value
        # bring 0s artificially to max value
        pre_exponentiation_list[inidices_not_to_extract] = np.max(pre_exponentiation_list)
        min_value = np.min(pre_exponentiation_list)
        pre_exponentiation_list = pre_exponentiation_list - (min_value - 1)
        # bring the already extracted back to 0 (since we just got them negative)
        pre_exponentiation_list[inidices_not_to_extract] = 0
        #print('pre_exponentiation_list after min subtract: ', pre_exponentiation_list[:40])

        # create probabilities exponential decay
        # so that it is more probable to select elements from the head
        # esponential decay y = a * (1 - b) * x
        max_prob = 1
        exp_list = np.array([max_prob * ((1 - decay)**e) for e in pre_exponentiation_list])

        # remove elements that have counter above thresholds
        # aka we already extracted them the required number of times
        inidices_not_to_extract = tuple(np.where(counters >= threshold))
        exp_list[inidices_not_to_extract] = 0

        total_value = sum(exp_list)
        probab_list = np.array([e/total_value for e in exp_list])
        logger.debug('Eponential probabilities (after normalization)')
        logger.debug(probab_list)
        # extract indices
        logger.debug('non zero: ' + str(np.count_nonzero(probab_list)))
        extracted_indices = \
            np.random.choice(input_list, n_extractions + n_warmup_functions,
                             p=probab_list, replace=False)
        logger.debug(extracted_indices)
        indices_to_consider = extracted_indices[:n_extractions]
        logger.debug(indices_to_consider)
        # update counters
        counters[indices_to_consider] = counters[indices_to_consider] + 1
        # reorder so that the wormup are at the beginning
        warmup_first_extracted_indices = extracted_indices[n_extractions:]
        warmup_first_extracted_indices = \
            np.concatenate((extracted_indices[-n_warmup_functions:], extracted_indices[:n_extractions]))
        logger.info(warmup_first_extracted_indices)
        return warmup_first_extracted_indices, counters
