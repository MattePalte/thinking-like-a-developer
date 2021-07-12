"""Interface to use the JavaParse CoDL tool."""

import json
import logging
import os
import shutil
import re
from tqdm import tqdm
import abc
from abc import ABCMeta
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from pathlib import Path
import time
import subprocess


logging.basicConfig()  # required
logger = logging.getLogger('codl-tool')
logger.setLevel(logging.INFO)


class CodlTool(object):

    def __init__(self, jar_path):
        self.jar_path = jar_path

    def load(self, source_files_folder, only_filenames=None):
        self.source_files_folder = source_files_folder
        if only_filenames is not None:
            self.only_filenames = only_filenames
        else:
            self.only_filenames = os.listdir(self.source_files_folder)
            # keep only java files
            java_regex = re.compile('(?!\.java)')
        n_source_files = len(self.only_filenames)
        logger.info(f' {n_source_files} source files to be inspected.')
        if n_source_files == 0:
            raise ValueError("Empty Folder - no source files: {source_files_folder}")
        self.tmp_file_list = os.path.join(os.getcwd(), "tmp_codl_filelist.txt")
        with open(self.tmp_file_list, 'w') as icodl_txt:
            for filename in self.only_filenames:
                full_path = os.path.join(self.source_files_folder, filename)
                icodl_txt.write(full_path + '\n')
        icodl_txt.close()
        logger.info(f' Tmp list saved here:\n {self.tmp_file_list}')

    def get_bank_folder(self, folder, force_overriding=False):
        """Load bank from a previous codl run folder or run now if empty."""
        # check if already tokenized
        # create a folder for all ICODL results
        self.out_folder = folder
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)
        n_files_already_present = len(os.listdir(self.out_folder))
        if n_files_already_present != 0:
            logger.info(f' Folder already contains {n_files_already_present} files.')
            if force_overriding and os.path.isdir(self.out_folder):
                logger.info(f' Overriding files with a new Codl run.')
                shutil.rmtree(self.out_folder)
            else:
                logger.info(f' Using previously created files.')
                return self.out_folder
        logger.info(f' Codl file tokens creation...')
        self._run_codl_from_shell(
            abs_jar_path=self.jar_path,
            abs_tmp_file_list=self.tmp_file_list,
            abs_out_folder=self.out_folder
        )
        return self.out_folder

    def _run_codl_from_shell(self,
                             abs_jar_path,
                             abs_tmp_file_list,
                             abs_out_folder):
        """Run the jar codl file fro mthe shell."""
        process = \
            subprocess.Popen([
                'java',
                '-jar',
                abs_jar_path,
                '-what',
                'tokensPerFunctionWithSpaces',
                '-files',
                abs_tmp_file_list,
                '-outdir',
                abs_out_folder],
                stdout=subprocess.PIPE,
                universal_newlines=True)

        output, errors = process.communicate()
        return_code = process.poll()
        if return_code is not None:
            logger.info(' CODL command return code: ' + str(return_code))
