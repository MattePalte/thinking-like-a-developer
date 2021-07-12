"""
It formats the Source Code according to the language style.
"""

from pathlib import Path
import os
import subprocess
import logging
import abc
from abc import ABCMeta
from tqdm import tqdm
from .functionbank import AllamanisBank
import re

logging.basicConfig()  # required
logger = logging.getLogger('formatter')
logger.setLevel(logging.INFO)

tqdm.pandas()


class Formatter(ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, functions_code_bank):
        """Pretty format the source code of the functions in the bank."""
        raise NotImplementedError


class JavaFormatter(object):

    def transform(self, functions_code_bank, limit=None):
        """Pretty format the source code of the functions in the bank."""
        self._prepare_tmp_folder()
        df_input = functions_code_bank.to_dataframe()
        if limit is not None:
            df_input = df_input.head(limit)
        df_input['formatted_lines'] = \
            df_input.progress_apply(
                lambda row:
                self._format_java_code(
                    unformatted_java_code=row.source_code_string), axis=1)
        return AllamanisBank.from_dataframe(df_input)

    def _format_java_code(self, unformatted_java_code):
        self._prepare_input_file(unformatted_java_code)
        self._run_astyle_from_shell()
        beautiful_lines = self._read_astyle_output_python()
        # Handle corner-case if there is throw Exception in the title
        start_line_of_function = 0
        try:
            # remove tokens until the first open bracket line
            start_line_of_function = beautiful_lines.index("{")
        except ValueError:
            logger.debug('Handling exception first { not found')
            # check if the function starts with an exception
            for i in range(len(beautiful_lines)):
                line = beautiful_lines[i]
                if re.search('(?<=throws) [a-zA-z_$0-9]+Exception ', line) is not None:  # noqa
                    start_line_of_function = i
                    beautiful_lines[i] = "{"
                    logger.debug('Method with exception in the signture')
                    break
        beautiful_lines = beautiful_lines[start_line_of_function:]
        return beautiful_lines

    def _prepare_tmp_folder(self):
        """Prepare a tmp folder."""
        current_path = os.getcwd()
        self.tmp_folder_path = os.path.join(current_path, 'tmp_astyle')
        Path(self.tmp_folder_path).mkdir(parents=True, exist_ok=True)

    def _prepare_input_file(self, unformatted_java_code):
        self.in_file_path = \
            os.path.join(self.tmp_folder_path, 'tmp_in_file.java')
        with open(self.in_file_path, 'w') as in_file:
            in_file.write(str(unformatted_java_code))

    def _run_astyle_from_shell(self):
        self.out_file_path = \
            os.path.join(self.tmp_folder_path, 'tmp_out_file.java')
        process = \
            subprocess.Popen([
                'astyle',
                '--style=allman', '--pad-param-type',
                '--max-code-length=80', '--break-after-logical',
                '--style=bsd', '-A1', '--style=break', '--pad-oper',
                '--stdin=' + self.in_file_path,
                '--stdout=' + self.out_file_path],
                stdout=subprocess.PIPE,
                universal_newlines=True)

        output, errors = process.communicate()
        return_code = process.poll()
        if return_code is not None:
            logger.debug('Astyle command return code: ' + str(return_code))
        # while True:
        #     output = process.stdout.readline()
        #     return_code = process.poll()
        #     if return_code is not None:
        #         logger.debug('Astyle command return code: ' + str(return_code))
        #         # Process has finished, read rest of the output
        #         for output in process.stdout.readlines():
        #             logger.debug(output.strip())
        #         break

    def _read_astyle_output_cat(self):
        process = \
            subprocess.Popen([
                'cat', self.out_file_path],
                stdout=subprocess.PIPE,
                universal_newlines=True)

        while True:
            return_code = process.poll()
            if return_code is not None:
                logger.debug('Read file via cat ret code: ' + str(return_code))
                # Process has finished, read rest of the output
                beautiful_lines = process.stdout.readlines()
                beautiful_lines = \
                    [line.rstrip('\n') for line in beautiful_lines]
                # for line in beautiful_lines:
                #     print(line)
                break
        return beautiful_lines

    def _read_astyle_output_python(self):
        with open(self.out_file_path, 'r') as in_file:
            lines = in_file.readlines()
        beautiful_lines = \
            [line.rstrip('\n') for line in lines]
        return beautiful_lines
