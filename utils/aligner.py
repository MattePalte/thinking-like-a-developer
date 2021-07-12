"""
Align the formatted lines and single tokens.

Extract the coordinates (start_char, line) of every token in the
formatted lines.
"""

import re
import logging
from tqdm import tqdm
from .functionbank import AllamanisBank
from .functionbank import TokenInCode


logging.basicConfig()  # required
logger = logging.getLogger('aligner')
logger.setLevel(logging.INFO)


class Aligner(object):

    def __init__(self):
        pass

    def transform(self, functions_code_bank):
        """Find position for every token."""
        df_input = functions_code_bank.to_dataframe()
        df_input['tokens_in_code'] = \
            df_input.progress_apply(
                lambda row:
                self._produce_tokens(
                    function_name=row.function_name,
                    tokens=row.tokens_list,
                    id_same_identifier_list=row.id_same_identifier_list[1:-1],
                    beautiful_lines=row.formatted_lines), axis=1)
        logger.info('Alignment completed')
        return AllamanisBank.from_dataframe(df_input)

        # for function in tqdm(functions_code_bank):
        #     beautiful_lines = function.get_formatted_lines()
        #     clean_beautiful_lines = [
        #         re.sub(f'(?<=[^a-zA-Z_$0-9]){function.function_name}(?=[^a-zA-Z_$0-9])', '%SELF%', line)  # noqa
        #         for line in beautiful_lines
        #     ]
        #     tokens_in_code = self._augment_tokens(
        #         filtered_original_tokens=function.get_tokens(),
        #         id_same_identifier_list=function.id_same_identifier_list,
        #         beautiful_lines=clean_beautiful_lines)
        #     function.set_tokens_in_code(tokens_in_code)
        # logger.info('Alignment completed')

    def _produce_tokens(self,
                        function_name, tokens,
                        id_same_identifier_list,
                        beautiful_lines):
        print('_produce_tokens: ', id_same_identifier_list)
        clean_beautiful_lines = [
            re.sub(f'(?<=[^a-zA-Z_$0-9]){function_name}(?=[^a-zA-Z_$0-9])', '%SELF%', line)  # noqa
            for line in beautiful_lines
        ]
        tokens_in_code = self._augment_tokens(
            filtered_original_tokens=tokens,
            id_same_identifier_list=id_same_identifier_list,
            beautiful_lines=clean_beautiful_lines)
        return tokens_in_code

    def _get_position_of(self, query_token, all_lines,
                         start_char=0, start_line=1):
        """Locate (start_char, line) of the given token in the lines passed."""
        query_token = query_token.lower()
        line_index = start_line
        current_line = all_lines[line_index]
        line_to_search = \
            re.sub("(?<=[^\"'])\\\\n(?=[^\"'])", ' ',
                   current_line[start_char:])
        current_char = line_to_search.lower().find(query_token)
        # print(current_char)
        while ((current_char < 0) and (line_index < len(all_lines) - 1)):
            start_char = 0
            line_index += 1
            current_line = all_lines[line_index]
            line_to_search = \
                re.sub("(?<=[^\"'])\\\\n(?=[^\"'])", ' ',
                       current_line[start_char:])
            current_char = line_to_search.lower().find(query_token)
        query_first_char = current_char + start_char
        return query_first_char, line_index, current_line

    def _augment_tokens(self,
                        filtered_original_tokens,
                        id_same_identifier_list,
                        beautiful_lines,
                        verbose=False):
        """Augment tokens with position in the beautiful lines coordinates."""
        if ((filtered_original_tokens is None) or
                (len(filtered_original_tokens) == 0) or
                (id_same_identifier_list is None) or
                (len(id_same_identifier_list) == 0)):
            return None
        # EXTRACT NEW TOKEN OBJECT WITH POSITIONS
        token_objects = []
        start_char = 0
        start_line = 0  # skip the function declaration
        print('_augment_tokens: ', id_same_identifier_list)
        for i, (id_same_identifier, t) in enumerate(zip(id_same_identifier_list, filtered_original_tokens)):  # noqa
        # for i, t in enumerate(filtered_original_tokens):
            if verbose:
                logger.debug('*' * 50)
            if verbose:
                logger.debug(f'Search of: |{t}|')
                logger.debug('Raw')
                logger.debug(t)

            # search for the token
            first_char_position, line_found, raw_line = \
                self._get_position_of(
                    query_token=t,
                    all_lines=beautiful_lines,
                    start_char=start_char,
                    start_line=start_line)
            # save if found
            if first_char_position != -1:
                if verbose:
                    logger.debug('Found: ' + first_char_position + line_found)
                if verbose:
                    logger.debug(f'| {raw_line} |')
                # create new token
                new_token_object = TokenInCode(
                    index_id=i,
                    token_group=id_same_identifier,
                    text=t,
                    start_char=first_char_position,
                    line=line_found)
                print(new_token_object.__dict__)
                # add it
                token_objects.append(new_token_object.__dict__)
                # increment the start
                start_char = first_char_position + len(t)
                start_line = line_found
            else:
                break
        try:
            last_added_token = token_objects[-1]['text']
        except IndexError:
            logger.debug('beautiful_lines: ')
            logger.debug(beautiful_lines)
            return None
        if (last_added_token != '}'):
            logger.debug(f'\nLast token: {last_added_token}')
            logger.debug(f'ERROR: Check function at position: {i}')
            return None
        else:
            logger.debug("Matched body of method")
            return token_objects
