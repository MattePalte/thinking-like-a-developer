"""
It stores the human logs and converts them to attention weights.
"""

import json
import csv
import logging
import os
import glob
import re
from tqdm import tqdm
import abc
from abc import ABCMeta
import pandas as pd
import numpy as np
import random
from copy import deepcopy
import yaml
from time import time
import shutil
from pathlib import Path
import subprocess
import numpy as np
import boto3  # amazon access
from pandarallel import pandarallel
pandarallel.initialize()

logging.basicConfig()  # required
logger = logging.getLogger('human-log-bank')
logger.setLevel(logging.INFO)


class HumanLogBank(object):

    def __init__(self, data, df_scored_users=None, df_human_logs_data=None):
        self.human_logs_data = data
        self.df_scored_users = df_scored_users
        self.df_human_logs_data = df_human_logs_data

    @classmethod
    def from_folder_of_single_file(cls, eventlogs_folder, eventlogs_files):
        """Load the human logs from a folder of singel human log json."""
        data_group = []
        for filename in eventlogs_files:
            with open(os.path.join(eventlogs_folder, filename), 'r') as in_file:  # noqa
                data = json.load(in_file)
                data_group += data
        return cls(data=data_group)

    @classmethod
    def from_mongodbexport_file(cls, abs_path):
        """Load the human logs from mongodbexport dump of a collection."""
        with open(abs_path, 'r') as in_file:  # noqa
            data = json.load(in_file)
            return cls(data=data)

    @classmethod
    def from_continuos_mongodbexport_file(cls, abs_path):
        """Load the human logs from mongodbexport dump of a collection.

        This contains multiple submission for the same questions."""
        df_mongodb_collection = \
            pd.read_json(abs_path, orient='records')
        df_output = cls._aggregate(df_mongodb_collection)
        data = df_output.to_dict(orient='records')
        return cls(data=data)

    @classmethod
    def from_mongodbexport_via_settings(cls, setting_file_path, folder_path):
        """Load the human logs from mongodbexport as specified in settings

        The attirbute collection is searched for in the ymal config file."""
        with open(setting_file_path, 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
        collection = config_dict['mongodb_atlas']['collection']
        collection_rating_key = config_dict['mongodb_atlas']['collection_rating']
        raw_data_path = os.path.join(
            folder_path,
            "most_recent_" + collection + ".json")
        rating_data_path = os.path.join(
            folder_path,
            "most_recent_" + collection_rating_key + ".json")
        new_humanbank = cls.from_continuos_mongodbexport_file(
            abs_path=raw_data_path)
        new_humanbank.enrich_with_ratings(
            path_dataframe_json=rating_data_path)
        cols = new_humanbank.to_dataframe().columns
        logger.info(' Raw columns available:')
        [logger.info(" - " + c) for c in cols]
        return new_humanbank

    def enrich_with_ground_truth(self, path_dataframe_json):
        """Enrich the human log information with the database one."""
        if self.df_human_logs_data is None:
            self.df_human_logs_data = self.to_dataframe()

        col_before = set(self.df_human_logs_data.columns)
        df_big_experiment = pd.read_json(path_dataframe_json, orient='records')
        self.df_human_logs_data = \
            self.df_human_logs_data.merge(
                on='uuid', how='inner', right=df_big_experiment)
        col_after = set(self.df_human_logs_data.columns)
        logger.info(' New columns added:')
        [logger.info(f' - {x}') for x in col_after - col_before]

    def enrich_with_ratings(self, path_dataframe_json):
        """Enrich the human log information with the ratings."""
        if self.df_human_logs_data is None:
            self.df_human_logs_data = self.to_dataframe()
        df_ratings = pd.read_json(path_dataframe_json, orient='records')
        # remove all the records without a valid index to convert in float
        admissible_indices = \
            [str(e) for e in range(20)]
        before_len = len(self.df_human_logs_data)
        self.df_human_logs_data = \
            self.df_human_logs_data[self.df_human_logs_data['id'].isin(admissible_indices)]
        after_len = len(self.df_human_logs_data)
        logger.info(f'We dropped {before_len-after_len} records due to incorrect indices.')
        self.df_human_logs_data['fileindex'] = \
            self.df_human_logs_data['id'].astype('float')
        df_ratings.dropna(subset=['fileindex'], inplace=True)
        df_ratings['fileindex'] = df_ratings['fileindex'].astype('float')
        self.df_human_logs_data = \
            self.df_human_logs_data.merge(
                on=['randomcode', 'fileindex'], how='inner',
                right=df_ratings[['randomcode', 'fileindex', 'rating']])

    def enrich_with_amturk_user_flag(
            self,
            path_amturk_credentials,
            api_endpoint='https://mturk-requester.us-east-1.amazonaws.com',
            region_name='us-east-1',
            assignment_ids_list=[]):
        """Add a column is_amtruk_user."""
        df_credentials = pd.read_csv(path_amturk_credentials)
        aws_access_key_id = df_credentials.iloc[0]['Access key ID']
        aws_secret_access_key = df_credentials.iloc[0]['Secret access key']
        mturk = boto3.client(
            'mturk',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=api_endpoint
        )
        randomcodes_amturk = []
        next_token = None
        for assignment_id in assignment_ids_list:
            while True:
                print(assignment_id)
                if next_token is None:
                    # first call
                    response = mturk.list_assignments_for_hit(
                        HITId=assignment_id,
                        MaxResults=100
                    )
                else:
                    # subsequent calls
                    response = mturk.list_assignments_for_hit(
                        HITId=assignment_id,
                        MaxResults=100,
                        NextToken=next_token
                    )
                df_amturk_api_submissions = pd.DataFrame(response['Assignments'])
                if len(df_amturk_api_submissions) == 0:
                    # exit if no assignemnts left - go to next HIT id batch
                    next_token = None
                    break
                df_amturk_api_submissions['Answer.surveycode'] = \
                    df_amturk_api_submissions['Answer'].apply(
                        lambda text:
                            re.search('<FreeText>(.+?)</FreeText>', text).group(1))
                new_codes = \
                    list(df_amturk_api_submissions['Answer.surveycode'])
                randomcodes_amturk += new_codes
                if 'NextToken' not in response.keys():
                    # exit if no next token
                    next_token = None
                    break
                else:
                    next_token = response['NextToken']
        logger.info(f'Total of {len(randomcodes_amturk)} amturk submissions')
        # add new column to the general df
        self.df_human_logs_data['is_amtruk_user'] = \
            self.df_human_logs_data['randomcode'].isin(randomcodes_amturk).astype('int')
        self.df_scored_users['is_amtruk_user'] = \
            self.df_scored_users['Answer.surveycode'].isin(randomcodes_amturk).astype('int')
        n_correspondence_in_data = self.df_scored_users['is_amtruk_user'].sum()
        logger.info(f'Total of {n_correspondence_in_data} amturk submissions inserted a code with a correspondence in my user log.')
        n_sessions = len(self.df_scored_users['is_amtruk_user'])
        logger.info(f'Total of {n_sessions} submissions (including partial submissions).')

    def enrich_with_amturk_user_flag_from_csv(
            self,
            path_amturk_user_info):
        """Add a column is_amtruk_user."""
        col_before = set(self.df_human_logs_data.columns)
        df_user_info = pd.read_csv(path_amturk_user_info)
        self.df_human_logs_data = pd.merge(
            left=self.df_human_logs_data, right=df_user_info, on='randomcode')
        col_after = set(self.df_human_logs_data.columns)
        logger.info(' New columns added:')
        [logger.info(f' - {x}') for x in col_after - col_before]

    def parse_attention(self, attentioner):
        """Parse the attention as defined in the attentioner."""
        self.df_human_logs_data = \
            attentioner.parse_attention(self.df_human_logs_data)

    def to_dataframe(self):
        """Convert the human logs into rows of a Pandas DataFrame."""
        rows = []
        if self.df_human_logs_data is not None:
            return self.df_human_logs_data
        if (self.human_logs_data is None):
            raise ValueError("Your Bank doesn't contains any human log.")
        for human_log in self.human_logs_data:
            rows.append(human_log)
        df = pd.DataFrame(rows)
        return df

    def save_backup(self, backup_folder='backup_human_bank',
                    static_suffix=None):
        """Save both summary user performance and attention maps."""
        Path(backup_folder).mkdir(parents=True, exist_ok=True)
        if static_suffix is None:
            static_suffix = str(int(time()))
        if (self.df_scored_users is not None):
            filename = 'users_' + static_suffix + '.json'
            score_path = os.path.join(backup_folder, filename)
            self.df_scored_users.to_json(score_path, orient='records')
            logger.info('Total number of unique human logs: ' +
                        f'{len(self.df_human_logs_data)}')
        self.df_human_logs_data = self.to_dataframe()
        if (self.df_human_logs_data is not None):
            if ('att_vector' not in self.df_human_logs_data.columns):
                raise Exception('No attention parsed')
            filename = 'attentions_' + static_suffix + '.json'
            attention_path = os.path.join(backup_folder, filename)
            self.df_human_logs_data.to_json(attention_path, orient='records')
            logger.info('Total number of unique participant sessions: ' +
                        f'{len(self.df_scored_users)}')

    @classmethod
    def from_backup(cls, backup_folder='backup_human_bank',
                    only_most_recent=False):
        """Read both summary user performance and attention maps."""
        dfs_users = []
        dfs_attentions = []
        # read users
        json_users = \
            [os.path.join(backup_folder, fn)
             for fn in os.listdir(backup_folder)
             if fn.startswith('users_')]
        if only_most_recent:
            json_users = [sorted(json_users)[-1]]
        for json_path in json_users:
            df = pd.read_json(json_path, orient='records')
            dfs_users.append(df)
        df_scored_users = pd.concat(dfs_users, ignore_index=True)
        # read attention
        json_attentions = \
            [os.path.join(backup_folder, fn)
             for fn in os.listdir(backup_folder)
             if fn.startswith('attentions_')]
        if only_most_recent:
            json_attentions = [sorted(json_attentions)[-1]]
        for json_path in json_attentions:
            df = pd.read_json(json_path, orient='records')
            dfs_attentions.append(df)
        df_human_logs_data = pd.concat(dfs_attentions, ignore_index=True)
        # log messages
        logger.info(f'Total number of unique human logs: {len(df_human_logs_data)}')
        logger.info(f'Total number of unique participant sessions: {len(df_scored_users)}')
        return cls(
            data=None,
            df_scored_users=df_scored_users,
            df_human_logs_data=df_human_logs_data)

    def get_users_df(self):
        return deepcopy(self.df_scored_users)

    def get_attentions_df(self):
        return deepcopy(self.df_human_logs_data)

    @classmethod
    def _aggregate(cls, df_original):
        """Aggregate the logs from multiple deliveries."""
        # join the lists - trace of events
        df_grouped = df_original.groupby(
                by=['randomcode', 'experimentset', 'id']).agg({
                    'tokeninteraction': 'sum',
                    'mousetrace': 'sum',
                    'optionsinteraction': 'sum',
                    'tracevisibletokens': 'sum',
                    'timeopenpage': 'min',
                    'time': 'max',
                    'tottimemilli': 'max'
                    })
        #  enrich the records - add to each the full list of events
        df_long_lists = df_grouped.reset_index(drop=False)
        different_cols = \
            list(set(df_original.columns)
                 - set(['tokeninteraction', 'mousetrace',
                        'optionsinteraction', 'tracevisibletokens',
                        'timeopenpage', 'time', 'tottimemilli']))
        df_condensed = df_long_lists.merge(
            on=['randomcode', 'experimentset', 'id'],
            right=df_original[different_cols])
        # reorder: last is the most recent
        df_chronological_condensed = \
            df_condensed.sort_values(
                by=['randomcode', 'experimentset', 'id', 'time'])
        # discard duplicates and keep only the most recent submission
        # since it contains the submitted function name by the user
        df_single_complete_log = \
            df_chronological_condensed.drop_duplicates(
                subset=['randomcode', 'experimentset', 'id'],
                keep='last')
        return df_single_complete_log.reset_index(drop=True)

    def compute_user_level_mistakes(self, only_users=[]):
        """Compute if answer is correct.

        only_users: list[str]
            list of randomcodes
        """
        df = self.df_human_logs_data
        col_before = set(self.df_human_logs_data.columns)
        # enrich columns
        df['is_correct'] = \
            (df['option_correct'] == df['functionnamebyuser']).astype('int')
        df['error_on_tfidf'], df['error_on_random'] = \
            zip(*df.apply(lambda row:
                (int(row.functionnamebyuser in row.options_tfidf),
                 int(row.functionnamebyuser in row.options_random)), axis=1))
        df['total_time'] = df['tottimemilli'] / 1000
        df['is_warmup'] = \
            (df['id'].astype('int') < 3).astype('int')
        df['id'] = df['id'].astype('int')
        df['n_mouse_traces'] = \
            df.apply(lambda row: len(row.mousetrace), axis=1)
        df['n_token_interactions'] = \
            df.apply(lambda row: len(row.tokeninteraction), axis=1)
        # ensure no duplicate and chronological order
        df.sort_values(by='time', ascending=True, inplace=True)
        df.drop_duplicates(
            subset=['randomcode', 'uuid'], keep='first', inplace=True
        )
        logger.info(f'Total number of unique human logs: {len(df)}')
        # summarize per user performance
        df_scored_users = \
            df[df['is_warmup'] == 0].groupby(
                by=['nickname', 'randomcode', 'experimentset']).agg({
                    'is_correct': 'sum',
                    'error_on_tfidf': 'sum',
                    'error_on_random': 'sum',
                    'total_time': ['mean', 'median'],
                    'id': 'count',
                    'time': 'mean',
                    'uuid': pd.Series.nunique
                    }).reset_index(drop=False).rename(
                columns={
                    'is_correct': 'n_correct',
                    'error_on_tfidf': 'n_tfidf_mistakes',
                    'error_on_random': 'n_random_mistakes',
                    'total_time': 'avg_time',
                    'id': 'n_total',
                    'uuid': 'n_unique',
                    'randomcode': 'Answer.surveycode'
                })
        # manual rename of median column
        df_scored_users.columns = df_scored_users.columns.droplevel(1)
        occurrences = 0
        new_columns = []
        for c in df_scored_users.columns:
            if occurrences == 1 and c == 'avg_time':
                c = 'median_time'
            if c == 'avg_time':
                occurrences += 1
            new_columns.append(c)
        df_scored_users.columns = new_columns
        df_scored_users['pesudo_correct'] = \
            df_scored_users['n_correct'] + df_scored_users['n_tfidf_mistakes']
        self.df_scored_users = df_scored_users
        self.df_human_logs_data = df
        logger.info('Total number of unique participant sessions: ' +
                    f'{len(self.df_scored_users)}')
        col_after = set(self.df_human_logs_data.columns)
        logger.info(' New columns added:')
        [logger.info(f' - {x}') for x in col_after - col_before]

    def validate(self, strategy):
        """Validate with a validation strategy."""
        df_validated = strategy.validate(self.df_scored_users)
        approved_users = \
            list(df_validated[df_validated['is_valid']]['Answer.surveycode'])
        # add new column to 'from_valid_user'
        self.df_human_logs_data['from_valid_user'] = \
            self.df_human_logs_data['randomcode'].apply(
                lambda code: code in approved_users
            )
        return df_validated

    def drop_records(self, where_col, value):
        """Remove all the records with the given value in the given column."""
        self.df_human_logs_data = \
            self.df_human_logs_data[
                self.df_human_logs_data[where_col] != value
            ]


class CsvAmazonValidator(object):

    def __init__(self, in_amturk_folder):
        list_of_files = glob.glob(os.path.join(in_amturk_folder, '*'))
        # * means all if need specific format then *.csv
        most_recent_csv = max(list_of_files, key=os.path.getctime)
        if 'batch' not in most_recent_csv:
            raise Exception('No amturk found')
        logger.info(f'AMTurk file detected {most_recent_csv}')
        self.df_participants_amturk = pd.read_csv(most_recent_csv)
        self.initial_columns = self.df_participants_amturk.columns
        amazon_turk_participants = \
            self.df_participants_amturk['Answer.surveycode']
        logger.info(f'Total users {len(self.df_participants_amturk)}')

    def output_csv(self, df_users, validator, out_amturk_path):
        df_users_graded = validator.validate(df_users)
        df_amturk_users_to_rate = \
            self.df_participants_amturk[
                (self.df_participants_amturk['AssignmentStatus'] == "Submitted")]
        # keep only valid data with a correspondence in the data
        df_user_with_data_correspondence = \
            df_users_graded.merge(
                right=df_amturk_users_to_rate, on=['Answer.surveycode'])
        # get codes that did not find a match
        codes_all_to_rate = \
            df_amturk_users_to_rate['Answer.surveycode'].unique()
        codes_with_data_correspondence = \
            df_user_with_data_correspondence['Answer.surveycode'].unique()
        code_no_data_correspondence = \
            list(set(codes_all_to_rate) - set(codes_with_data_correspondence))
        # reject because of no data correspondence (insert work id or nickname)
        df_user_no_valid_survey_code = \
            self.df_participants_amturk[
                self.df_participants_amturk['Answer.surveycode'].isin(code_no_data_correspondence)]
        reject_msg_give_me_correct_code = \
            'Unfortunately you did not provide your survey code that is given to you at the end of the survey. You inserted either your worker id or nickname, but this information is not sufficient to link you and your answers. Unless you can get your code here and mail it to me: https://bit.ly/3b6jqh8, you have to retake the experiment here and send me an email: https://bit.ly/3rNd1yc . Only then I will be able to revert the decision if you meet 70% correct answers.'  # noqa
        df_user_no_valid_survey_code['Reject'] = \
            reject_msg_give_me_correct_code
        # GRADE
        reject_msg = 'The threshold of 70% correct answers was not met, even giving credits for nearly correct answers.'
        df_user_with_data_correspondence.loc[
            df_user_with_data_correspondence['is_valid'], 'Approve'] = \
            'x'
        df_user_with_data_correspondence.loc[
            ~df_user_with_data_correspondence['is_valid'], 'Reject'] = \
            reject_msg
        # JOIN
        df_all_users_to_rate = \
            pd.concat([
                df_user_with_data_correspondence,
                df_user_no_valid_survey_code], axis=0)
        df_all_users_to_rate = \
            df_all_users_to_rate.replace(np.nan, '', regex=True)
        df_all_users_to_rate = df_all_users_to_rate[self.initial_columns]
        # dump
        df_all_users_to_rate.to_csv(
            os.path.join(out_amturk_path, 'output.csv'),
            quoting=csv.QUOTE_ALL, index=False)
        return df_all_users_to_rate


class StandardValidator(object):

    def __init__(self):
        pass

    def validate(self, df):
        """Create the is_valid column based on other features."""
        df['is_valid'] = \
            (((df['median_time'] > 3)
                & (df['pesudo_correct'] >= 13)
                & (df['n_correct'] >= 3)
                & (df['n_unique'] == 17)) |
            ((df['median_time'] > 3)
                & (df['pesudo_correct'] >= 11)
                & (df['n_correct'] >= 6)
                & (df['n_unique'] == 17)))
        return df


class AnomalyAnalyzer(object):

    def __init__(self, pyod_analyzer):
        self.pyod = pyod_analyzer

    def spot_anomalies(self, df,
                       feature_columns=[
                           'n_correct', 'n_tfidf_mistakes', 'n_random_mistakes',
                           'avg_time', 'median_time']):
        """Create the is_valid column based on other features."""
        self.pyod.fit(df[feature_columns])
        # get the prediction labels and outlier scores of the training data
        y_train_pred = self.pyod.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = self.pyod.decision_scores_  # raw outlier scores
        df['is_anomaly'] = y_train_pred
        df['anomaly_score'] = y_train_scores
        return df


class Attentioner(object):

    def __init__(self):
        pass

    def parse_attention(self, df_humanlogs_data):
        """Enrich Human Logs with computed attention."""
        df_humanlogs_data['att_vector'], df_humanlogs_data['att_tokens'], \
            df_humanlogs_data['att_vector_w_click'] = \
            zip(*df_humanlogs_data.parallel_apply(
                lambda row:
                self._get_attention_weights(data=row), axis=1))
        return df_humanlogs_data

    def _get_attention(self, index_token, df_interaction):
        """Compute the attention (total second) for the given token.

        Compute attention for every token aka. count the second each
        token was over."""
        # no dataframe of interactions in case the user subitted without looking
        # at the code
        df_interaction = deepcopy(df_interaction)
        if len(df_interaction) == 0:
            return 0
        single_token = \
            df_interaction[df_interaction["position"] == int(index_token)]
        # remove clicked (to handle them separately)
        single_token = single_token[single_token["clicked"] == 0]
        # compute fixation on each token
        single_token['fixation'] = single_token['t'].diff()
        # count the time from the event where the mouse leave the token
        single_token = single_token[single_token["over"] == 0]
        attention = np.sum(single_token["fixation"])
        return attention

    def _get_attention_from_click(self, index_token, df_interaction, submission_time):
        """Compute the clicked attention (total second) for the given token.

        Compute attention for every token aka. count the second each
        token was visible due to a click."""
        # no dataframe of interactions in case the user subitted without looking
        # at the code
        df_interaction = deepcopy(df_interaction)
        if len(df_interaction) == 0:
            return 0
        single_token = \
            df_interaction[df_interaction["position"] == int(index_token)]
        # keep only clicked events
        single_token = single_token[single_token["clicked"] == 1]
        # get only time
        # get only time
        time_events = list(single_token['t'])
        # append submission time at the end if necessary
        if len(time_events) % 2 == 1:
            time_events.append(submission_time)
        click_attention = \
            sum([-e if i % 2 == 0 else e
                for i, e in enumerate(time_events)])
        return click_attention

    def _get_attention_weights(self, data):
        """Get the attention weights of the given function."""
        # USE INTERACTIONS
        token_interaction = data['tokeninteraction']
        df_token_interaction = pd.DataFrame(token_interaction)
        df_token_interaction = \
            df_token_interaction.sort_values(by='t')

        # check clicked tokens to draw squares around them
        clicked_tokens = np.array(data['finalclickedtokens'])
        clicked_tokens_indices = np.where(clicked_tokens == 1)[0].tolist()

        # COMPUTE ATTENTION
        attentions = []
        click_attentions = []
        attention_w_click = []
        texts = []
        for i, t in enumerate(data['tokens']):
            new_attention = \
                self._get_attention(index_token=t['id'],
                                    df_interaction=df_token_interaction)

            new_click_attention = \
                self._get_attention_from_click(
                    index_token=t['id'],
                    df_interaction=df_token_interaction,
                    submission_time=data['tottimemilli'])
            texts.append(t['text'])
            attentions.append(new_attention)
            click_attentions.append(new_click_attention)

            # consider only token
            df_single_token = \
                df_token_interaction[
                    df_token_interaction["position"] == int(t['id'])]
            # remove clicked (to handle them separately)
            df_single_token_hover = \
                df_single_token[df_single_token["clicked"] == 0]
            hover_intervals = list(df_single_token_hover['t'])
            if len(hover_intervals) % 2 == 1:
                hover_intervals = hover_intervals[:-1]

            df_single_token_click = \
                df_single_token[df_single_token["clicked"] == 1]
            click_intervals = list(df_single_token_click['t'])
            if len(click_intervals) % 2 == 1:
                click_intervals.append(data['tottimemilli'])

            assert len(hover_intervals) % 2 == 0
            assert len(click_intervals) % 2 == 0

            hover_intervals_pairs = []
            click_intervals_pairs = []

            if len(hover_intervals) >= 2:
                hover_intervals_pairs = \
                    list(zip(hover_intervals[::2], hover_intervals[1::2]))
            if len(click_intervals) >= 2:
                click_intervals_pairs = \
                    list(zip(click_intervals[::2], click_intervals[1::2]))

            #if len(click_intervals_pairs) > 1:
            #    import pdb
            #    pdb.set_trace()

            double_attention = \
                self._get_time_for_double_counting(
                    sequence_hover=hover_intervals_pairs,
                    sequence_click=click_intervals_pairs
                )

            all_attention = new_attention + new_click_attention - double_attention
            # DEBUG
            # if (all_attention < 0):
            #    abs_delta_time = data['time'] - data['timeopenpage']
            #    df_relevant = df_token_interaction[df_token_interaction['token'] == t['text']]
            #    df_relevant = df_relevant.sort_values(by='t')
            #    df_relevant_clicked = df_relevant[df_relevant['clicked'] == 1]
            #    import pdb
            #    pdb.set_trace()

            attention_w_click.append(all_attention)

        return attentions, texts, attention_w_click

    def _get_time_for_double_counting(self, sequence_hover, sequence_click):
        """Get the milliseconds that the we gave double attention.

        get_time_for_double_counting(
            sequence_hover=[(4,5), (7,10), (14, 16), (20,24)],
            sequence_click=[(14, 30)])

            intersections:
            { 14 , 16 }
            { 20 , 24 }

            Result: 6
        """
        # i and j pointers for arr1
        # and arr2 respectively
        i = j = 0

        n = len(sequence_hover)
        m = len(sequence_click)

        intersection_intervals_list = []
        # Loop through all intervals unless one
        # of the interval gets exhausted
        while i < n and j < m:

            c_hover = sequence_hover[i]
            c_click = sequence_click[j]
            # Left bound for intersecting segment
            l = max(c_hover[0], c_click[0])

            # Right bound for intersecting segment
            r = min(c_hover[1], c_click[1])

            # If segment is valid print it
            if l <= r:
                # print('{', l, ',', r, '}')
                intersection_intervals_list += [(l, r)]

            # If i-th interval's right bound is
            # smaller increment i else increment j
            if c_hover[1] < c_click[1]:
                i += 1
            else:
                j += 1
        intersection_time = np.sum([
            interval[1] - interval[0]
            for interval in intersection_intervals_list
        ])
        return intersection_time

class Downloader(object):

    def __init__(self, setting_file_path):
        # READ CONFGURATION
        with open(setting_file_path, 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
        uri = config_dict['mongodb_atlas']['endpoint']
        username = config_dict['mongodb_atlas']['username']
        password = config_dict['mongodb_atlas']['password']
        self.uri = uri.replace('<username>', username).replace('<password>', password)

        self.database = config_dict['mongodb_atlas']['database']
        self.collection = config_dict['mongodb_atlas']['collection']
        self.collection_delivery_key = config_dict['mongodb_atlas']['collection_delivery']
        self.collection_rating_key = config_dict['mongodb_atlas']['collection_rating']
        logger.info(f"Database: {self.database}")
        logger.info(f"Collection Experiment: {self.collection}")
        logger.info(f"Collection Availability: {self.collection_delivery_key}")

    def get_most_recent_collection(self, out_folder='mongo_backups'):
        """Download json dump of mongo db experiments."""
        out_exp = self._get_collection(
            collection_name_key=self.collection,
            out_folder=out_folder)
        out_rating = self._get_collection(
            collection_name_key=self.collection_rating_key,
            out_folder=out_folder)
        return out_exp, out_rating

    def _get_collection(self, collection_name_key, out_folder='mongo_backups'):
        """Download json dump of mongo db experiments."""
        tmp_filename = \
            self._run_mongo_export(collection_name_key, out_folder=out_folder)
        out_fixed_file = \
            os.path.join(out_folder, 'most_recent_' + collection_name_key + '.json')
        logger.info(f"Default most recent download path: {out_fixed_file}")
        shutil.copyfile(tmp_filename, out_fixed_file)
        return out_fixed_file

    def _run_mongo_export(self, collection_name_key,
                          out_folder='mongo_backups'):
        """Download the experiment collection with mongoexport tool."""
        filename_mongodb_exp = collection_name_key + '-' + str(int(time())) + '.json'
        logger.info(f"Output folder: {out_folder}")
        logger.info(f"Temporary filename: {filename_mongodb_exp}")
        only_main_uri = self.uri[:self.uri.rfind('/') + 1] + self.database
        logger.info(f"Connecting to: {only_main_uri}")
        self.out_timestamped_file = \
            os.path.join(out_folder, filename_mongodb_exp)
        completed_process = \
            subprocess.run([
                'mongoexport' +
                ' --forceTableScan' +
                ' --uri ' + self.uri[:self.uri.rfind('/') + 1] + self.database +
                ' --collection ' + collection_name_key +
                ' --out ' + self.out_timestamped_file +
                ' --jsonArray'], capture_output=True, shell=True)

        logger.info('Completed - Output')
        logger.info(completed_process)
        return self.out_timestamped_file
