"""This script converts the raw data of MongoDB to data ready for analysis."""
import sys
import click
sys.path.append('..')  # noqa
from utils.humanlogbank import HumanLogBank
from utils.humanlogbank import Attentioner
from utils.humanlogbank import Downloader
from utils.humanlogbank import StandardValidator
from utils.humanlogbank import AnomalyAnalyzer
from utils.humanlogbank import CsvAmazonValidator
from utils.modelofcode import ModelOfCode
from utils.comparer import Comparer
import pandas as pd
import numpy as np
import os
import json

# HELPER FUNCTIONS

PROJECT_NAMES = [
    "hibernate-orm",
    "intellij-community",
    "liferay-portal",
    "gradle",
    "hadoop-common",
    "presto",
    "wildfly",
    "spring-framework",
    "cassandra",
    "elasticsearch"
]


def step_start(step_name):
    """Print a well formated step start."""
    click.echo('=' * 79)
    half = int(abs(80 - len(step_name)) / 2)
    click.echo(' ' * half + step_name + ' ' * half)
    click.echo('-' * 79)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    '--setting_path', default='../config/settings.yaml',
    help='path to setting.yaml file (default: ../config/settings.yaml )')
@click.option(
    '--save_path', default='../data/raw_data/',
    help='path to save raw human logs (default: ../data/raw_data/ ')
def download(setting_path, save_path):
    """Download raw human logs from MongoDB.

    REQUIREMENT: you need mongoexport tool installed on your system.
    """
    click.echo('Downloading raw data')
    dwn = Downloader(setting_file_path=setting_path)
    downloaded_mongo_file, json_ratings = \
        dwn.get_most_recent_collection(out_folder=save_path)


@cli.command()
@click.option(
    '--weights_path', default='../data/raw_data/extreme_summarizer/',
    help='path to setting.yaml file (default: ../data/raw_data/extreme_summarizer/)')
@click.option(
    '--save_path', default='../data/precomputed_model_prediction/extreme_summarizer/',
    help='path to save preprocessed transfomer attention weights (default: ' +
         '../data/precomputed_model_prediction/extreme_summarizer/')
@click.option(
    '--setting_path', default='../config/settings.yaml',
    help='path to setting.yaml file (default: ../config/settings.yaml )')
@click.option(
    '--raw_file_folder_path', default='../data/raw_data/',
    help='path to find raw human logs (default: ../data/raw_data/ )')
def prapareextsummarizer(
        weights_path,
        save_path,
        setting_path,
        raw_file_folder_path):
    """Prepare the attention weights for the Ext.Summarizer model.

    In detail: it condenses the attention weights of multiple predicted tokens
    in a single method-level set of attention weights.
    REQUIREMENT: you need to have the attention weights in your raw data folder.
    """

    step_start(step_name='EXT. SUMMARIZER WEIGHTS READING')
    # prepare attention weights
    project_paths = [
        weights_path + '/GPU_' + str(n) + '_attentions.json'
        for n in PROJECT_NAMES
    ]
    model_of_code = \
        ModelOfCode.from_precomputed_prediction_multi_projects(
            path_list=project_paths,
            project_name_list=PROJECT_NAMES)

    step_start(step_name='HUMAN DATA LOADING')
    human_bank = \
        HumanLogBank.from_mongodbexport_via_settings(
            setting_file_path=setting_path,
            folder_path=raw_file_folder_path)
    df_attentions = human_bank.get_attentions_df()

    step_start(step_name='ATTENTION WEIGHTS AGGREGATION (LONG: ca 9 hours)')
    # uncomment to run 9 hours computation
    model_of_code.aggregate_attention()

    df_model_of_code = model_of_code.to_dataframe()
    df_model_of_code.tail()

    aggregated_weights = \
        weights_path + '/gpu_all_projects_avg_max_copy_prediction.json'

    step_start(step_name='SAVE AGGREGATE ATTENTION WEIGHTS')
    model_of_code.save(aggregated_weights)

    step_start(step_name='IDENTIFY THOSE FOR WHICH WE HAVE HUMAN DATA')
    model_of_code = ModelOfCode.load(aggregated_weights)

    df_machine_attention = model_of_code.to_dataframe()
    # remove last line because it is not correcly parsed
    df_machine_attention = df_machine_attention.iloc[:len(df_machine_attention) - 1]
    # drop att column
    df_machine_attention.drop(columns=['att_vector'], inplace=True)

    cmp = Comparer(
        df_human=df_attentions,
        df_machine=df_machine_attention)

    df_intersection = cmp.get_compare_df()

    df_machine_with_uuid = \
        df_intersection.rename(columns={'tokens_x': 'tokens'})
    df_machine_with_uuid = df_machine_with_uuid[[
        'uuid', 'id_body_hash', 'file_name', 'original_name', 'project_name',
        'tokens', 'body_string',
        'predicted_tokens', 'att_vector_max', 'att_vector_avg',
        'copy_att_vector_max', 'copy_att_vector_avg', 'copy_prob_max',
        'copy_prob_avg']]
    df_machine_with_uuid = \
        df_machine_with_uuid.drop_duplicates(subset=['uuid'])

    df_double_check_reconstruction = pd.merge(
        left=df_attentions,
        right=df_machine_with_uuid,
        on=['uuid', 'id_body_hash', 'file_name', 'project_name']
    )

    step_start(step_name='SAVE PROCESSED MACHINE ATTENTION')
    df_machine_with_uuid.to_json(
        os.path.join(save_path, 'machine_attention.json'),
        orient='records'
    )


@cli.command()
@click.option(
    '--weights_path', default='../data/raw_data/transformer/',
    help='path to setting.yaml file (default: ../data/raw_data/transformer/)')
@click.option(
    '--save_path', default='../data/precomputed_model_prediction/transformer/',
    help='path to save preprocessed transfomer attention weights (default: ' +
         '../data/precomputed_model_prediction/transformer/')
@click.option(
    '--path_functions', default='../data/precomputed_functions_info/functions_sampled_for_the_experiment.json',
    help='path to method proposed during the experiment (default: ../data/precomputed_functions_info/functions_sampled_for_the_experiment.json)')
def praparetransformer(weights_path, save_path, path_functions):
    """Prepare the attention weights for the Transformer model.

    In detail: it condenses the attention weights of multiple attention heads.
    REQUIREMENT: you need to have the attention weights in your raw data folder.
    """
    step_start(step_name='TRANSFORMER WEIGHTS READING')

    dfs = []

    for project in PROJECT_NAMES:
        print(f'Project: {project}')

        filepath_model_input = f'../data/input_transformer/{project}_human_annotated.code'
        filepath_model_output_predicted_name = f'{weights_path}/{project}_transformer_beam.json'
        filepath_model_output_attention_weights_copy = f'{filepath_model_output_predicted_name}.attention_copy'
        filepath_model_output_attention_weights_regular = f'{filepath_model_output_predicted_name}.attention_transformer'

        with open(filepath_model_input, 'r') as file_model_input:
            lines = file_model_input.readlines()
            lines = [l.rstrip() for l in lines]
            s_model_input = pd.Series(lines)

        with open(filepath_model_output_predicted_name, 'r') as file:
            tmp_dict = json.load(file)
            s_model_output_predicted_name = \
                pd.Series([[t.lower() for t in e[1][0].split()]
                           for e in tmp_dict.items()])

        with open(filepath_model_output_attention_weights_copy, 'r') as file:
            tmp_dict = json.load(file)
            s_model_output_attention_weights_copy = \
                pd.Series([e[1] for e in tmp_dict.items()])

        with open(filepath_model_output_attention_weights_regular, 'r') as file:
            tmp_dict = json.load(file)
            s_model_output_attention_weights_regular = \
                pd.Series([e[1] for e in tmp_dict.items()])

        eight_attentions_data = {}
        for i in range(8):
            key = f'att_transformers_regular_{i}'
            filename = f'{filepath_model_output_predicted_name}.attention_transformer_{i}'
            with open(filename, 'r') as file:
                tmp_dict = json.load(file)
                s_attention_weights_regular_single_head = \
                    pd.Series([e[1] for e in tmp_dict.items()])
            eight_attentions_data[key] = s_attention_weights_regular_single_head

        static_data = {
            'serialized_body_for_transformer': s_model_input,
            'predicted_tokens': s_model_output_predicted_name,
            'att_transformers_copy': s_model_output_attention_weights_copy,
            'att_transformers_regular': s_model_output_attention_weights_regular
        }
        data = {**static_data, **eight_attentions_data}

        df_transformer_output_current_project = \
            pd.DataFrame(data)

        dfs.append(df_transformer_output_current_project)

    df_transformer_output_all_projects = pd.concat(dfs)

    step_start(step_name='READING METHODS LIST')
    df_original_functions = pd.read_json(path_functions, orient='records')
    # derive the serializad version of the method body that was used by the transformer model as input.
    # e.g. something like: {&*separator*&this&*separator*&.&*separator*&
    df_original_functions['serialized_body_for_transformer'] = \
        df_original_functions['tokens_in_code'].apply(
            lambda list_tokens:
                '&*separator*&'.join([t['text'] for t in list_tokens])
        )

    step_start(step_name='ENRICH RECORDS WITH UUID')
    df_enriched_output_with_uuid = \
        pd.merge(
            left=df_transformer_output_all_projects,
            right=df_original_functions, #[['uuid', 'serialized_body_for_transformer', 'tokens_list', 'project_name']],
            on='serialized_body_for_transformer',
            how='inner'
        )
    df_enriched_output_with_uuid = \
        df_enriched_output_with_uuid.drop_duplicates(
            subset=['uuid']
        )

    step_start(step_name='AGGREGATE MULTI HEADS')
    df = df_enriched_output_with_uuid
    columns_to_average = [
        f'att_transformers_regular_{i}' for i in range(8)
    ]

    def create_average(row, column_names):
        """Create an average attention out of multiple attention columns."""
        vectors = [
            np.array(row[c]) for c in column_names
        ]
        matrix = np.vstack(vectors)
        avg_vector = np.mean(matrix, axis=0)
        print(avg_vector.shape)
        return list(avg_vector)

    df['att_transformers_regular_avg'] = df.apply(
        lambda row: create_average(row, columns_to_average),
        axis=1
    )
    df_enriched_output_with_uuid = df

    step_start(step_name='SAVE PROCESSED MACHINE ATTENTION')
    df_enriched_output_with_uuid.to_json(
        os.path.join(save_path, 'machine_attention.json'),
        orient='records')



@cli.command()
@click.option(
    '--setting_path', default='../config/settings.yaml',
    help='path to setting.yaml file (default: ../config/settings.yaml )')
@click.option(
    '--raw_file_folder_path', default='../data/raw_data/',
    help='path to find raw human logs (default: ../data/raw_data/ )')
@click.option(
    '--original_functions_path', default='../data/precomputed_functions_info/functions_sampled_for_the_experiment.json',
    help='path to original functions used for the experiment (default: ../data/precomputed_functions_info/functions_sampled_for_the_experiment.json )')
@click.option(
    '--user_csv_path', default='../data/user_info/users_provenance.csv',
    help='path to user info (default: ../data/user_info/users_provenance.csv )')
@click.option(
    '--include_rejected_users', default=False,
    help='if you wish to include all submissions also from rejected users ')
@click.option(
    '--include_warmup', default=False,
    help='if you wish to include all submissions also the three warm ups ')
@click.option(
    '--user_filter', default=2,
    help=('if you wish to keep only users from Amazon Mechanical Turk (1),'
          'students (0) or both (2). Default: both.'))
@click.option(
    '--output_path_folder', default='../data/processed_human_attention',
    help=('path to save processed attention (default: ../data/processed_human_attention )'))
def preparehuman(
        setting_path,
        raw_file_folder_path,
        original_functions_path,
        user_csv_path,
        include_rejected_users,
        include_warmup,
        user_filter,
        output_path_folder):
    """Convert the raw attention into human attention weights."""
    click.echo('Preparing data for analysis...')
    # Download data from MongoDB endpoint via mongoexport

    # prefix for the final output file
    prefix_output_file = 'processed'

    # initialize the human bank
    step_start(step_name='DATA LOADING')
    human_bank = \
        HumanLogBank.from_mongodbexport_via_settings(
            setting_file_path=setting_path,
            folder_path=raw_file_folder_path)

    # enrich with original data
    step_start(step_name='COMPARE WITH CORRECT ANSWERS')
    # compute correction
    human_bank.enrich_with_ground_truth(
        path_dataframe_json=original_functions_path)

    # validate
    step_start(step_name='COUNT ERRORS')
    human_bank.compute_user_level_mistakes()
    df_users = human_bank.get_users_df()
    # validate
    human_bank.validate(strategy=StandardValidator())

    # add amazon mechanical turk provenance
    step_start(step_name='ADD AMAZON')
    human_bank.enrich_with_amturk_user_flag_from_csv(
        path_amturk_user_info=user_csv_path
    )

    # filter out rejected users
    if not include_rejected_users:
        step_start(step_name='KEEP ONLY APPROVED USERS')
        human_bank.drop_records(
            where_col='from_valid_user', value=False
        )
    else:
        step_start(step_name='KEEP ALL USERS (ALSO REJECTED)')

    # filter out warmup tasks
    if not include_warmup:
        step_start(step_name='DROPPING WARMUP TASKS')
        human_bank.drop_records(
            where_col='is_warmup', value=1
        )
    else:
        step_start(step_name='KEEP ALL TASK (ALSO WARMUPS)')

    # filter out not amturk
    if user_filter != 2:
        step_start(step_name='FILTER USERS')
        human_bank.drop_records(
            where_col='is_amtruk_user', value=1 - user_filter
        )
        if user_filter == 1:
            prefix_output_file += '_only_AMTurk'
            print('Keeping only Amazon turkers')
        if user_filter == 0:
            prefix_output_file += '_only_Students'
            print('Keeping only student')

    # compute attention
    step_start(step_name='EXTRACT ATTENTION WEIGHTS')
    standard_attentioner = Attentioner()
    human_bank.parse_attention(standard_attentioner)

    # save the output
    step_start(step_name='SAVE THE HUMAN ATTENTION')
    if include_rejected_users:
        prefix_output_file += '_also_rejected_users'
    human_bank.save_backup(
        backup_folder=output_path_folder,
        static_suffix=prefix_output_file)


if __name__ == '__main__':
    cli()
