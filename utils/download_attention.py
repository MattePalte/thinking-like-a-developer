import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import os
import json
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches

from copy import copy
import numpy as np

import yaml
import pymongo

from pprint import pprint
from datetime import datetime
import argparse

from bson.json_util import dumps

import pdb


def get_attention(index_token, df_interaction):
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


def get_attention_weights(data):
    """Get the attention weights of the given function."""
    # USE INTERACTIONS
    token_interaction = data['tokeninteraction']
    df_token_interaction = pd.DataFrame(token_interaction)

    # check clicked tokens to draw squares around them
    clicked_tokens = np.array(data['finalclickedtokens'])
    clicked_tokens_indices = np.where(clicked_tokens == 1)[0].tolist()

    # COMPUTE ATTENTION
    attentions = []
    for i, t in enumerate(data['tokens']):
        new_attention = \
            get_attention(index_token=t['id'],
                          df_interaction=df_token_interaction)
        attentions.append(new_attention)

    return attentions


def main(nickname):
    # READ CONFGURATION
    with open('../config/settings.yaml', 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    uri = config_dict['mongodb_atlas']['endpoint']
    username = config_dict['mongodb_atlas']['username']
    password = config_dict['mongodb_atlas']['password']
    uri = uri.replace('<username>', username).replace('<password>', password)

    database = config_dict['mongodb_atlas']['database']
    collection_name = config_dict['mongodb_atlas']['collection']

    download_folder = config_dict['utils']['tmp_download_folder']

    # OPEN DATABASE CONNECTION
    client = pymongo.MongoClient(uri)
    collection = client[database][collection_name]

    cursor = collection.find({'nickname': nickname})

    image_names = []
    titles = []

    print(f'Downloading data for nickname: {nickname}')
    print(f'From (online) collection: {collection_name}')

    for data in cursor:
        timestamp = data['time'] / 1000
        dt_object = datetime.fromtimestamp(timestamp)
        print("Date:", dt_object)
        # print(data)
        path = \
            os.path.join('..',
                         download_folder,
                         nickname + "-" + str(timestamp) + ".json")
        # Save files
        with open(path, 'w') as out_file:
            out_file.write(dumps(data))
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download the human attentions json from mongodb.')
    parser.add_argument(
        '--nickname',
        help='Nickname of the user to extract')
    args = parser.parse_args()
    if args.nickname is not None:
        main(nickname=args.nickname)
    else:
        parser.print_help()
