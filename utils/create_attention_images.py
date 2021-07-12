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


class Token(object):

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

    def draw(self, ax, global_attention=0):
        """Draw the patch on the plot."""
        alpha = 0.1
        if global_attention != 0:
            alpha = float(self.attention) / global_attention
        rect = \
            patches.Rectangle((self.x, self.y),
                              self.width, self.height,
                              linewidth=1,
                              edgecolor='r',
                              facecolor='red',
                              alpha=alpha)
        new_rect = copy(rect)
        ax.add_patch(new_rect)

    def draw_PIL(self, drw, global_attention=0,
                 guessed_right=False):
        """Draw the patch on the plot."""
        alpha = 0.1
        if global_attention != 0:
            alpha = int((float(self.attention) / global_attention) * 255)
        if self.attention == 0:
            alpha = 0
        if (guessed_right):
            color = (26, 255, 26, alpha)  # green
        else:
            color = (255, 102, 102, alpha)  # red
        border = None
        if self.clicked:
            border = 'red'
        rect = \
            drw.rectangle([self.x,
                           self.y,
                           self.x + self.width,
                           self.y + self.height],
                          outline=border,
                          width=2,
                          fill=color)

    def add_attention(self, df_token_interaction):
        self.attention = get_attention(index_token=self.index_id,
                                       df_interaction=df_token_interaction)

    def __repr__(self):
        return 'x:' + str(self.x).zfill(3) \
                + ' - y:' + str(self.y).zfill(3) \
                + ' - width:' + str(self.width).zfill(4) \
                + ' - height:' + str(self.height).zfill(4) \
                + ' - |' + self.text + '|'


def process(data, folder="../data/image_attention/"):
    """Display attention of the given function."""

    # PREPARE IMAGE
    path_font_file = '../public/FreeMono.ttf'
    surce_code_content = data['formattedcode']
    img_name = folder + data['id'] + data['rawdictionarykey'][1:] + '.png'

    ratio = (8.4/14)
    char_height = 20
    char_width = char_height * ratio

    # compute max width required
    lines = surce_code_content.splitlines()
    lines_len = [len(line) for line in lines]
    max_width = int(max(lines_len) * char_width)
    max_height = int(char_height * len(lines))

    img = Image.new('RGB', (max_width, max_height), color=(255, 255, 255))
    fnt = ImageFont.truetype(path_font_file, char_height)
    drw = ImageDraw.Draw(img, 'RGBA')
    drw.text((0, 0), surce_code_content, font=fnt, fill=(0, 0, 0))
    # CAN BE DELAYED AT AFTER TOKEN DRAWING img.save(img_name)

    # USE INTERACTIONS
    token_interaction = data['tokeninteraction']
    df_token_interaction = pd.DataFrame(token_interaction)

    # check clicked tokens to draw squares around them
    clicked_tokens = np.array(data['finalclickedtokens'])
    clicked_tokens_indices = np.where(clicked_tokens == 1)[0].tolist()

    # INSTANTIATE TOKENS
    # get the positon form the metadata of tokens
    tokens = []
    for i, t in enumerate(data['tokens']):
        # print(t)
        new_token = \
            Token(index_id=t['id'],
                  text=t['text'],
                  x=char_width * int(t['charStart']),
                  y=char_height * int(t['line']),
                  width=char_width * len(t['text']),
                  height=char_height,
                  clicked=(i in clicked_tokens_indices))
        tokens.append(new_token)

    # COMPUTE ATTENTION
    global_attention = 1
    # compute attention
    for token in tokens:
        token.add_attention(df_token_interaction=df_token_interaction)

    # COMPUTE REFERENCE ATTENTION TO RESCALE
    # sum all the attention received by the tokens
    global_attention = 0
    attentions = []
    for token in tokens:
        attentions.append(token.attention)
    global_attention = max(attentions) * 1.33

    # check user was right to decide the color of the tokens (red vs green)
    real = data['functionname']
    guess = data['functionnamebyuser']
    guessed_right = (real == guess)

    for token in tokens:
        # print(token)
        token.draw_PIL(drw, global_attention, guessed_right)

    img.save(img_name)
    title = f'Real Name: {real} - User Choice: {guess}'
    return img_name, title


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

    # OPEN DATABASE CONNECTION
    client = pymongo.MongoClient(uri)
    collection = client[database][collection_name]

    cursor = collection.find({'nickname': nickname})

    image_names = []
    titles = []

    print(f'Producing report for nickname: {nickname}')
    print(f'From (online) collection: {collection_name}')

    for data in cursor:
        timestamp = data['time'] / 1000
        dt_object = datetime.fromtimestamp(timestamp)
        print("Date:", dt_object)
        data = data
        img_name, title = process(data)
        image_names.append(img_name)
        titles.append(title)

    # Create web page
    html_content = f"<html><body><h1>Report for {nickname}</h1>"
    for (url, title) in zip(image_names, titles):
        # html_content += f'<img style="width: 100%;" src="{url}"/><br>\n'
        html_content += f"<h2>{title}</h2>"
        html_content += f'<img src="{url}"/><br>\n'

    html_content += "</body></html>"

    with open('../public/results.html', 'w') as out_file:
        out_file.write(html_content)
    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a Human Attention report.')
    parser.add_argument('--nickname', help='Nickname of the user to extract')
    args = parser.parse_args()
    if args.nickname is not None:
        main(nickname=args.nickname)
    else:
        parser.print_help()
    #
