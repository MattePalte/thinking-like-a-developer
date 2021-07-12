"""
It compares human and machine attention weights.
"""

import json
import logging
import os
import re
from tqdm import tqdm
import abc
from abc import ABCMeta
import pandas as pd
import numpy as np
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import os
import json
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
from matplotlib.pyplot import imshow

from copy import copy
import numpy as np

import yaml
import pymongo

from pprint import pprint
from datetime import datetime
import argparse


logging.basicConfig()  # required
logger = logging.getLogger('attention-comparer')
logger.setLevel(logging.INFO)


class VisualToken(object):

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

    def draw_PIL(self, drw, global_attention=0,
                 guessed_right=False,
                 human=True,
                 almost_correct=False):
        """Draw the patch on the plot."""
        alpha = 0.1
        if global_attention != 0:
            alpha = int((float(self.attention) / global_attention) * 255)
        if self.attention == 0:
            alpha = 0
        if human:
            # human
            if (almost_correct):
                color = (255, 127, 80, alpha)  # orange)
            else:
                if (guessed_right):
                    color = (26, 255, 26, alpha)  # green
                else:
                    color = (255, 102, 102, alpha)  # red
        else:
            # Machine
            color = (0, 191, 255, alpha)  # blue
        border = None
        if self.clicked:
            border = 'red'
        rect = \
            drw.rectangle([
                self.x,
                self.y,
                self.x + self.width,
                self.y + self.height],
                outline=border,
                width=2,
                fill=color)

    def add_attention(self, attention):
        self.attention = attention

    def __repr__(self):
        return 'x:' + str(self.x).zfill(3) \
                + ' - y:' + str(self.y).zfill(3) \
                + ' - width:' + str(self.width).zfill(4) \
                + ' - height:' + str(self.height).zfill(4) \
                + ' - |' + self.text + '|'


class Visualizer(object):

    def __init__(self, df_human):
        self.df_human = deepcopy(df_human)

        self.df_human['is_warmup'] = \
            (self.df_human['id'].astype('int') < 3).astype('int')

        self.df_human = self.df_human[self.df_human['is_warmup'] == 0]

        self.df_human.sort_values(by='time', ascending=True, inplace=True)
        self.df_human.drop_duplicates(
            subset=['randomcode', 'uuid'], keep='first', inplace=True
        )

    def plot_token_heatmap(self,
                           survey_code_col,
                           correct_col, almost_correct_col,
                           user_selection__col,
                           formatted_col, attention_col,
                           tokens_col, clicked_col,
                           id_col,
                           sortby,
                           only_users=None,
                           limit=None):
        """Plot Human and Machine heatmaps on token side by side."""
        df = deepcopy(self.df_human)
        df.sort_values(by=sortby, inplace=True)
        if only_users is not None:
            df = df[df[survey_code_col].isin(only_users)]

        counter = 0
        for row in df.iterrows():
            counter += 1
            if limit is not None and counter > limit:
                break
            idx = row[0]
            record = row[1]
            correct_answered = \
                record[correct_col] == record[user_selection__col]

            almost_correct = \
                record[user_selection__col].lower() in [
                    x.lower().replace('_', '')
                    for x in record[almost_correct_col]]

            idx = record[id_col]
            user_code = record[survey_code_col]
            print('*' * 50)
            print(f"Ground Truth: {record[correct_col]} - Provenance: {record['nickname']} - {user_code}")
            print(f'Similar options: {record[almost_correct_col]}')

            fig, ax = self.process_single(
                tokens=record[tokens_col],
                human=True,
                attention=record[attention_col],
                formattedcode=record[formatted_col],
                correct_answered=correct_answered,
                almost_correct=almost_correct,
                final_clicked_tokens=record[clicked_col])

            ax.set_title(
                f'Ground Truth: {record[correct_col]} '
                + f'- User Selection: {record[user_selection__col]}')

            plt.show()

    def process_single(self, tokens, attention, human,
                       formattedcode,
                       correct_answered, almost_correct,
                       final_clicked_tokens=None):
        """Display attention of the given function."""
        # PREPARE IMAGE
        path_font_file = '../public/FreeMono.ttf'
        surce_code_content = formattedcode
        #img_name = folder + data['id'] + data['rawdictionarykey'][1:] + '.png'

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

        # check clicked tokens to draw squares around them
        if final_clicked_tokens is not None:
            clicked_tokens = np.array(final_clicked_tokens)
            clicked_tokens_indices = np.where(clicked_tokens == 1)[0].tolist()
        else:
            clicked_tokens_indices = []

        # INSTANTIATE TOKENS
        # get the positon form the metadata of tokens
        viz_tokens = []
        # DEBUG print(tokens)
        # DEBUG print(formattedcode)
        for i, t in enumerate(tokens):
            # print(t)
            new_token = \
                VisualToken(
                    index_id=t['index_id'],
                    text=t['text'],
                    x=char_width * int(t['start_char']),
                    y=char_height * int(t['line']),
                    width=char_width * len(t['text']),
                    height=char_height,
                    clicked=(i in clicked_tokens_indices))
            viz_tokens.append(new_token)

        # COMPUTE ATTENTION
        global_attention = 1
        # compute attention
        for att, viz_token in zip(attention, viz_tokens):
            viz_token.add_attention(att)

        # COMPUTE REFERENCE ATTENTION TO RESCALE
        # sum all the attention received by the tokens
        global_attention = 0
        attentions = []
        for viz_token in viz_tokens:
            attentions.append(viz_token.attention)
        global_attention = max(attentions) * 1.33

        # check user was right to decide the color of the tokens (red vs green)
        # correct_answered decides the color
        for viz_token in viz_tokens:
            # print(token)
            viz_token.draw_PIL(drw, global_attention, correct_answered, human, almost_correct)

        #img.save(img_name)
        #return img_name
        imshow(np.asarray(img))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        if human:
            plt.title('Human')
        else:
            plt.title('Machine')
        ax = plt.gca()
        return fig, ax


def plot_statistics(df, column_name, ax=None, color='blue'):
    df = deepcopy(df)
    mean = df[column_name].mean()
    median = df[column_name].median()

    if ax is None:
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)})
        sns.boxplot(x=df[column_name], ax=ax_box, color=color)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        ax_box.set(xlabel='')
        ax_box.yaxis.label.set_size(14)
        ax_box.xaxis.label.set_size(14)
    else:
        ax_hist = ax

    sns.histplot(x=df[column_name], ax=ax_hist, color=color)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')
    ax_hist.legend({f'Mean {mean:.2f}':mean, f'Median {median:.2f}':median})
    ax_hist.yaxis.label.set_size(14)
    ax_hist.xaxis.label.set_size(14)

    if ax is None:
        plt.show()


def inspect(df, column_name, comparer,
            machine_col='att_vector_avg',
            human_col='att_vector_w_click',
            n_records_per_side=5,
            center=False,
            center_position=0.5,
            columns_to_observe=None):
    df = df.sort_values(by=column_name, ascending=True)
    df = df.drop_duplicates(subset='uuid')

    if center:
        center_position = int(len(df) * center_position)
        uuid_center = \
            df.iloc[center_position - n_records_per_side:center_position + n_records_per_side]['uuid']
        randomcode_center = \
            df.iloc[center_position - n_records_per_side:center_position + n_records_per_side]['randomcode']
        for uuid, randomcode in zip(uuid_center, randomcode_center):
            comparer.plot_token_heatmap_side_by_side(
                    machine_col=machine_col,
                    human_col=human_col,
                    only_uuids=[uuid],
                    only_users=[randomcode],
                    columns_to_observe=columns_to_observe
                )
    else:
        # head
        print(f'Low value of {column_name}')
        uuid_head = df.head(n_records_per_side)['uuid']
        randomcode_head = df.head(n_records_per_side)['randomcode']
        for uuid, randomcode in zip(uuid_head, randomcode_head):
            comparer.plot_token_heatmap_side_by_side(
                    machine_col=machine_col,
                    human_col=human_col,
                    only_uuids=[uuid],
                    only_users=[randomcode],
                    columns_to_observe=columns_to_observe
                )
        #print(uuid_head)
        # tail
        print(f'High value of {column_name}')
        uuid_tail = df.tail(n_records_per_side)['uuid']
        randomcode_tail = df.tail(n_records_per_side)['randomcode']
        #print(uuid_tail)
        for uuid, randomcode in zip(uuid_tail, randomcode_tail):
            comparer.plot_token_heatmap_side_by_side(
                    machine_col=machine_col,
                    human_col=human_col,
                    only_uuids=[uuid],
                    only_users=[randomcode],
                    columns_to_observe=columns_to_observe
                )


# ----------------------------------------------------------------

from matplotlib import colors


class FlexibleVisualToken(object):

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

    def draw_PIL(self, drw, global_attention=0,
                 named_color='lime'):
        """Draw the patch on the plot."""
        alpha = 0.1
        if global_attention != 0:
            alpha = int((float(self.attention) / global_attention) * 255)
        if self.attention == 0:
            alpha = 0
        color_rgb = list(colors.to_rgb(named_color))
        color_rgb = [int(c * 255) for c in color_rgb]
        color_rgba = color_rgb + [alpha]
        color_rgba = tuple(color_rgba)
        border = None
        if self.clicked:
            border = 'red'
        rect = \
            drw.rectangle([
                self.x,
                self.y,
                self.x + self.width,
                self.y + self.height],
                outline=border,
                width=2,
                fill=color_rgba)

    def add_attention(self, attention):
        self.attention = attention

    def __repr__(self):
        return 'x:' + str(self.x).zfill(3) \
                + ' - y:' + str(self.y).zfill(3) \
                + ' - width:' + str(self.width).zfill(4) \
                + ' - height:' + str(self.height).zfill(4) \
                + ' - |' + self.text + '|'


def plot_maps(df,
              weight_cols=[],
              label_cols=None,
              colors_for_cols=None,
              predictor_entity_name='Entity Predictor',
              prediction_col=None,
              max_records=None,
              output_in_folder=None,
              add_participant_id=True,
              add_square_for_clicks=False,
              limit_visualization_to=3):
    """Print the attention weights on the method body."""
    assert len(weight_cols) > 0
    assert len(df) > 0

    counter_visualized_maps = 0

    for i, row in enumerate(df.iterrows()):
        #print(i)
        if max_records is not None and i > max_records:
            break
        content = row[1]
        for j, attention_type in enumerate(weight_cols):
            named_color = colors_for_cols[j] \
                if colors_for_cols is not None else 'red'
            final_clicked_tokens = content['finalclickedtokens'] \
                if add_square_for_clicks else []
            fig, ax = plot_single_map(
                tokens_in_code=content['tokens_in_code'],
                attention_weights=content[attention_type],
                formattedcode=content['formattedcode'],
                final_clicked_tokens=final_clicked_tokens,
                named_color=named_color
            )
            if output_in_folder is not None:
                attention_name = label_cols[j] \
                    if label_cols is not None else attention_type
                filename = f'{i}-{predictor_entity_name}-{attention_name}-mtd:{content["uuid"]}'
                if add_participant_id:
                    filename += f'-ptc:{content["randomcode"]}'
                filename = "".join([c for c in filename if c != ' ']) + '.png'
                filepath = os.path.join(output_in_folder, filename)
                print(filepath)
                prediction = content[prediction_col] \
                    if prediction_col is not None else 'undefined'
                if isinstance(prediction, list):
                    prediction = [p for p in prediction if p != '%END%']
                    prediction = [p for p in prediction if p != '%UNK%']

                title = f'{predictor_entity_name}: {prediction} - Original: {content["function_name"]}'
                title += f' (what you see: {attention_name} weights)'
                plt.title(title)
                fig.savefig(filepath, format='png')
            if counter_visualized_maps < limit_visualization_to:
                plt.show()
                counter_visualized_maps += 1


def plot_single_map(tokens_in_code,
                    attention_weights,
                    named_color,
                    formattedcode,
                    final_clicked_tokens=None):
    """Display attention of the given function."""
    # PREPARE IMAGE
    path_font_file = '../public/FreeMono.ttf'
    surce_code_content = formattedcode
    #img_name = folder + data['id'] + data['rawdictionarykey'][1:] + '.png'

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

    # check clicked tokens to draw squares around them
    if final_clicked_tokens is not None:
        clicked_tokens = np.array(final_clicked_tokens)
        clicked_tokens_indices = np.where(clicked_tokens == 1)[0].tolist()
    else:
        clicked_tokens_indices = []

    # INSTANTIATE TOKENS
    # get the positon form the metadata of tokens
    viz_tokens = []
    # DEBUG print(tokens)
    # DEBUG print(formattedcode)
    for i, t in enumerate(tokens_in_code):
        # print(t)
        new_token = \
            FlexibleVisualToken(
                index_id=t['index_id'],
                text=t['text'],
                x=char_width * int(t['start_char']),
                y=char_height * int(t['line']),
                width=char_width * len(t['text']),
                height=char_height,
                clicked=(i in clicked_tokens_indices))
        viz_tokens.append(new_token)

    # COMPUTE ATTENTION
    global_attention = 1
    # compute attention
    for att, viz_token in zip(attention_weights, viz_tokens):
        viz_token.add_attention(att)

    # COMPUTE REFERENCE ATTENTION TO RESCALE
    # sum all the attention received by the tokens
    global_attention = 0
    attentions = []
    for viz_token in viz_tokens:
        attentions.append(viz_token.attention)
    global_attention = max(attentions) * 1.33

    # check user was right to decide the color of the tokens (red vs green)
    # correct_answered decides the color
    for viz_token in viz_tokens:
        # print(token)
        viz_token.draw_PIL(drw, global_attention, named_color)

    #img.save(img_name)
    #return img_name
    imshow(np.asarray(img))
    fig = plt.gcf()
    #print(f'max_width: {max_width}')
    #print(f'max_width: {max_height}')
    FACTOR = 60
    fig.set_size_inches(max_width / FACTOR, max_height / FACTOR)

    plt.title('undefined')

    ax = plt.gca()
    return fig, ax
