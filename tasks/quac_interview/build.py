#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager
from parlai_internal.agents.interviewee.interviewee import IntervieweeAgent
from parlai_internal.utilities.util import get_correct_alignement
import torch

RESOURCES = [
    DownloadableFile(
        'https://s3.amazonaws.com/my89public/quac/train_v0.2.json',
        'train_v0.2.json',
        'ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a',
        zipped=False,
    ),
    DownloadableFile(
        'https://s3.amazonaws.com/my89public/quac/val_v0.2.json',
        'val_v0.2.json',
        '09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378',
        zipped=False,
    ),
]

VERSION = '0.2'

SHOULD = '__SHOULD__'
MAYBE = '__MAYBE__'
SHOULD_NOT = '__SHOULDNOT__'

YES = '__YES__'
NO = '__NO__'
NEITHER = '__NEITHER__'

MAP_CONTINUATION = {'m': MAYBE, 'f': SHOULD, 'n': SHOULD_NOT}
MAP_AFFIRMATION = {'y': YES, 'n': NO, 'x': NEITHER}

MODEL_KEY = 'valhalla/longformer-base-4096-finetuned-squadv1'

OUTPUT_FORMAT = (
    'text:{answer}\tfollowup:{continuation}\tyesno:'
    '{affirmation}\tanswer_starts:{start}\tlabels:{labels}\t'
    'context:{context}\ttitle:{title}\tsection_title:{section_title}\t'
    'background:{background}\tcontext_token_weights:{context_token_weights}\tturn_id:{turn_id}'
)

BUILD_IMPORTANCE_WEIGHT =True

def compute_context_tokens_weights(title, context, model, tokenizer, agent_dict, opt):
    tokens_weights = []
    encoding = tokenizer.encode_plus(title, context, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    context_encodings = tokenizer.encode_plus(context)
    if opt and not opt['no_cuda']:
        input_ids = input_ids.to(torch.device('cuda:{}'.format(opt.get('gpu', 0))))
        attention_mask = attention_mask.to(torch.device('cuda:{}'.format(opt.get('gpu', 0))))
    with torch.no_grad():
        output_ = model(input_ids, attention_mask=attention_mask)
    sep_idx = encoding['input_ids'][0].tolist().index(tokenizer.sep_token_id)
    start_scores = output_.start_logits[0][sep_idx + 2:-1]
    end_scores = output_.end_logits[0][sep_idx + 2:-1]
    model_weights = start_scores + end_scores
    #turning the weight to positive
    model_weights = model_weights - model_weights.min()
    model_weights = model_weights/model_weights.sum()
    agent_tokens, agent_offset = agent_dict.bulk_tokenize([context], return_offsets=True)
    agent_tokens, agent_offset = agent_tokens[0], agent_offset[0]
    assert len(agent_tokens) == len(agent_offset)
    for i, w in enumerate(agent_tokens):
        if w == "" or w == " ":
            tokens_weights.append(0)
            continue
        start, end = agent_offset[i]
        start_idx, end_idx = get_correct_alignement(context, w, start)
        start_positions_context = context_encodings.char_to_token(start_idx)
        end_positions_context = context_encodings.char_to_token(end_idx - 1)
        weights = model_weights[start_positions_context - 1:end_positions_context]
        weights = float(weights.sum())
        tokens_weights.append(weights)
    assert len(tokens_weights) == len(agent_tokens)
    return tokens_weights

def title_synthesis(title, section_title):
    return title+ ": "+section_title


def _parse_answers(q_a):
    starts = []
    labels = []
    for each in q_a['answers']:
        starts.append(str(each['answer_start']))
        labels.append(each['text'].replace('|', ' __PIPE__ '))
    return '|'.join(starts), '|'.join(labels)


def _handle_paragraph(each, background_info=None, model=None, tokenizer=None, agent_dict=None, opt=None):
    output = []
    answers = []
    continuations = []
    affirmations = []
    starts = []
    for idx, q_a in enumerate(each['qas']):
        question_txt = q_a['question']
        title = ""
        section_title = ""
        background = ""
        context = ""
        context_token_weights = []
        if idx == 0:
            context = each['context'].replace('\n', '\\n')
            if background_info:
                title = background_info.get('title', "")
                section_title = background_info.get('section_title', "")
                background = background_info.get('background', "")
                if context and model and tokenizer and agent_dict:
                    combined_title = title_synthesis(title, section_title)
                    context_token_weights = compute_context_tokens_weights(combined_title, context, model,
                                                                           tokenizer, agent_dict, opt)
            output.append(
                OUTPUT_FORMAT.format(
                    answer="",
                    continuation="",
                    affirmation="",
                    start="",
                    labels=question_txt,
                    context=context,
                    title=title,
                    section_title=section_title,
                    background=background,
                    context_token_weights=context_token_weights,
                    turn_id=str(idx),
                )
            )
        else:
            output.append(
                OUTPUT_FORMAT.format(
                    answer=answers[-1],
                    continuation=continuations[-1],
                    affirmation=affirmations[-1],
                    start=starts[-1],
                    labels=question_txt,
                    context="",
                    title="",
                    section_title="",
                    background="",
                    context_token_weights=context_token_weights,
                    turn_id=str(idx)
                )
            )
        start, labels = _parse_answers(q_a)
        starts.append(start)
        answers.append(labels)
        continuations.append(MAP_CONTINUATION.get(q_a['followup']))
        affirmations.append(MAP_AFFIRMATION.get(q_a['yesno']))
        output.append('\n')
    output.append(
        OUTPUT_FORMAT.format(
            answer=answers[-1],
            continuation=continuations[-1],
            affirmation=affirmations[-1],
            start=starts[-1],
            labels="",
            context="",
            title="",
            section_title="",
            background="",
            context_token_weights=context_token_weights,
            turn_id=str(idx+1)
        )
    )
    output.append('\t\tepisode_done:True\n')
    return ''.join(output)


def make_parlai_format(outpath, dtype, data, model=None, tokenizer=None, opt=None):
    print('building parlai:' + dtype)
    if model and tokenizer:
        agent_dict = IntervieweeAgent.dictionary_class()(opt)
    else:
        agent_dict = None
    with PathManager.open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        for line in data:
            background_info = {'title': line['title'],
                               'section_title': line['section_title'],
                               'background': line['background']}
            for each in line['paragraphs']:
                fout.write(_handle_paragraph(each, background_info, model, tokenizer, agent_dict, opt))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'QuACQuestions')
    version = VERSION

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        if BUILD_IMPORTANCE_WEIGHT:
            model, tokenizer = init_model(opt)
        else:
            model, tokenizer = None, None

        with PathManager.open(os.path.join(dpath, RESOURCES[0].file_name)) as f:
            data = json.load(f)['data']
            make_parlai_format(dpath, 'train', data, model, tokenizer, opt)

        with PathManager.open(os.path.join(dpath, RESOURCES[1].file_name)) as f:
            data = json.load(f)['data']
            make_parlai_format(dpath, 'valid', data, model, tokenizer, opt)

        # Mark the data as built.
        del model
        del tokenizer
        build_data.mark_done(dpath, version_string=version)

def init_model(opt):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
    model = AutoModelForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
    model.eval()
    if not opt['no_cuda']:
        model.to(torch.device('cuda:{}'.format(opt.get('gpu', 0))))
    return model, tokenizer

