"""Library of functions for calculating rewards
Note that rewards should be normalized for best results.
"""
import os
import string
import pickle
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords


EPSILON = np.finfo(np.float32).eps
stopwords = stopwords.words('english')
question_words = {'who', 'what', 'why', 'where', 'how', 'when'}
punct = list(string.punctuation)
contractions = ["'s", "'d", "'ld", "n't", "'re", "'ll", "'ve"]
filters = set(stopwords + contractions + punct)


def normalizeZ(x):
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + EPSILON)


def discount(rewards, gamma=0.9):
    """Convert raw rewards from batch of episodes to discounted rewards.
    Args:
        rewards: [batch_size, episode_len]
    Returns:
        discounted: [batch_size, episode_len]
    """
    batch_size = rewards.shape[0]
    episode_len = rewards.shape[1]
    discounted = np.zeros_like(rewards)
    running_add = np.zeros((batch_size))
    for step in reversed(range(episode_len)):
        running_add = gamma * running_add + rewards[:, step]
        discounted[:, step] = running_add
    return discounted


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1) / np.sqrt((np.sum(a * a, axis=1) * np.sum(b * b, axis=1)))

def reward_utterance_repetition(conversations, master_history=None, last_action=None, agent_dictionary=None):
    """Allocates negative reward if a bot repeats in its current utterance.
    """
    master_conv = None
    if master_history and last_action:
        master_conv = master_history.dialogues
        master_conv[-1].answer = last_action['text']
        conversations = [master_conv] + conversations
    num_convs = len(conversations)
    diverged_rewards = []
    master_rewards = None
    master_cache = {}
    for i in range(num_convs):
        dialogue_reward = []
        conv = conversations[i]
        episode_num = len(conv)
        if master_cache:
            bot_responses = [turn.question for turn in conv if turn.generated]
        else:
            bot_responses = [turn.question for turn in conv]
        tokenized = [resp.split() for resp in bot_responses]
        filtered = [[w for w in resp if w not in filters] for resp in tokenized]
        for j in range(len(conv)):
            if master_rewards and not conv[j].generated:
                dialogue_reward.append(master_rewards[j])
            else:
                filter_index = j+len(filtered)-episode_num
                repeats = len(filtered[filter_index]) - len(set(filtered[filter_index]))
                dialogue_reward.append(repeats * -1)
        if master_conv and i ==0:
            master_rewards = dialogue_reward
        else:
            diverged_rewards.append(dialogue_reward)
    if master_conv:
        return master_rewards, diverged_rewards
    else:
        return diverged_rewards


def reward_you(conversations, master_history=None, last_action=None, agent_dictionary=None):
    """Allocates reward for any sentence that contains the reward 'you'. Used
    for debugging

    Args:
        conversations: list of lists containing batch_size conversations
        and each conversation is of length 2 * episode_len + 1
    Returns:
        rewards: numpy array of size [batch_size, episode_len]
    """
    master_conv = None
    if master_history and last_action:
        master_conv = master_history.dialogues
        master_conv[-1].answer = last_action['text']
        conversations = [master_conv] + conversations
    num_convs = len(conversations)
    diverged_rewards = []
    master_rewards = None
    for i in range(num_convs):
        dialogue_reward = []
        conv = conversations[i]
        for j in range(len(conv)):
            if master_rewards and not conv[j].generated:
                dialogue_reward.append(master_rewards[j])
            else:
                you = conv[j].question.count('you')
                dialogue_reward.append(you)
        if master_conv and i==0:
            master_rewards = dialogue_reward
        else:
            diverged_rewards.append(dialogue_reward)
    if master_conv:
        return master_rewards, diverged_rewards
    else:
        return diverged_rewards


def reward_conversation_repetition(conversations, master_history=None, last_action=None, agent_dictionary=None):
    """Allocates negative reward if the bot repeats a word it has said in a
    previous conversation turn.
    """
    master_conv = None
    if master_history and last_action:
        master_conv = master_history.dialogues
        master_conv[-1].answer = last_action['text']
        conversations = [master_conv] + conversations
    num_convs = len(conversations)
    diverged_rewards = []
    master_rewards = None
    master_cache = {}
    for i in range(num_convs):
        dialogue_reward = []
        conv = conversations[i]
        episode_num = len(conv)
        if master_cache:
            bot_responses = [turn.question for turn in conv if turn.generated]
        else:
            bot_responses = [turn.question for turn in conv]
        tokenized = [filter(lambda x: x not in stopwords and x not in question_words, resp.split()) for resp in bot_responses]
        filtered = [set(resp).difference(filters) for resp in tokenized]
        if master_cache:
            filtered = master_cache['filtered'][:episode_num-len(bot_responses)] + filtered
        for j in range(len(conv)):
            if j == 0:
                dialogue_reward.append(0)
            elif master_rewards and not conv[j].generated:
                dialogue_reward.append(master_rewards[j])
            else:
                current = filtered[j]
                prev = set.union(*filtered[:j])
                repeats = current.intersection(prev)
                dialogue_reward.append(-1*len(repeats))
        if master_conv and i ==0:
            master_rewards = dialogue_reward
            master_cache['filtered'] = filtered
        else:
            diverged_rewards.append(dialogue_reward)
    if master_conv:
        return master_rewards, diverged_rewards
    else:
        return diverged_rewards

def reward_simple_coverage(conversations, master_history=None, last_action=None, agent_dictionary=None):
    """
    This reward every tokens captured from the reponse generated by the interviewee.
    This is a simple reward method, so every tokens captured are considered equally important
    """
    master_conv = None
    if master_history and last_action:
        master_conv = master_history.dialogues
        master_conv[-1].answer = last_action['text']
        master_conv[-1].cache = master_history.get_cache(last_action)
        # conversations[0][-1].cache = master_history.get_cache(last_action)
        conversations = [master_conv] + conversations
    num_convs = len(conversations)
    context_tok_idx = agent_dictionary.txt2vec(master_history.context)
    total_toks = len(context_tok_idx)
    diverged_rewards = []
    master_rewards = None
    master_cache = {}
    for i in range(num_convs):
        dialogue_reward = []
        conv = conversations[i]
        episode_num = len(conv)
        if master_cache:
            interviewer_coverage = [set(range(*turn.cache['token_start_end'])) for turn in conv if turn.generated]
            interviewer_coverage = master_cache['coverages'][:episode_num-len(interviewer_coverage)] + interviewer_coverage
        else:
            interviewer_coverage = [set(range(*turn.cache['token_start_end'])) for turn in conv]
        for j in range(len(conv)):
            if master_rewards and not conv[j].generated:
                dialogue_reward.append(master_rewards[j])
            else:
                if j==0:
                    prev_coverage = set()
                else:
                    prev_coverage = set.union(*interviewer_coverage[:j])
                new_capture_count = len(interviewer_coverage[j].difference(prev_coverage))
                dialogue_reward.append(new_capture_count/total_toks)
        if master_conv and i==0:
            master_rewards = dialogue_reward
            master_cache['coverages'] = interviewer_coverage
        else:
            diverged_rewards.append(dialogue_reward)
    if master_conv:
        return master_rewards, diverged_rewards
    else:
        return diverged_rewards


def reward_bot_response_length(conversations, master_history=None, last_action=None, agent_dictionary=None):
    """Allocates reward for longer bot outputs/responses.
    """
    master_conv = None
    if master_history and last_action:
        master_conv = master_history.dialogues
        master_conv[-1].answer = last_action['text']
        conversations = [master_conv] + conversations
    rewards = []
    # Flattened bot responses
    bot_responses = [turn.question for conv in conversations for turn in conv]

    # Clean punctuation to avoid ? ? ? ? ? long responses
    punct_map = str.maketrans('', '', string.punctuation)
    bot_responses = [resp.translate(punct_map) for resp in bot_responses]
    response_length = [len(resp.split()) for resp in bot_responses]
    index = 0
    for i, conv in enumerate(conversations):
        conv_reward = response_length[index:index+len(conv)]
        rewards.append(conv_reward)
        index += len(conv)
    if master_conv:
        return rewards[0], rewards[1:]
    else:
        return rewards

def reward_question(conversations, master_history=None, last_action=None, agent_dictionary=None):
    """Allocates reward for any bot utterance that asks questions."""
    question_words = ['who', 'what', 'why', 'where', 'how', 'when']
    master_conv = None
    if master_history and last_action:
        master_conv = master_history.dialogues
        master_conv[-1].answer = last_action['text']
        conversations = [master_conv] + conversations
    num_convs = len(conversations)
    diverged_rewards = []
    master_rewards = None
    for i in range(num_convs):
        dialogue_reward = []
        conv = conversations[i]
        for j in range(len(conv)):
            if master_rewards and not conv[j].generated:
                dialogue_reward.append(master_rewards[j])
            else:
                question = conv[j].question.lower()
                if any(q in question for q in question_words) and '?' in question:
                    dialogue_reward.append(1)
                else:
                    dialogue_reward.append(0)
        if master_conv and i ==0:
            master_rewards = dialogue_reward
        else:
            diverged_rewards.append(dialogue_reward)
    if master_conv:
        return master_rewards, diverged_rewards
    else:
        return diverged_rewards





