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

def reward_utterance_repetition(conversations, master_conv=None):
    """Allocates negative reward if a bot repeats in its current utterance.
    """
    if master_conv:
        conversations = [master_conv] + conversations
    num_convs = len(conversations)
    max_episode = len(master_conv)
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
        for j in len(conv):
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

def reward_you(conversations, master_conv=None):
    """Allocates reward for any sentence that contains the reward 'you'. Used
    for debugging

    Args:
        conversations: list of lists containing batch_size conversations
        and each conversation is of length 2 * episode_len + 1
    Returns:
        rewards: numpy array of size [batch_size, episode_len]
    """
    if master_conv:
        conversations = [master_conv] + conversations
    num_convs = len(conversations)
    max_episode = len(master_conv)
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
        for j in len(conv):
            if master_rewards and not conv[j].generated:
                dialogue_reward.append(master_rewards[j])
            else:
                filter_index = j+len(filtered)-episode_num
                you = conv[j].question.count('you')
                dialogue_reward.append(you * -1)
        if master_conv and i ==0:
            master_rewards = dialogue_reward
        else:
            diverged_rewards.append(dialogue_reward)
    if master_conv:
        return master_rewards, diverged_rewards
    else:
        return diverged_rewards


def reward_conversation_repetition(conversations, master_conv=None):
    """Allocates negative reward if the bot repeats a word it has said in a
    previous conversation turn.
    """
    num_convs = len(conversations)
    episode_len = len(conversations[0])
    rewards = np.zeros((num_convs, episode_len))
    for i in range(num_convs):
        conv = conversations[i]
        bot_responses = [turn.question for turn in conv]
        tokenized = [resp.split() for resp in bot_responses]
        filtered = [set(resp).difference(filters) for resp in tokenized]

        for j in range(1, episode_len):
            current = filtered[j]
            prev = set.union(*filtered[:j])
            repeats = current.intersection(prev)
            rewards[i][j] = len(repeats)

    # You get more negative rewards with more repeats
    return -1 * rewards


def reward_bot_response_length(conversations, master_conv=None):
    """Allocates reward for longer bot outputs/responses.
    """
    num_convs = len(conversations)
    episode_len = len(conversations[0])
    # Flattened bot responses
    bot_responses = [turn.question for conv in conversations for turn in conv]

    # Clean punctuation to avoid ? ? ? ? ? long responses
    punct_map = str.maketrans('', '', string.punctuation)
    bot_responses = [resp.translate(punct_map) for resp in bot_responses]
    response_length = [len(resp.split()) for resp in bot_responses]
    rewards = np.array(response_length).reshape(num_convs, episode_len)
    return rewards


def reward_word_similarity(conversations):
    """Allocates reward when bot repeats word appearing in user utterance
    """
    num_convs = len(conversations)
    episode_len = len(conversations[0])
    # Flattened responses
    bot_responses = [turn.question for conv in conversations for turn in conv]
    user_responses = [turn.answer for conv in conversations for turn in conv]

    user_tokenized = [sent.split() for sent in user_responses]
    bot_tokenized = [sent.split() for sent in bot_responses]

    # Don't reward for repeating stopwords, question words, or <unknown>
    filter = set.union(filters, question_words, {'<unk>'})
    bot_filtered = [set(resp).difference(filter)
                    for resp in bot_tokenized]

    rewards = np.zeros(num_convs * episode_len)
    for i in range(num_convs * episode_len):
        in_common = [w for w in bot_filtered[i] if w in user_tokenized[i]]

        # Normalize by response len to prevent spamming response
        if len(bot_tokenized[i]):
            rewards[i] = len(in_common) / len(bot_tokenized[i])

    rewards = rewards.reshape(num_convs, episode_len)
    return rewards


def reward_question(conversations):
    """Allocates reward for any bot utterance that asks questions."""
    num_convs = len(conversations)
    episode_len = len(conversations[0])
    rewards = np.zeros(num_convs * episode_len)

    # Flattened responses
    bot_responses = [turn.question for conv in conversations for turn in conv]
    question_words = ['who', 'what', 'why', 'where', 'how', 'when']

    for i, resp in enumerate(bot_responses):
        resp = resp.lower()
        if any(q in resp for q in question_words) and '?' in resp:
            rewards[i] += 1

    rewards = rewards.reshape(num_convs, episode_len)
    return rewards

