"""Library of functions for calculating rewards
Note that rewards should be normalized for best results.
"""

import string
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader


import numpy as np
import torch
from nltk.corpus import stopwords

COLA_MODEL_KEY = "textattack/roberta-base-CoLA"
EPSILON = np.finfo(np.float32).eps
stopwords = stopwords.words('english')
question_words = {'who', 'what', 'why', 'where', 'how', 'when'}
punct = list(string.punctuation)
contractions = ["'s", "'d", "'ld", "n't", "'re", "'ll", "'ve"]
filters = set(stopwords + contractions + punct)


def normalize_rewards(rewards):
    index = 0
    reward_index_map = []
    flat_rewards = []
    nmlz_rewards = None
    for conv in rewards:
        reward_index_map.append([])
        for r in conv:
            flat_rewards.append(r)
            reward_index_map[-1].append(index)
            index += 1
    if flat_rewards:
        nmlz_rewards = normalizeZ(np.array(flat_rewards))
        for i, conv in enumerate(rewards):
            for j, _ in enumerate(conv):
                index = reward_index_map[i][j]
                rewards[i][j] = nmlz_rewards[index]
    return rewards



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

class BasedRewardScorer(object):

    def __init__(self, name, weight, use_cuda=False):
        self.global_reward = False
        self.weight = weight
        self.name = name
        self.use_cuda = use_cuda
        self.required_normalise = False

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
        return None

class BasedGlobalRewardScorer(BasedRewardScorer):

    def __init__(self, name, weight, use_cuda=False):
        super().__init__(name, weight, use_cuda=use_cuda)
        self.global_reward = True

class BasedLocalRewardScorer(BasedRewardScorer):

    def __init__(self, name, weight, use_cuda=False):
        super().__init__(name, weight, use_cuda=use_cuda)

class LinguisticAcceptabilityScorer(BasedLocalRewardScorer):

    def __init__(self, name, weight, use_cuda=True):
        from torch.utils.data import TensorDataset


        super().__init__(name, weight, use_cuda=use_cuda)
        self.init_model()
        self.batch_size = 32
        self.softmax = torch.nn.Softmax(dim=1)

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
        batches, index_map, conversations = self.batchify(conversations, master_history, last_action, agent_dictionary)
        la_scores = self.compute_score(batches)
        master_rewards, diverge_rewards = self.assign_rewards(la_scores, index_map, conversations)
        return master_rewards, diverge_rewards

    def init_model(self):
        self.model = RobertaForSequenceClassification.from_pretrained(COLA_MODEL_KEY)
        self.tokenizer = RobertaTokenizer.from_pretrained(COLA_MODEL_KEY)
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

    def batchify(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
        master_conv = None
        if master_history and last_action:
            master_conv = master_history.dialogues
            master_conv[-1].answer = last_action['text']
            conversations = [master_conv] + conversations
        num_convs = len(conversations)
        index_map = []
        dialogue_questions = []
        for i in range(num_convs):
            conv = conversations[i]
            if i != 0:
                bot_responses = [(i, turn.question) for i, turn in enumerate(conv) if turn.generated]
            else:
                bot_responses = [(i, turn.question) for i, turn in enumerate(conv)]
            for eps_indx, d in bot_responses:
                dialogue_questions.append(d)
                index_map.append((i, eps_indx))
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        encoded_dict = self.tokenizer(
            dialogue_questions,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=30,
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(encoded_dict['input_ids'], encoded_dict['attention_mask'])
        batches = DataLoader(
            dataset,  # The training samples.
            batch_size=self.batch_size  # Trains with this batch size.
        )
        return batches, index_map, conversations

    def compute_score(self, batches):
        scores = []
        # Evaluate data for one epoch
        for batch in batches:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            if self.use_cuda:
                b_input_ids = batch[0].cuda()
                b_input_mask = batch[1].cuda()

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = self.model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               return_dict=True)
            logits =self.softmax(result.logits.detach().cpu()).numpy()
            scores.extend(list(logits[:, 1]))
        return scores

    def assign_rewards(self, la_scores, index_map, conversations):
        master_rewards = []
        diverge_reward = []
        prev_conv_index = -1
        prev_dialogue_idnex = -1
        for i, (c_index, d_index) in enumerate(index_map):
            r = la_scores[i]
            if c_index == 0:
                master_rewards.append(r)
            elif c_index != prev_conv_index:
                if c_index - prev_conv_index > 1:
                    diverge_reward.append(master_rewards)
                diverge_reward.append([])
                if d_index != 0:
                    for j in range(d_index):
                        diverge_reward[-1].append(master_rewards[j])
                diverge_reward[-1].append(r)
            else:
                diverge_reward[-1].append(r)
            prev_conv_index = c_index
            prev_dialogue_idnex = d_index
        return master_rewards, diverge_reward


class UtteranceRepetitionScorer(BasedLocalRewardScorer):

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
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
                    filter_index = j + len(filtered) - episode_num
                    repeats = len(filtered[filter_index]) - len(set(filtered[filter_index]))
                    dialogue_reward.append(repeats * -1)
            if master_conv and i == 0:
                master_rewards = dialogue_reward
            else:
                diverged_rewards.append(dialogue_reward)
        if master_conv:
            return master_rewards, diverged_rewards
        else:
            return diverged_rewards


class YouScorer(BasedLocalRewardScorer):

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
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


class ConversationRepetitionScorer(BasedLocalRewardScorer):

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
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


class SimpleCoverageScorer(BasedGlobalRewardScorer):

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
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
                    if conv[j].answer == "CANNOTANSWER":
                        dialogue_reward.append(0)
                    else:
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


class OutputLengthScorer(BasedLocalRewardScorer):

    def __init__(self, name, weight, use_cuda=True):
        from torch.utils.data import TensorDataset
        super().__init__(name, weight, use_cuda=use_cuda)
        self.required_normalise = True

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
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


class QuestionTokensScorer(BasedLocalRewardScorer):

    def reward(self, conversations, master_history=None, last_action=None, agent_dictionary=None):
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


REWARD_MAP = {'reward_question': QuestionTokensScorer, 'reward_you': YouScorer,
              'reward_conversation_repetition': ConversationRepetitionScorer,
              'reward_utterance_repetition': UtteranceRepetitionScorer,
              'reward_bot_response_length': OutputLengthScorer,
              'reward_simple_coverage': SimpleCoverageScorer,
              'reward_linguistic_acceptability': LinguisticAcceptabilityScorer}
