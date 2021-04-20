import os
import copy
from parlai_internal.agents.interviewee.interviewee import IntervieweeAgent
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.utils.misc import warn_once
from parlai_internal.utilities.flow_lstm_util import util
from parlai_internal import reward_funcs
from .build import build
import torch

NO_ANSWER_REPLY = "CANNOTANSWER"
SUPPORTED_REWARDS = {'reward_question', 'reward_you',
                     'reward_conversation_repetition', 'reward_utterance_repetition',
                     'reward_bot_response_length', 'reward_simple_coverage', 'reward_linguistic_acceptability'}
DEFAULT_REWARD_LIST = {'reward_linguistic_acceptability'}


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt == 'test':
        warn_once('WARNING: Test set not included. Setting datatype to valid.')
        dt = 'valid'
    return os.path.join(opt['datapath'], 'QuACQuestions', dt + '.txt')

class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        """
        Get a specific example from the dataset.
        """
        ex = self.episodes[episode_idx][entry_idx]
        is_training = self.datatype == "train"
        qas_id = str(episode_idx) + "_" + str(entry_idx)
        answer_text = ex['text']
        question_text = ex['labels'][0]
        # is_impossible = answer_text == NO_ANSWER_REPLY
        start = ex["answer_starts"]
        if start:
            if not is_training:
                start_position_character = int(ex["answer_starts"].split('|')[0])
                answer_text = ex['text'].split('|')[0]
            else:
                start_position_character = int(ex["answer_starts"])
            char_start_end = (start_position_character, start_position_character + len(answer_text))
        else:
            char_start_end = (-1, -1)

        action = {
            'id': 'quac',
            'turn_id': ex['turn_id'],
            'qas_id': qas_id,
            'labels': ex['labels'],
            'context': ex['context'],
            'single_label_text': question_text,
            'episode_done': ex['episode_done'],
            # 'is_impossible': is_impossible,
            'followup': ex['followup'],
            'yesno': ex['yesno'],
            'text': answer_text,
            'no_answer_reply': NO_ANSWER_REPLY,
            'background': ex['background'],
            'section_title': ex['section_title'],
            'title': ex['title'],
            'character_start_end': char_start_end
        }
        return action

class ReinforcementLearningTeacherAgent(DefaultTeacher, IntervieweeAgent):

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.dict = self.build_dictionary()
        self.query_truncate = opt['query_maximum_length']
        self.context_truncate = opt['context_maximum_length']
        self.history_truncate = opt['history_maximum_length']
        self.rl_mode = opt['reinforcement_learning']
        self.exploration_steps = opt['exploration_steps']
        self.use_cuda = not opt['no_cuda']
        self.rewards_list = opt.get('rewards_list', DEFAULT_REWARD_LIST)
        self.rewards_weights = opt.get('rewards_weights', [1/len(self.rewards_list)]*len(self.rewards_list))
        # now set up any fields that all instances may need
        self.EMPTY = torch.zeros(0, dtype=torch.long)
        self.NULL_IDX = self.dict[self.dict.null_token]
        self.START_IDX = self.dict[self.dict.start_token]
        self.END_IDX = self.dict[self.dict.end_token]
        self.model = self.build_model()
        self.model.eval()
        if self.use_cuda:
            self.model.cuda()
        self.truncate = self.dict.maxtokens
        self.history = self.build_history()
        self.diverged_dialogues = None
        self.reward_scorer = []
        self.setup_scorer()
        super().__init__(opt, shared)

    def setup_scorer(self):
        if not self.rl_mode or not self.rewards_list:
            return
        assert len(self.rewards_list) == len(self.rewards_weights)
        for i, r in enumerate(self.rewards_list):
            weight = self.rewards_weights[i]
            scorer = reward_funcs.REWARD_MAP[r](r, weight, use_cuda=self.use_cuda)
            self.reward_scorer.append(scorer)

    def get(self, episode_idx, entry_idx=None):
        action = super().get(episode_idx, entry_idx)
        action['model_answers'] = []
        histories_dialogues = []
        model_answers = []
        if len(self.history.dialogues) > 0:
            histories_dialogues.append(self.history.dialogues)
        if self.diverged_dialogues and len(self.diverged_dialogues.lineages) > 0:
            history_diverged_dialogues = self.diverged_dialogues.get_dialogues(active_only=True)
            histories_dialogues.extend(history_diverged_dialogues)
        if histories_dialogues:
            model_answers = self.get_model_answer(histories_dialogues, action)
        if model_answers:
            action['model_answers'] = model_answers
        if action['episode_done'] and model_answers:
            rewards = self.get_reward(model_answers, action)
            action['rewards'] = rewards
        return action

    def get_reward(self, model_answers, action):
        histories_dialogues = [self.history.dialogues] + self.diverged_dialogues.get_dialogues()
        for i, m_ans in enumerate(model_answers):
            histories_dialogues[i][-1].answer = m_ans['text']
            histories_dialogues[i][-1].cache = self.history.get_cache(m_ans)
        rewards = self.compute_rewards(histories_dialogues, last_action=action)
        return rewards

    def compute_rewards(self, conversations, last_action):
        rewards = {}
        for scorer in self.reward_scorer:
            rewards[scorer.name] = {"master": None, "diverged_rewards": None, "weight": scorer.weight, 'global': scorer.global_reward}
            master_rewards, diverged_rewards = scorer.reward(conversations, self.history, last_action=last_action,
                                                            agent_dictionary=self.dict)
            rewards[scorer.name]["master"] = master_rewards
            rewards[scorer.name]["diverged_rewards"] = diverged_rewards
        return rewards

    def get_master_dialogue(self, lastest_diverged, last_response):
        master = copy.copy(lastest_diverged[:-1])
        last_turn = copy.deepcopy(lastest_diverged[-1])
        last_turn.answer = last_response
        master.append(last_turn)
        return master


    def get_model_answer(self, histories_dialogues, action):
        retvals = []
        original_answer = action['text']
        original_question = action['single_label_text']
        for dialogues in histories_dialogues:
            obs = copy.copy(action)
            obs['text'] = dialogues[-1].question
            obs['single_label_text'] = original_answer
            tokenized_data = self.tokenize_from_history(obs, dialogues)
            vectorized_data = util.map_data(tokenized_data, self.dict)
            features = util.generate_features(tokenized_data, vectorized_data, self.model.args['max_turns'])
            if 'token_start_end' not in action:
                action['token_start_end'] = (tokenized_data['qas'][-1]['start'], tokenized_data['qas'][-1]['end'])
            obs['text_vec'] = features
            retvals.append(obs)
        # restore the ground truch question text
        action['text'] = original_answer
        action['single_label_text'] = original_question
        if retvals:
            batch = self.batchify(retvals)
            with torch.no_grad():
                _, _, reward_items, stats, preds = self.model(**self._model_input(batch))
            logits, outputs = preds['logits'], preds['outputs']
            # update each retval with the output answers and also swap the question and text key
            for i, retval in enumerate(retvals):
                token_start = int(preds['tokens_start_end'][0][i].data)
                token_end = int(preds['tokens_start_end'][1][i].data)
                retval['text'] = outputs[i]
                retval['single_label_text'] = original_question
                retval['yesno'] = int(logits['yesno'][i].argmax())
                retval['followup'] = int(logits['followup'][i].argmax())
                retval['token_start_end'] = (token_start, token_end+1)
                # retval['reward'] = reward.data
                # retval['reward_items'] = reward_items
        return retvals

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        super().observe(observation)
        if 'history' in observation:
            self.history = observation['history']
        if 'diverged_dialogues' in observation:
            self.diverged_dialogues = observation['diverged_dialogues']
        return observation
