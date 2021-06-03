import os
import copy
from parlai_internal.agents.interviewee.interviewee import IntervieweeAgent
from parlai_internal.utilities.dialogue_history import DialogueTurn
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.utils.misc import warn_once
from parlai_internal.utilities.flow_lstm_util import util
from parlai_internal.reward_funcs import REWARD_MAP, DEFAULT_REWARD_LIST, DEFAULT_EVA_LIST
from .build import build
import torch

NO_ANSWER_REPLY = "CANNOTANSWER"


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
        is_last_episode = episode_idx == len(self.episodes) -1
        context_token_weights = ex.get("context_token_weights", None)
        if context_token_weights:
            context_token_weights = eval(context_token_weights)
            if len(context_token_weights) > 0:
                context_token_weights = context_token_weights
            else:
                context_token_weights = None
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
            'character_start_end': char_start_end,
            'context_token_weights': context_token_weights,
            'is_last_episode': is_last_episode
        }
        return action

class ReinforcementLearningTeacherAgent(DefaultTeacher, IntervieweeAgent):

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.dict = self.build_dictionary()
        self.query_truncate = opt['query_maximum_length']
        self.context_truncate = opt['context_maximum_length']
        self.history_truncate = opt['history_maximum_length']
        self.self_critic = opt['rl_baseline_method']
        self.history = self.build_history()
        self.rl_mode = opt['reinforcement_learning']
        self.exploration_steps = opt['exploration_steps']
        self.use_cuda = not opt['no_cuda']
        # now set up any fields that all instances may need
        self.EMPTY = torch.zeros(0, dtype=torch.long)
        self.NULL_IDX = self.dict[self.dict.null_token]
        self.START_IDX = self.dict[self.dict.start_token]
        self.END_IDX = self.dict[self.dict.end_token]
        if opt['datatype']== 'valid':
            self.rewards_list = opt.get('rewards_list', DEFAULT_EVA_LIST)
        else:
            self.rewards_list = opt.get('rewards_list', DEFAULT_REWARD_LIST)
        self.model = self.build_model()
        self.model.eval()
        if self.use_cuda:
            self.model.cuda()
        self.truncate = self.dict.maxtokens
        self.diverged_dialogues = None
        self.reward_scorer = []
        self.setup_scorer()
        super().__init__(opt, shared)


    def setup_scorer(self):
        if not self.rl_mode or not self.rewards_list:
            return
        for i, r in enumerate(self.rewards_list):
            scorer = REWARD_MAP[r](r, use_cuda=self.use_cuda)
            self.reward_scorer.append(scorer)

    def get(self, episode_idx, entry_idx=None):
        action = super().get(episode_idx, entry_idx)
        action['model_answers'] = []
        histories_dialogues = []
        model_answers = []
        g_model_answers = []
        if len(self.history.dialogues) > 0:
            histories_dialogues.append(self.history.dialogues)
        if self.diverged_dialogues and len(self.diverged_dialogues.lineages) > 0:
            history_diverged_dialogues = self.diverged_dialogues.get_dialogues(active_only=True)
            histories_dialogues.extend(history_diverged_dialogues)
        if histories_dialogues:
            model_answers = self.get_model_answer(histories_dialogues, action)
            if self.self_critic and self.check_greedy_questions(histories_dialogues):
                g_model_answers = self.get_model_answer(histories_dialogues, action, use_greedy_question=True)
        if model_answers:
            action['model_answers'] = model_answers
        if g_model_answers:
            action['g_model_answers'] = g_model_answers
        if action['episode_done'] and model_answers:
            rewards = self.get_reward(model_answers, action, greedy_answers=g_model_answers)
            action['rewards'] = rewards
        return action

    def get_reward(self, model_answers, action, greedy_answers=None):
        histories_dialogues = [self.history.dialogues] + self.diverged_dialogues.get_dialogues()
        greedy_conversation = None
        use_greedy = self.self_critic and greedy_answers
        ans_count = len(model_answers)
        for i, m_ans in enumerate(model_answers):
            # Reserve for the longest lineage when doing validaiton
            if self.datatype == 'valid' or self.datatype == 'test':
                if ans_count == i+1:
                    histories_dialogues[-1][-1].answer = m_ans['text']
                    histories_dialogues[-1][-1].cache = self.history.get_cache(m_ans)
                    histories_dialogues[-1][-1].reward = m_ans['reward_items']
                    continue
            histories_dialogues[i][-1].answer = m_ans['text']
            histories_dialogues[i][-1].cache = self.history.get_cache(m_ans)
            histories_dialogues[i][-1].reward = m_ans['reward_items']
            greedy_output = None
            if use_greedy:
                greedy_output = {'answer': greedy_answers[i]['text'], 'cache': self.history.get_cache(greedy_answers[i]), 'reward': greedy_answers[i]['reward_items']}
            histories_dialogues[i][-1].update(greedy_output=greedy_output)
        if use_greedy:
            greedy_conversation = self.get_greedy_conversation(histories_dialogues)
        rewards = self.compute_rewards(histories_dialogues, last_action=action, greedy_conversation=greedy_conversation)
        return rewards

    def compute_rewards(self, conversations, last_action, greedy_conversation=None):
        rewards = {}
        for scorer in self.reward_scorer:
            rewards[scorer.name] = {"master": None, "diverged_rewards": None, "greedy_rewards": None,
                                    'global': scorer.global_reward, 'required_normalise':scorer.required_normalise}
            master_rewards, diverged_rewards = scorer.reward(conversations, self.history, last_action=last_action,
                                                            agent_dictionary=self.dict)
            if self.self_critic and greedy_conversation:
                _, greedy_rewards = scorer.reward(greedy_conversation, self.history, last_action=last_action,
                                                                 agent_dictionary=self.dict)
                rewards[scorer.name]["greedy_rewards"] = greedy_rewards
            rewards[scorer.name]["master"] = master_rewards
            rewards[scorer.name]["diverged_rewards"] = diverged_rewards
        return rewards

    def get_master_dialogue(self, lastest_diverged, last_response):
        master = copy.copy(lastest_diverged[:-1])
        last_turn = copy.deepcopy(lastest_diverged[-1])
        last_turn.answer = last_response
        master.append(last_turn)
        return master


    def get_model_answer(self, histories_dialogues, action, use_greedy_question=False):
        retvals = []
        original_answer = action['text']
        original_question = action['single_label_text']
        for dialogues in histories_dialogues:
            obs = copy.copy(action)
            if use_greedy_question and dialogues[-1].greedy_question:
                obs['text'] = dialogues[-1].greedy_question
            else:
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
                retval['reward_items'] = {r_name: float(r[i].detach().cpu()) for r_name, r in reward_items.items()}
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

    def check_greedy_questions(self, histories_dialogues):
        return any([t.greedy_question for dialogue in histories_dialogues for t in dialogue])

    def get_greedy_conversation(self, conversations):
        greedy_conversation = []
        for conv in conversations:
            greedy_conversation.append([])
            for d in conv:
                if d.greedy_question:
                    new_d = DialogueTurn(question_text=d.greedy_question, answer_text=d.greedy_answer, reward=d.greedy_reward, cache=d.greedy_cache)
                    new_d.generated = True
                    new_d.complete = True
                    greedy_conversation[-1].append(new_d)
                else:
                    greedy_conversation[-1].append(d)
        return greedy_conversation
