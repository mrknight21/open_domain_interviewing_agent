from parlai.core.message import Message
from parlai.core.torch_agent import History, Optional
from collections import deque
import torch
import copy

class DialogueTurn(object):

    question: Optional[str]
    answer: Optional[str]
    log_prob: Optional[torch.LongTensor]
    reward: Optional[torch.LongTensor]
    complete: bool
    generated: bool

    def __init__(self, question_text, answer_text=None, log_prob=None, reward=None, cache=None):
        self.question = question_text
        self.answer = answer_text
        self.log_prob = log_prob
        self.reward = reward
        self.cache = cache
        self.generated = False
        self.complete = False
        if log_prob is not None:
            self.generated = True
        if self.question and self.answer:
            self.complete = True


    def update(self, answer_text=None, log_prob=None, reward=None, cache=None):
        if answer_text:
            self.answer = answer_text
            self.complete = True
        if log_prob is not None:
            self.log_prob = log_prob
            self.generated = True
        if reward is not None:
            self.reward = reward
        if cache is not None:
            self.cache = cache


class DialogueHistory(History):
    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        super().__init__(opt, **kwargs)
        self.context = None
        self.dialogues = []
        self.title = None
        self.background = None
        self.section_title = None

    def reset(self):
        """
        Clear the history.
        """
        self.history_raw_strings = []
        self.dialogues = []
        self.history_strings = []
        self.history_vecs = []
        self.context = None
        self.title = None
        self.background = None
        self.section_title = None

    def _update_dialogues(self, text, log_prob=None, reward=None, cache=None):
        """
        Update the history dialogue with te given observation.
        dialogue is a tuple with index 0 from the others and the index 1 from self
        :param text: the current observed utterance text
        """

        if self.size > 0:
            while len(self.dialogues) >= self.size/2:
                self.dialogues.pop(0)
        dialogue = DialogueTurn(question_text=text, log_prob=log_prob, reward=reward, cache=cache)
        if self.dialogues and not self.dialogues[-1].complete:
            self.dialogues[-1].update(text, log_prob=log_prob, reward=reward, cache=cache)
        else:
            self.dialogues.append(dialogue)

    def get_cache(self, obs):
        token_start_end = obs.get('token_start_end', None)
        if not token_start_end:
            character_start_end = obs.get('character_start_end', (-1, -1))
            token_start_end = (-1, -1)
        else:
            character_start_end = (-1, -1)
        cache = {
            'character_start_end':character_start_end,
            'token_start_end': token_start_end,
            'yesno': obs['yesno'], 'followup': obs['followup']}
        return cache

    def update_history(self, obs: Message, temp_history: Optional[str] = None):
        """
        Update the history with the given observation.

        :param obs:
            Observation used to update the history.
        :param temp_history:
            Optional temporary string. If it is not None, this string will be
            appended to the end of the history. It will not be in the history
            on the next dialogue turn. Set to None to stop adding to the
            history.
        """
        if "text" in obs and obs["text"] is not None:
            log_prob = obs.get('log_prob', None)
            reward = obs.get('reward', None)
            text = obs['text']
            cache = self.get_cache(obs)
            if not self.context and obs.get('context', None):
                    self.context = obs['context']
            if not self.background and obs.get('background', None):
                    self.background = obs['background']
            if not self.title and obs.get('title', None):
                    self.title = obs['title']
            if not self.section_title and obs.get('section_title', None):
                    self.section_title = obs['section_title']
            self._update_raw_strings(text)
            if self.add_person_tokens:
                text = self._add_person_tokens(
                    obs[self.field], self.p1_token, self.add_p1_after_newln
                )
            # update history string
            self._update_strings(text)
            # update history dialogues
            self._update_dialogues(text, log_prob=log_prob, reward=reward, cache=cache)
            # update history vecs
            self._update_vecs(text)
        self.temp_history = temp_history

    def get_history_vec(self):
        """
        Override from parent class to possibly add [SEP] token.
        """
        if not self.sep_last_utt or len(self.history_vecs) <= 1:
            return super().get_history_vec()

        history = deque(maxlen=self.max_len)
        for vec in self.history_vecs[:-1]:
            history.extend(vec)
            history.extend(self.delimiter_tok)
        history.extend([self.dict.end_idx])  # add [SEP] token
        history.extend(self.history_vecs[-1])

        return history

    def add_reply(self, text, log_prob=None, reward=None):
        """
        Add your own response to the history.
        """
        self._update_raw_strings(text)
        if self.add_person_tokens:
            text = self._add_person_tokens(text, self.p2_token)
        # update history string
        self._update_strings(text)
        # update history vecs
        self._update_vecs(text)
        # update history dialogues
        self._update_dialogues(text, log_prob=log_prob, reward=reward)


class Lineage(object):

    def __init__(self, dialogues):
        if not dialogues:
            self.dialogues = []
        else:
            self.dialogues = copy.deepcopy(dialogues)
        # indicate after which dialogue all dialogues are generated
        self.gen_start_index = len(self.dialogues)
        self.freeze = False

    def _update_dialogues(self, text, log_prob=None, reward=None, cache=None):
        """
        Update the history dialogue with te given observation.
        dialogue is a tuple with index 0 from the others and the index 1 from self
        :param text: the current observed utterance text
        """

        dialogue = DialogueTurn(question_text=text, log_prob=log_prob, reward=reward, cache=cache)
        if self.dialogues and not self.dialogues[-1].complete:
            self.dialogues[-1].update(text, log_prob=log_prob, reward=reward, cache=cache)
        else:
            self.dialogues.append(dialogue)

    def get_conversation(self):
        conversation = []
        for turn in self.dialogues:
            if turn.question:
                conversation.append(turn.question)
                if turn.answer:
                    conversation.append(turn.answer)
        return conversation

class DialogueLineages(object):

    def __init__(self):
        self.lineages = deque()

    def reset(self):
        self.lineages = deque()

    def get_cache(self, obs):
        cache = {
            'character_start_end': obs.get('character_start_end', None),
            'token_start_end': obs.get('token_start_end', None),
            'yesno': obs['yesno'], 'followup': obs['followup']}
        return cache

    def add_lineage(self, text, history, message=None, log_prob=None, reward=None, cache=None):
        new_lineage = Lineage(history.dialogues)
        if message:
            # create new lineage from ground truth answer
            cache = self.get_cache(message)
            reward = message
            log_prob = message.get('log_prob', None)
            reward = message.get('reward', None)
            text = message.get('text', "")
            new_lineage._update_dialogues(text, log_prob=log_prob, reward=reward, cache=cache)
        else:
            # create new lineage from ground truth question
            new_lineage._update_dialogues(text, log_prob=log_prob, reward=reward, cache=cache)
        self.lineages.appendleft(new_lineage)

    def get_dialogues(self, active_only=False):
        dialogues = []
        for l in self.lineages:
            if active_only:
                if not l.freeze:
                    dialogues.append(l.dialogues)
            else:
                dialogues.append(l.dialogues)
        return dialogues

