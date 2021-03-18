
import sys
import os
import torch
from parlai.core.message import Message
from parlai.core.torch_agent import Optional, Batch, Output
from parlai.core.metrics import AverageMetric
from parlai_internal.agents.torch_span_agent.torch_span_agent import TorchSpanAgent, DialogueHistory
from parlai_internal.utilities.flow_lstm_util.models.seq2seq import TeacherModel
from parlai_internal.utilities.flow_lstm_util import constants
from parlai_internal.utilities.flow_lstm_util.models.trainer import unpack_batch
from parlai_internal.utilities.flow_lstm_util import util
from parlai_internal.utilities.flow_lstm_util.dictionary_agent import InterviewDictionaryAgent



class IntervieweeHistory(DialogueHistory):

    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        super().__init__(opt, **kwargs)
        self.title = None
        self.background = None
        self.section_title = None
        self.history_cache = []

    def reset(self):
        """
        Clear the history.
        """
        self.history_raw_strings = []
        self.history_cache = []
        self.history_dialogues = []
        self.history_strings = []
        self.history_vecs = []
        self.context = None
        self.title = None
        self.background = None
        self.section_title = None

    def _update_cache(self, obs):
        cache = {
            'character_start_end': obs['character_start_end'],
            'yesno': obs['yesno'], 'followup': obs['followup']}
        self.history_cache.append(cache)

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
            if not self.context and obs.get('context', None):
                    self.context = obs['context']
            if not self.background and obs.get('background', None):
                    self.background = obs['background']
            if not self.title and obs.get('title', None):
                    self.title = obs['title']
            if not self.section_title and obs.get('section_title', None):
                    self.section_title = obs['section_title']
            text = obs['text']
            self._update_raw_strings(text)
            if self.add_person_tokens:
                text = self._add_person_tokens(
                    obs[self.field], self.p1_token, self.add_p1_after_newln
                )
            # update history string
            self._update_strings(text)
            # update history dialogues
            self._update_dialogues(text)
            # update history vecs
            self._update_vecs(text)
            self._update_cache(obs)
        self.temp_history = temp_history

class IntervieweeAgent(TorchSpanAgent):
    """
    Interviewee agent.

    This agent uses the QA pretrained Teacher model from Qi et al 2020,
    https://github.com/qipeng/stay-hungry-stay-focused as the interviewee.
    This agent is only expected to be used for evaluation and reinforcement learning
    If additional training is required, we would prefer to do the training with the original code
    and then use the model in Parl AI.
    """

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return InterviewDictionaryAgent

    def load_teacher(self):
        model = None
        try:
            print(f"Loading answerer/teacherriminator model from '{constants.TEACHER_FILE}'...")
            config_path = os.path.join(self.opt['datapath'], constants.FINE_TUNE_FILE)
            teacher_path = os.path.join(self.opt['datapath'], constants.TEACHER_FILE)
            model_checkpoint = torch.load(config_path, lambda storage, loc: storage)
            teacher_checkpoint = torch.load(teacher_path, lambda storage, loc: storage)
            config = model_checkpoint['config']
            config['teacher_elmo'] = False
            model = TeacherModel(config, use_cuda=not self.opt['no_cuda'])
            model.load_state_dict(teacher_checkpoint['model'], strict=False)
        except BaseException:
            import pdb
            pdb.set_trace()
            print("Cannot answerer/teacherriminator load model from {}".format(constants.TEACHER_FILE))
            sys.exit(1)
        return model

    def build_model(self):
        """
        Construct the model.
        """
        model = self.load_teacher()
        self.criterion = None
        return model

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return IntervieweeHistory

    def build_history(self):
        """
        Return the constructed history object.
        """
        # Note that contants.SEP does not exist in the original teacher model.
        # It is a temporary measure ro fulfull the history delimiator requirement
        # , and will require to be replaced or removed before vectorization.
        self.opt['delimiter'] = self.dict.sep_token
        history = self.history_class()(
            self.opt,
            maxlen=self.opt['text_truncate'],
            size=self.opt['history_size'],
            p1_token=self.P1_TOKEN,
            p2_token=self.P2_TOKEN,
            dict_agent=self.dict,
        )
        history.delimiter_tok = self.dict.sep_idx
        return history

    def _set_text_vec(self, obs, history, truncate, is_training=True):
        tokenized_data = self.tokenize_from_history(obs)
        vectorized_data = util.map_data(tokenized_data, self.dict)
        features = util.generate_features(tokenized_data, vectorized_data, self.model.args['max_turns'])
        obs['text_vec'] = features
        labels_with_special_tokens = []
        for l in obs.get('labels', obs.get('eval_labels', [])):
            if l == "":
                labels_with_special_tokens.append(dict.cls_token)
            else:
                labels_with_special_tokens.append(l)
        if 'labels' in obs:
            obs.force_set('labels', labels_with_special_tokens)
        elif 'eval_labels' in obs:
            obs.force_set('eval_labels', labels_with_special_tokens)
        return obs

    
    def batchify(self, obs_batch, sort=False):
        batch = Batch(batchsize=0)
        if len(obs_batch) == 0:
            return batch
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]
        if len(valid_obs) == 0:
            return batch
        valid_inds, exs = zip(*valid_obs)
        src_text_numb = [len(obs['text_vec']['src_text']) for obs in obs_batch]
        retval = self._collate_fn(obs_batch)
        batch = Batch(
            batchsize=len(valid_obs),
            valid_indices=valid_inds,
            no_answer_reply=obs_batch[0].get('no_answer_reply', 'CANNOTANSWER'),
            src=retval['src'],
            src_char=retval['src_char'],
            src_text=retval['src_text'],
            bg=retval['bg'],
            bg_char=retval['bg_char'],
            bg_text=retval['bg_text'],
            tgt_in=retval['tgt_in'],
            tgt_out=retval['tgt_out'],
            tgt_out_char=retval['tgt_out_char'],
            tgt_text=retval['tgt_text'],
            turn_ids=retval['turn_ids'],
            ctx=retval['ctx'],
            ctx_char=retval['ctx_char'],
            ctx_text=retval['ctx_text'],
            start=retval['start'],
            end=retval['end'],
            yesno=retval['yesno'],
            followup=retval['followup'],
            this_turn=retval['this_turn'],
            ans_mask=retval['ans_mask']
        )
        return batch


    def _model_input(self, batch):
        inputs = unpack_batch(batch, self.use_cuda)
        inputs = {k: c for k, c in inputs.items() if c is not None}
        src, tgt_in, tgt_out, turn_ids, ctx = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids'], inputs['ctx']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constants.PAD_ID)
        bg_mask = bg.eq(constants.PAD_ID) if bg is not None else None
        batch_size = batch.batchsize
        return {'src': src, 'src_mask': src_mask, 'turn_ids': turn_ids, 'tgt_in': tgt_in,
                'bg': bg, 'bg_mask': bg_mask, 'tgt_out': tgt_out, 'ctx': ctx, 'ans_mask': inputs['ans_mask'],
                'start': inputs['start'], 'end': inputs['end'], 'this_turn': inputs['this_turn'],
                'src_char': inputs['src_char'], 'tgt_out_char': inputs['tgt_out_char'], 'ctx_char': inputs['ctx_char'],
                'bg_char': inputs.get('bg_char', None), 'yesno': inputs['yesno'], 'followup': inputs['followup'],
                'src_text': batch.src_text, 'bg_text': batch.bg_text,
                'tgt_text': batch.tgt_text, 'ctx_text': batch.ctx_text}

    def compute_loss(self, batch, return_output=False):
        loss, reward, reward_items, stats, preds = self.model(**self._model_input(batch))
        outputs = {'reward': reward, 'reward_items': reward_items, 'stats': stats, 'pred':preds}
        batches_count = [1] * batch.batchsize
        self.record_local_metric('loss',
                                 AverageMetric.many([loss.data.cpu()] * batch.batchsize, batches_count))
        if return_output:
            return loss, outputs
        else:
            return loss

    def eval_step(self, batch):
        if batch.batchsize <= 0:
            return
        else:
            bsz = batch.batchsize
        self.model.eval()
        loss, outputs = self.compute_loss(batch, return_output=True)
        batch_best_preds = outputs['pred']['outputs']
        return Output(batch_best_preds)

    def tokenize_from_history(self, item=None,  history=None):
        if not history:
            history = self.history
        strings_to_tokenize = [history.title, history.section_title, history.context, history.background]
        qas = []
        if len(history.dialogues) > 1:
            for qa in self.history.history_dialogues[:-1]:
                strings_to_tokenize.append(qa.question)
                strings_to_tokenize.append(qa.answer)
                qas.append((qa.question, qa.answer))
        strings_to_tokenize.append(item['text'])
        strings_to_tokenize.append(item['single_label_text'])
        qas.append((item['text'], item['single_label_text']))
        tokenized, offsets = self.dict.bulk_tokenize(strings_to_tokenize, return_offsets=True)
        retval = {'title': tokenized[0], 'section_title': tokenized[1], 'context': tokenized[2], 'background': tokenized[3], 'qas':[]}
        tokenized = tokenized[4:]
        ctx_offsets = [(st-offsets[2][0][0], en-offsets[2][0][0]) for st, en in offsets[2]]
        parsed_idx = 0
        for idx, qa in enumerate(qas):
            if idx < len(qas) -1:
                cache = self.history.history_cache[idx]
            else:
                cache = item
            item_yesno = cache.get('yesno')
            item_followup = cache.get('followup')
            ans_st = -1
            ans_en = -1
            char_st = cache.get('character_start_end')[0]
            char_en = cache.get('character_start_end')[1]
            ans = tokenized[1]
            if item_yesno == '__YES__':
                ans = ['Yes', ','] + tokenized[1]
                yesno = constants.YESNO_TO_ID['y']
            elif item_yesno == '__NO__':
                ans = ['No', ','] + tokenized[1]
                yesno = constants.YESNO_TO_ID['n']
            else:
                yesno = constants.YESNO_TO_ID['x']
            if item_followup == '__SHOULDNOT__':
                followup = constants.FOLLOWUP_TO_ID['n']
            elif item_followup == '__SHOULD__':
                followup = constants.FOLLOWUP_TO_ID['f']
            else:
                followup = constants.FOLLOWUP_TO_ID['m']
            for idj, (st, en) in enumerate(ctx_offsets):
                if en > char_st and ans_st < 0:
                    ans_st = idj
                if st >= char_en and ans_en < 0:
                    ans_en = idj
            if ans_en < 0:
                ans_en = len(ctx_offsets)
            assert ''.join(tokenized[1]) in ''.join(retval['context'][ans_st:ans_en]), '{} {}'.format(str(retval['context'][ans_st:ans_en]), str(tokenized[1]))
            retval['qas'].append({'question': tokenized[0], 'answer': ans,
            'start': ans_st, 'end': ans_en, 'yesno': yesno, 'followup': followup})
            tokenized = tokenized[2:]
            offsets = offsets[2:]
            parsed_idx += 2
        return retval

    def _collate_fn(self, obs_batch):
        batch_data = [x['text_vec'] for x in obs_batch]
        src = torch.from_numpy(util.pad_lists([x['src_idx'] for x in batch_data]))
        src_char = torch.from_numpy(util.pad_lists([x['src_char'] for x in batch_data]))
        ctx = torch.from_numpy(util.pad_lists([x['ctx_idx'] for x in batch_data]))
        ctx_char = torch.from_numpy(util.pad_lists([x['ctx_char'] for x in batch_data]))
        tgt_in = torch.from_numpy(util.pad_lists([x['tgt_in_idx'] for x in batch_data]))
        tgt_in_char = torch.from_numpy(util.pad_lists([x['tgt_in_char'] for x in batch_data]))
        tgt_out = torch.from_numpy(util.pad_lists([x['tgt_out_idx'] for x in batch_data]))
        tgt_out_char = torch.from_numpy(util.pad_lists([x['tgt_out_char'] for x in batch_data]))
        # neg_out = torch.from_numpy(self.pad_lists([x['neg_out_idx'] for x in batch_data]))
        # neg_out_char = torch.from_numpy(self.pad_lists([x['neg_out_char'] for x in batch_data]))

        this_turn = torch.from_numpy(util.pad_lists([x['this_turnid'] for x in batch_data]))
        ans_mask = torch.from_numpy(util.pad_lists([x['ans_mask'] for x in batch_data]))

        turn_ids = torch.from_numpy(util.pad_lists([x['turn_ids'] for x in batch_data], fill_val=-1))
        start = torch.from_numpy(util.pad_lists([[x['start'][-1]] for x in batch_data], fill_val=-1))
        end = torch.from_numpy(util.pad_lists([[x['end'][-1]] for x in batch_data], fill_val=-1))
        yesno = torch.from_numpy(util.pad_lists([[x['yesno'][-1]] for x in batch_data], fill_val=-1))
        followup = torch.from_numpy(util.pad_lists([[x['followup'][-1]] for x in batch_data], fill_val=-1))

        retval = {'src': src,
                  'src_char': src_char,
                  'src_text': [x['src_text'] for x in batch_data],
                  'tgt_in': tgt_in,
                  'tgt_out': tgt_out,
                  'tgt_out_char': tgt_out_char,
                  'tgt_text': [x['tgt_text'] for x in batch_data],
                  'turn_ids': turn_ids,
                  'ctx': ctx,
                  'ctx_char': ctx_char,
                  'ctx_text': [x['ctx_text'] for x in batch_data],
                  'start': start,
                  'end': end,
                  'yesno': yesno,
                  'followup': followup,
                  'this_turn': this_turn,
                  'ans_mask': ans_mask}
        if 'bg_text' in batch_data[0]:
            bg = torch.from_numpy(util.pad_lists([x['bg_idx'] for x in batch_data]))
            bg_char = torch.from_numpy(util.pad_lists([x['bg_char'] for x in batch_data]))
            retval['bg'] = bg
            retval['bg_char'] = bg_char
            retval['bg_text'] = [x['bg_text'] for x in batch_data]
        return retval
