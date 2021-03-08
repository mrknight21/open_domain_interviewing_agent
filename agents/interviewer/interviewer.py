import os
import torch
from parlai.core.message import Message
from parlai.core.torch_agent import Optional, Batch, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent, PPLMetric
from parlai_internal.utilities.flow_lstm_util.dictionary_agent import InterviewDictionaryAgent
from parlai.core.metrics import AverageMetric
from parlai_internal.agents.interviewee.interviewee import IntervieweeHistory
from parlai_internal.utilities.flow_lstm_util.models.seq2seq import Seq2SeqModel
from parlai_internal.utilities.flow_lstm_util.models import trainer
from parlai_internal.utilities.flow_lstm_util import constants
from parlai_internal.utilities.flow_lstm_util import util



class InterviewerHistory(IntervieweeHistory):

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
            if text:
                if "|" in text:
                    texts = text.split('|')
                    text = max(texts, key=len)
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



class InterviewerAgent(TorchGeneratorAgent):
    """
    Interviewer agent.
    """

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return InterviewDictionaryAgent

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
            maxlen=self.text_truncate,
            size=self.histsz,
            p1_token=self.P1_TOKEN,
            p2_token=self.P2_TOKEN,
            dict_agent=self.dict,
        )
        history.delimiter_tok = self.dict.sep_idx
        return history

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return InterviewerHistory

    def build_model(self):
        """
        Construct the model.
        """
        model = self.load_question_generation_model()
        return model

    def build_criterion(self):
        """
        Construct and return the loss function.

        By default torch.nn.CrossEntropyLoss.

        If overridden, this model should produce a sum that can be used for a per-token loss.
        """
        return torch.nn.NLLLoss(ignore_index=constants.PAD_ID, reduction='none')

    def load_question_generation_model(self):
        filename = self.opt['init_model']
        if not filename:
            filename = os.path.join(self.opt['datapath'], constants.FINE_TUNE_FILE)
        print(f"Loading model from '{filename}'...")
        checkpoint = torch.load(filename, lambda storage, loc: storage)
        args = checkpoint['config']
        if self.dict.vocab is not None:
            args['vocab'] = self.dict.vocab
        model = Seq2SeqModel(args, use_cuda=self.use_cuda)
        model.load_state_dict(checkpoint['model'])
        return model

    def _set_text_vec(self, obs, history, truncate, is_training=True):
        tokenized_data = self.tokenize_from_history(obs)
        vectorized_data = util.map_data(tokenized_data, self.dict)
        features = util.generate_features(tokenized_data, vectorized_data, self.model.args['max_turns'])
        obs['text_vec'] = features
        labels_with_special_tokens = []
        for l in obs.get('labels', obs.get('eval_labels', [])):
            if l == "":
                labels_with_special_tokens.append("")
            else:
                labels_with_special_tokens.append(l)
        if 'labels' in obs:
            obs.force_set('labels', labels_with_special_tokens)
        elif 'eval_labels' in obs:
            obs.force_set('eval_labels', labels_with_special_tokens)
        return obs

    def _set_label_vec(self, obs, add_start, add_end, truncate):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        return obs

    def is_valid(self, obs):
        """
        Determine if an observation is valid or not.
        """
        valid = True
        labels = obs.get('labels', obs.get('eval_labels', []))
        if labels:
            if len(labels) == 1 and labels[0] == "":
                valid = False
        else:
            valid = False
        return valid

    def batchify(self, obs_batch, sort=False):
        is_training = self.is_training
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
            this_turn=retval['this_turn'],
            label_vec=retval['tgt_out'],
            labels=retval['tgt_text'],
        )
        return batch

    def compute_loss(self, batch, return_output=False):
        input = self._model_input(batch)
        loss, reward, reward_items, stats, preds = self.model(input)
        outputs = {'reward': reward, 'reward_items': reward_items, 'stats': stats, 'pred': preds}
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
        input = self._model_input(batch)
        # only the output of the last turn
        tgt_out = input['tgt_out'][:, -1:, :]
        log_probs = self.model(**input)[-1:, :, :]
        loss = self.criterion(log_probs.view(-1, self.dict.vocab_size), tgt_out.view(-1))
        pred_seqs, _ = self.predict(batch)
        texts = [" ".join(seq) for seq in pred_seqs]
        text = texts[-1]
        # preds = torch.stack(preds[-1])
        # save loss to metrics
        notnull = tgt_out.ne(self.dict.null_idx)
        target_tokens = notnull.long().sum(dim=-1).view(1)
        # correct = ((tgt_out == preds) * notnull).sum(dim=-1)
        loss = loss.view(log_probs.shape[:-1]).sum(dim=1)
        #
        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        # self.record_local_metric(
        #     'token_acc', AverageMetric.many(correct, target_tokens)
        # )
        return Output(text)

    def tokenize_from_history(self, item):
        strings_to_tokenize = [self.history.title, self.history.section_title, self.history.context, self.history.background]
        qas = []
        if len(self.history.history_dialogues) > 0:
            for qa in self.history.history_dialogues:
                strings_to_tokenize.append(qa[0])
                strings_to_tokenize.append(qa[1])
                qas.append((qa[0], qa[1]))
        strings_to_tokenize.append(item['single_label_text'])
        strings_to_tokenize.append(item['text'])
        qas.append((item['single_label_text'], ""))
        tokenized, offsets = self.dict.bulk_tokenize(strings_to_tokenize, return_offsets=True)
        retval = {'title': tokenized[0], 'section_title': tokenized[1], 'context': tokenized[2], 'background': tokenized[3], 'qas':[]}
        tokenized = tokenized[4:]
        parsed_idx = 0
        for idx, qa in enumerate(qas):
            retval['qas'].append({'question': tokenized[0], 'answer': tokenized[1]})
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
        this_turn = torch.from_numpy(util.pad_lists([x['this_turnid'] for x in batch_data]))
        turn_ids = torch.from_numpy(util.pad_lists([x['turn_ids'] for x in batch_data], fill_val=-1))

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
                  'this_turn': this_turn}
        if 'bg_text' in batch_data[0]:
            bg = torch.from_numpy(util.pad_lists([x['bg_idx'] for x in batch_data]))
            bg_char = torch.from_numpy(util.pad_lists([x['bg_char'] for x in batch_data]))
            retval['bg'] = bg
            retval['bg_char'] = bg_char
            retval['bg_text'] = [x['bg_text'] for x in batch_data]
        return retval

    def _model_input(self, batch):
        inputs = trainer.unpack_batch(batch, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constants.PAD_ID)
        bg_mask = bg.eq(constants.PAD_ID) if bg is not None else None
        batch_size = batch.batchsize
        return {'src': src, 'src_mask': src_mask, 'turn_ids': turn_ids, 'tgt_in': tgt_in,
                'bg': bg, 'bg_mask': bg_mask, 'tgt_out': tgt_out}

    def predict(self, batch, beam_size=1, return_pair_level=False, return_preds=False, return_rewards=False):
        inputs = trainer.unpack_batch(batch, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constants.PAD_ID)
        bg_mask = bg.eq(constants.PAD_ID) if bg is not None else None

        if not return_preds:
            self.model.eval()
        batch_size = src.size(0)
        preds = self.model.predict(src, src_mask, turn_ids, beam_size=beam_size, bg=bg, bg_mask=bg_mask, return_pair_level=return_pair_level)
        pred_seqs = [[self.dict.ind2tok[id_] for id_ in ids] for ids in preds] # unmap to tokens
        pred_seqs = util.prune_decoded_seqs(pred_seqs)
        return pred_seqs, preds

    def sample(self, batch, top_p=1, return_pair_level=False, return_preds=False):
        inputs = trainer.unpack_batch(batch, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        src_mask = src.eq(util.constant.PAD_ID)
        bg_mask = bg.eq(util.constant.PAD_ID) if bg is not None else None

        if not return_preds:
            self.model.eval()
        batch_size = src.size(0)
        preds = self.model.sample(src, src_mask, turn_ids, top_p=top_p, bg=bg, bg_mask=bg_mask, return_pair_level=return_pair_level)
        preds, nll = preds
        pred_seqs = [[self.vocab['id2word'][id_] for id_ in ids] for ids in preds] # unmap to tokens
        pred_seqs = util.prune_decoded_seqs(pred_seqs)
        return pred_seqs, preds, nll
