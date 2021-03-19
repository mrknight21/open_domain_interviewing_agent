import os
from typing import List, Tuple
import torch
import copy
from parlai.core.message import Message
from parlai.core.torch_agent import Optional, Batch, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent, PPLMetric, TorchGeneratorModel
from parlai_internal.utilities.flow_lstm_util.dictionary_agent import InterviewDictionaryAgent
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
from parlai_internal.agents.interviewee.interviewee import IntervieweeHistory
from parlai_internal.utilities.flow_lstm_util.models.seq2seq import Seq2SeqModel
from parlai_internal.utilities.flow_lstm_util.models import trainer
from parlai_internal.utilities.flow_lstm_util import constants, util
from parlai_internal.utilities.dialogue_history import MultiDialogueHistory
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt



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
            log_prob = obs.get('log_prob', None)
            reward = obs.get('reward', None)
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
                self._update_dialogues(text, log_prob=log_prob, reward=reward)
                # update history vecs
                self._update_vecs(text)
                self._update_cache(obs)
        self.temp_history = temp_history

class Sq2SqQuestionGenerationModel(TorchGeneratorModel):

    def __init__(self, opt, dict):
        # self.add_start_token = opt["add_start_token"]
        super().__init__(*self._get_special_tokens(opt, dict))

        # init the model
        self.sq2sq_model = self.load_question_generation_model(opt, dict)
        self.config = self.sq2sq_model.args

    def encoder(self, src, src_mask, bg, bg_mask, turn_ids=None, last_state_only=True):
        # prepare for encoder/
        h_in, src_mask, (hn, cn), h_bg = self.sq2sq_model.encode_sources(src, src_mask, bg, bg_mask, turn_ids=turn_ids)
        if last_state_only:
            return {'h_in': h_in[-1:, :, :], 'src_mask': src_mask[-1:, :], 'hn': hn[:, -1:, :], 'cn': cn[:, -1:, :],
                    'h_bg': h_bg[-1:, :], 'turn_ids': turn_ids[:, -1:, :]}
        else:
            return {'h_in': h_in, 'src_mask': src_mask, 'hn': hn, 'cn': cn,
                    'h_bg': h_bg, 'turn_ids': turn_ids}

    def decoder(self, decoder_input, encoder_states, incr_state):
        hids = []
        decoder_input = self.sq2sq_model.emb_drop(self.sq2sq_model.embedding(decoder_input))
        h_in, src_mask, hn, cn, h_bg, turn_ids = encoder_states['h_in'], encoder_states['src_mask'], \
                                                 encoder_states['hn'], encoder_states['cn'], encoder_states['h_bg'],\
                                                 encoder_states['turn_ids']
        if incr_state:
            dec_hidden, hids = incr_state['dec_hidden'], incr_state['hids']
            hn, cn = dec_hidden
        log_probs, dec_hidden, attn, h_out = self.sq2sq_model.decode(decoder_input, hn, cn, h_in, src_mask,
                turn_ids=turn_ids, previous_output=None if len(hids) == 0 else hids, h_bg=h_bg)
        hids.append(h_out.squeeze(1))
        incr_state = {'dec_hidden': dec_hidden, 'hids': hids}
        return log_probs, incr_state

    def load_question_generation_model(self, opt, dict):
        filename = opt['init_model']
        if not filename:
            filename = os.path.join(opt['datapath'], constants.FINE_TUNE_FILE)
        print(f"Loading model from '{filename}'...")
        checkpoint = torch.load(filename, lambda storage, loc: storage)
        args = checkpoint['config']
        if dict.vocab is not None:
            args['vocab'] = dict.vocab
        model = Seq2SeqModel(args, use_cuda=not opt['no_cuda'])
        model.load_state_dict(checkpoint['model'])
        return model

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        encs = {}
        for name, states in encoder_states.items():
            if name == 'turn_ids':
                encs[name] = states
            elif isinstance(states, torch.Tensor):
                encs[name] = torch.index_select(states, 0, indices)
            else:
                encs[name] = (torch.index_select(states[0], 0, indices), torch.index_select(states[1], 0, indices))
        return encs

    def output(self, tensor):
        """
        Compute output logits.
        """
        return tensor

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = {}
        for name, states in incremental_state.items():
            if name == 'dec_hidden':
                new_incr_state[name] = (torch.index_select(states[0], 0, inds), torch.index_select(states[1], 0, inds))
            if name == 'hids':
                new_incr_state[name] = [torch.index_select(hid, 0, inds) for hid in states]
        return new_incr_state

    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        if self.add_start_token:
            return super().decode_forced(encoder_states, ys)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def forward(self, xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        src = xs['src']
        src_mask = xs['src_mask']
        turn_ids = xs ['turn_ids']
        tgt_in = xs['tgt_in']
        bg = xs.get('bg', None)
        bg_mask = xs.get('bg_mask', None)
        # prepare for encoder/decoder
        B, T, L = tgt_in.size()
        tgt_in = tgt_in.view(B*T, L)
        dec_inputs = self.sq2sq_model.emb_drop(self.sq2sq_model.embedding(tgt_in))

        h_in, src_mask, (hn, cn), h_bg = self.sq2sq_model.encode_sources(src, src_mask, bg, bg_mask, turn_ids=turn_ids)

        log_probs, _, dec_attn, _ = self.sq2sq_model.decode(dec_inputs, hn, cn, h_in, src_mask, turn_ids=turn_ids, h_bg=self.sq2sq_model.drop(h_bg))
        _, preds = log_probs.max(dim=2)
        return log_probs[-1:, :, :], preds, {'h_in':h_in, 'src_mask': src_mask, 'hn': hn, 'cn': cn, 'h_bg': h_bg}



class InterviewerAgent(TorchGeneratorAgent):
    """
    Interviewer agent.
    """

    @classmethod
    def add_cmdline_args(cls, parser, partial_opt: Optional[Opt] = None) -> ParlaiParser:
        """
        Add CLI args.
        """
        TorchGeneratorAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group('Torch Span Classifier Arguments')
        # interactive mode
        parser.add_argument(
            '--print-scores',
            type='bool',
            default=False,
            help='print probability of chosen class during ' 'interactive mode',
        )
        # miscellaneous arguments
        parser.add_argument(
            '--reinforcement-learning',
            type='bool',
            default=False,
            help='train with reinforcement learning',
        )
        # query maximum length
        parser.add_argument(
            '--exploration-steps',
            type=int,
            default=0,
            help='maximum number of deviation turns allowed for history',
        )

    def __init__(self, opt: Opt, shared=None):
        self.rl_mode = opt['reinforcement_learning']
        self.exploration_steps = opt['exploration_steps']
        super().__init__(opt, shared)

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
        if self.rl_mode and self.exploration_steps >0:
            self.diverged_history = MultiDialogueHistory(self.opt, self.history_class())
        return history

    def diverged_history_update(self, observation):
        """
        This method update or harvest the diverged lineage history.
        The first answer is the model output for the most recent teacher force history
        The rest items follwo the index of the lineages
        :param model_outputs: observation with 'model_answer'
        :return: Update the diverged_history object
        """
        if 'model_answers' not in observation:
            return None
        model_output = observation['model_answers']
        for i, retval in enumerate(model_output):
            text = retval['text']
            reward = retval['reward']
            reward_items = retval['reward_items']
            if i == 0:
                self.diverged_history.add_lineage(text, self.history, message=retval, reward=reward)
            else:
                self.diverged_history.lineages[i].update_history(retval)



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
        return Sq2SqQuestionGenerationModel(self.opt, self.dict)

    def build_criterion(self):
        """
        Construct and return the loss function.

        By default torch.nn.CrossEntropyLoss.

        If overridden, this model should produce a sum that can be used for a per-token loss.
        """
        return torch.nn.NLLLoss(ignore_index=constants.PAD_ID, reduction='none')

    def observe(self, observation):
        if self.rl_mode and len(self.diverged_history.lineages) > 0:
            self.diverged_history_update(observation)
        super().observe(observation)


    def self_observe(self, self_message: Message) -> None:
        """
        Observe one's own utterance.

        This is used so that the agent can incorporate its own response into
        the dialogue history after a batch_act. Failure to implement this will
        result in an agent that cannot hear itself speak.

        :param self_message:
            The message corresponding to the output from batch_act.
        """
        if self.rl_mode and self.exploration_steps > 0 and self_message["text"] and not self_message['episode_done']:
            self.diverged_history.add_lineage(self_message["text"], self.history, log_prob=self_message.get("log_probs", None))
            self_message['history'] = self.history
            self_message['diverged_history'] = self.diverged_history
        super().self_observe(Message)

    def _set_text_vec(self, obs, history, truncate, is_training=True):
        histories = [self.history]
        original_answer = obs['text']
        original_question = obs['single_label_text']
        retvals = []
        if len(self.diverged_history.lineages) > 0:
            histories += self.diverged_history.lineages
        for hist in histories:
            retval = copy.copy(obs)
            if hist.dialogues:
                retval.force_set('text', hist.dialogues[-1].answer)
            tokenized_data = self.tokenize_from_history(retval, history)
            vectorized_data = util.map_data(tokenized_data, self.dict)
            features = util.generate_features(tokenized_data, vectorized_data, self.model.config['max_turns'])
            retval['text_vec'] = features
            retvals.append(retval)
        obs['text_vec'] = retvals[0]['text_vec']
        if len(retvals) > 1:
            obs['diverged_obs'] = retvals[1:]
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
        batch = Batch(batchsize=0, episode_done= obs_batch[0]['episode_done'])
        if len(obs_batch) == 0:
            return batch
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]
        if len(valid_obs) == 0:
            return batch
        valid_inds, exs = zip(*valid_obs)
        src_text_numb = [len(obs['text_vec']['src_text']) for obs in obs_batch]
        retval = self._collate_fn(obs_batch)

        inputs = trainer.unpack_batch(retval, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constants.PAD_ID)
        bg_mask = bg.eq(constants.PAD_ID) if bg is not None else None
        batch = Batch(
            batchsize=len(valid_obs),
            valid_indices=valid_inds,
            no_answer_reply=obs_batch[0].get('no_answer_reply', 'CANNOTANSWER'),
            src=src,
            src_char=inputs.get('src_char', None),
            src_text=inputs.get('src_text', None),
            bg=bg,
            bg_char=inputs.get('bg_char', None),
            bg_text=inputs.get('bg_text', None),
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            tgt_out_char=inputs.get('tgt_out_char', None),
            tgt_text=inputs.get('tgt_text', None),
            turn_ids=turn_ids,
            ctx=inputs.get('ctx', None),
            ctx_char=inputs.get('ctx_char', None),
            ctx_text=inputs.get('ctx_text', None),
            this_turn=retval['this_turn'],
            label_vec=tgt_out[:, -1:, :],
            text_vec=src,
            labels=inputs.get('tgt_text', None),
            text_lengths=src_text_numb,
            observations=obs_batch,
            episode_end= obs_batch[0]['episode_done']
        )
        return batch

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        return


    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        # if batch.label_lengths is None:
        #     return torch.randn(len(batch.text_lengths))
        model_output = self.model(self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        labels = batch.label_vec.view(batch.batchsize, -1)
        notnull = labels.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((labels == preds[-1, :]) * notnull).sum(dim=-1)

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        output = None
        if batch.batchsize <=0 and batch.episode_done:
            if self.rl_mode:
                self.reinforcement_backward_step()
            return None
        else:
            if self.rl_mode:
                return self.rl_train_step(batch)
            else:
                return super().train_step(batch)

    def rl_train_step(self, batch):
        self.model.train()
        maxlen = self.label_truncate or 256
        beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
        preds, scores = zip(*beam_preds_scores)
        self._add_generation_metrics(batch, preds)
        # bsz x beamsize
        beam_texts: List[List[Tuple[str, float]]] = []
        for beam in beams:
            beam_texts.append([])
            for tokens, score in beam.get_rescored_finished():
                try:
                    beam_texts[-1].append((self._v2t(tokens), score.item()))
                except KeyError:
                    logging.error("Decoding error: %s", tokens)
                    continue
        text = [self._v2t(p) for p in preds] if preds is not None else None
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds[0])
            self._compute_nltk_bleu(batch, text[0])
        retval = Output(text, log_probs=scores)
        return retval

    def reinforcement_backward_step(self, batch):
        return

    def tokenize_from_history(self, item, history=None):
        if not history:
            history = self.history
        strings_to_tokenize = [history.title, history.section_title, history.context, history.background]
        qas = []
        if len(history.history_dialogues) > 0:
            for turn in history.history_dialogues:
                strings_to_tokenize.append(turn.question)
                strings_to_tokenize.append(turn.answer)
                qas.append((turn.question, turn.answer))
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

    def _encoder_input(self, batch):
        src_mask = batch['src'].eq(constants.PAD_ID)
        bg_mask = batch['bg'].eq(constants.PAD_ID) if batch['bg'] is not None else None
        return [batch['src'], src_mask, batch['bg'], bg_mask, batch['turn_ids']]

    def _model_input(self, batch):
        src_mask = batch['src'].eq(constants.PAD_ID)
        bg_mask = batch['bg'].eq(constants.PAD_ID) if batch['bg'] is not None else None
        return {'src': batch['src'], 'src_mask': src_mask, 'turn_ids': batch['turn_ids'],
                'tgt_in': batch['tgt_in'], 'bg': batch['bg'], 'bg_mask': bg_mask, 'tgt_out': batch['tgt_out']}

    def predict(self, batch, beam_size=1, return_pair_level=False):
        inputs = trainer.unpack_batch(batch, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constants.PAD_ID)
        bg_mask = bg.eq(constants.PAD_ID) if bg is not None else None
        batch_size = src.size(0)
        preds, log_probs = self.model.sq2sq_model.predict(src, src_mask, turn_ids, beam_size=beam_size, bg=bg, bg_mask=bg_mask, return_pair_level=return_pair_level)
        pred_seqs = [[self.dict.ind2tok[id_] for id_ in ids] for ids in preds] # unmap to tokens
        pred_seqs = util.prune_decoded_seqs(pred_seqs)
        return pred_seqs, log_probs

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
        preds = self.model.sq2sq_model.sample(src, src_mask, turn_ids, top_p=top_p, bg=bg, bg_mask=bg_mask, return_pair_level=return_pair_level)
        preds, nll = preds
        pred_seqs = [[self.vocab['id2word'][id_] for id_ in ids] for ids in preds] # unmap to tokens
        pred_seqs = util.prune_decoded_seqs(pred_seqs)
        return pred_seqs, preds, nll
