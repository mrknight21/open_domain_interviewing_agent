import os
from typing import List, Tuple
import torch
import numpy as np
import copy
from parlai.core.message import Message
from parlai.core.torch_agent import Optional, Batch, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent, PPLMetric, TorchGeneratorModel
from parlai_internal.utilities.flow_lstm_util.dictionary_agent import InterviewDictionaryAgent
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
from parlai.utils.misc import warn_once
from parlai_internal.utilities.flow_lstm_util.models.seq2seq import Seq2SeqModel
from parlai_internal.utilities.flow_lstm_util.models import trainer
from parlai_internal.utilities.flow_lstm_util import constants, util
from parlai_internal.utilities.dialogue_history import DialogueHistory, DialogueLineages
from parlai_internal.reward_funcs import forward_average_discount, normalize_rewards, normalizeZ
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt



class InterviewerHistory(DialogueHistory):


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
        if "text" in obs and obs["text"]:
            super().update_history(obs, temp_history)
        else:
            if not self.context_token_weights and obs.get("context_token_weights", None):
                self.context_token_weights = obs["context_token_weights"]
            if not self.context and obs.get('context', None):
                self.context = obs['context']
            if not self.background and obs.get('background', None):
                self.background = obs['background']
            if not self.title and obs.get('title', None):
                self.title = obs['title']
            if not self.section_title and obs.get('section_title', None):
                self.section_title = obs['section_title']
        self.temp_history = temp_history

class Sq2SqQuestionGenerationModel(TorchGeneratorModel):

    def __init__(self, opt, dict):
        # self.add_start_token = opt["add_start_token"]
        super().__init__(*self._get_special_tokens(opt, dict))

        # init the model
        self.sq2sq_model = self.load_question_generation_model(opt, dict)
        self.config = self.sq2sq_model.args

    def encoder(self, src, src_mask, bg, bg_mask, turn_ids=None):
        # prepare for encoder/
        bs, ts = src.size()[:2]
        embed_d = self.sq2sq_model.dec_hidden_dim
        #get the last turn encoding for each batched conversation lineages
        if self.sq2sq_model.use_cuda:
            indices = torch.arange(bs).cuda()*ts
        else:
            indices = torch.arange(bs)*ts
        assert len(indices) == bs
        h_in, src_mask, (hn, cn), h_bg = self.sq2sq_model.encode_sources(src, src_mask, bg, bg_mask, turn_ids=turn_ids)
        h_in = torch.index_select(h_in, 0, indices).view(bs, -1, embed_d)
        src_mask = torch.index_select(src_mask, 0, indices)
        hn = torch.index_select(hn, 1, indices).view(bs, -1, embed_d)
        cn = torch.index_select(cn, 1, indices).view(bs, -1, embed_d)
        h_bg = torch.index_select(h_bg, 0, indices)
        turn_ids = turn_ids[:, -1, :]

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
        if opt.get('init_finetune', False):
            filename = os.path.join(opt['datapath'], constants.FINE_TUNE_FILE)
        else:
            filename = os.path.join(opt['datapath'], constants.BASE_MODEL_FILE)
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
        parser.add_argument(
            '--init-finetune',
            type='bool',
            default=False,
            help='init pretrained teacher model with finetuned mode',
        )
        parser.add_argument(
            '--use-master-baseline',
            type='bool',
            default=False,
            help='use reward from the master lineage as the baseline',
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
        parser.add_argument(
            '--reinforcement-gemma',
            type=float,
            default=0.9,
            help='the discount factor for reinforcement learning',
        )
        parser.add_argument(
            '--reinforcement-lambda',
            type=float,
            default=0.9,
            help='the percentage of the reinforcement account for total loss',
        )
        parser.add_argument(
            '--question-truncate',
            type=int,
            default=30,
            help='the limit for the number of tokens in a question',
        )

    def __init__(self, opt: Opt, shared=None):
        self.rl_mode = opt['reinforcement_learning']
        self.exploration_steps = opt['exploration_steps']
        self.reinforcement_gemma = opt['reinforcement_gemma']
        self.reinforcement_lambda = opt['reinforcement_lambda']
        self.question_truncate = opt['question_truncate']
        self.init_finetune = opt['init_finetune']
        self.use_master_baseline = opt['use_master_baseline']
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
        self.diverged_dialogues = DialogueLineages()
        return history

    def diverged_dialogues_update(self, observation):
        """
        This method handle the output from teacher model with update and harvest the diverged lineage history.
        The first answer is the model output for the most recent teacher force history
        The rest items follwo the index of the lineages
        :param model_outputs: observation with 'model_answer'
        :return: Update the diverged_dialogues object
        """
        if 'model_answers' not in observation:
            return None
        model_output = observation['model_answers']
        cnt = len(model_output)
        for i, retval in enumerate(model_output):
            text = retval['text']
            cache = self.diverged_dialogues.get_cache(retval)
            reward = retval['reward_items']
            if i == 0:
                self.diverged_dialogues.add_lineage(text, self.history, message=retval, reward=reward)
            else:
                if not self.is_training and i + 1 == cnt:
                    self.diverged_dialogues.lineages[-1]._update_dialogues(text, cache=cache, reward=reward)
                    continue
                self.diverged_dialogues.lineages[i]._update_dialogues(text, cache=cache, reward=reward)
        for lineage in self.diverged_dialogues.lineages:
            if not lineage.freeze:
                # freeze lineage that reach the exploration limits
                if len(list(filter(lambda x: x.complete and x.generated, lineage.dialogues))) >= self.exploration_steps:
                    # allow the longest gen lineage to grow for later evaluation
                    if not self.is_training and lineage.gen_start_index == 0:
                        continue
                    else:
                        lineage.freeze = True


    def self_observe_diverged_dialogue_update(self, self_message):
        """
        This method the diverged dialogue lineages with its QA model outputs
        :param model_outputs: message object  with 'text', 'log_probs', and 'diverged_outputs'
        :return: Update the diverged_dialogues object
        """
        self.diverged_dialogues.add_lineage(self_message["text"], self.history,
                                            log_prob=self_message.get("log_probs", None),
                                            ques_len=self_message.get("ques_len", None))
        model_outputs = self_message['diverged_outputs']
        cnt = len(model_outputs)
        for i, (text, logprob, ques_len) in enumerate(model_outputs):
            if not self.is_training and i+1==cnt:
                self.diverged_dialogues.lineages[-1]._update_dialogues(text, log_prob=logprob,
                                                                                  ques_len=ques_len)
                continue
            lineage_index = i + 1
            self.diverged_dialogues.lineages[lineage_index]._update_dialogues(text, log_prob=logprob, ques_len=ques_len)


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
        if self.rl_mode and len(self.diverged_dialogues.lineages) > 0:
            self.diverged_dialogues_update(observation)
        super().observe(observation)
        if observation.get('model_answers', None):
            master_reward_items = observation['model_answers'][0].get('reward_items', None)
            if master_reward_items:
                self.history.dialogues[-1].update(reward=master_reward_items)
        if self.rl_mode and observation['episode_done']:
            self.history.rewards = observation.get('rewards', None)


    def self_observe(self, self_message: Message) -> None:
        """
        Observe one's own utterance.

        This is used so that the agent can incorporate its own response into
        the dialogue history after a batch_act. Failure to implement this will
        result in an agent that cannot hear itself speak.

        :param self_message:
            The message corresponding to the output from batch_act.
        """
        # It is a very implicit and tedious way of identifying on going reinforcement episode, need better expression
        if self.rl_mode and self.exploration_steps > 0 and not self_message.get('episode_end', True):
                self.self_observe_diverged_dialogue_update(self_message)
                self_message['diverged_dialogues'] = self.diverged_dialogues
        super().self_observe(self_message)
        if self.rl_mode and self.exploration_steps > 0:
            # It is a very implicit and tedious way of identifying on going reinforcement episode, need better expression
            if not self_message.get('episode_end', True):
                self_message['history'] = self.history
            else:
                self.diverged_dialogues.reset()

    def _set_text_vec(self, obs, history, truncate, is_training=True):
        history_dialogues = [self.history.dialogues]
        # original_answer = obs['text']
        # original_question = obs['single_label_text']
        retvals = []
        if len(self.diverged_dialogues.lineages) > 0:
            history_dialogues += [lineage.dialogues for lineage in self.diverged_dialogues.lineages if not lineage.freeze]
        for dialogues in history_dialogues:
            retval = copy.copy(obs)
            if dialogues:
                retval.force_set('text', dialogues[-1].answer)
            tokenized_data = self.tokenize_from_history(retval, dialogues)
            vectorized_data = util.map_data(tokenized_data, self.dict)
            features = util.generate_features(tokenized_data, vectorized_data, self.model.config['max_turns'])
            retval['text_vec'] = features
            retvals.append(retval)
        #The master history lineage
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

    def unpacking_obs_batcch_with_divergence(self, obs_batch):
        new_obs_batch = []
        index_batch_map = {}
        index = 0
        master_batch_size = 0
        for batch_index, obs in enumerate(obs_batch):
            master_batch_size += 1
            master_id = str(batch_index) + "-0"
            index_batch_map[index] = master_id
            new_obs_batch.append(obs)
            if 'diverged_obs' in obs and len(obs['diverged_obs']) > 0:
                for div_index, div_obs in enumerate(obs['diverged_obs']):
                    index += 1
                    div_id = str(batch_index) + "-" + str(div_index+1)
                    index_batch_map[index] = div_id
                    new_obs_batch.append(div_obs)
        return new_obs_batch, index_batch_map

    def get_preprocessed_batches(self, obs_batch, valid_inds):
        src_text_numb = [len(obs['text_vec']['src_text']) for obs in obs_batch]
        retval = self._collate_fn(obs_batch)
        inputs = trainer.unpack_batch(retval, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        batch = Batch(
            batchsize=len(obs_batch),
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
            episode_end= obs_batch[0]['episode_done'],
        )
        return batch


    def batchify(self, org_batch, sort=False):
        batch = Batch(batchsize=0, episode_done=org_batch[0]['episode_done'], valid_indices=list(range(len(org_batch))))
        if batch.episode_done or len(org_batch) == 0:
            return batch
        valid_obs = [(i, ex) for i, ex in enumerate(org_batch) if self.is_valid(ex)]
        if len(valid_obs) == 0:
            return batch
        valid_inds, exs = zip(*valid_obs)
        master_batch = self.get_preprocessed_batches(org_batch, valid_inds)
        if self.rl_mode and self.exploration_steps > 0 and len(self.diverged_dialogues.lineages) > 0:
            div_batch, index_batch_id_map = self.unpacking_obs_batcch_with_divergence(org_batch)
            div_valid_inds = [i for i, b in enumerate(div_batch) if self.is_valid(b)]
            master_batch['diverged_batch'] = self.get_preprocessed_batches(div_batch, div_valid_inds)
        return master_batch

    #This function is for stress test before an intensive training is carried out.
    #Need time to implement this
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

        self.record_local_metric('tf_loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('tf_ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'tf_token_acc', AverageMetric.many(correct, target_tokens)
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
        if batch.batchsize <= 0 and batch.episode_done:
            if self.rl_mode:
                self.reinforcement_backward_step()
                return
            return None
        else:
            if self.rl_mode:
                div_batch = batch.get('diverged_batch', None)
                if not div_batch:
                    self.model.train()
                    self.zero_grad()
                    div_batch = batch
                self.history.dialogues_nll_loss.append(self.compute_loss(batch))
                return self.rl_train_step(div_batch)
            else:
                return super().train_step(batch)

    def rl_train_step(self, batch):
        maxlen = self.question_truncate or 30
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
        retval = Output(text[:1], log_probs=scores[:1], episode_end=[batch.episode_end], ques_len=[len(preds[0])-1],  diverged_outputs=[[(t, scores[i], len(preds[i])-1) for i, t in enumerate(text[1:])]])
        return retval

    def reinforcement_backward_step(self):
        total_reward = []
        local_rewards_filters = {}
        local_rewards_tracker = {}
        gen_reward_index = []
        reward_tracker = {}
        if not self.history.rewards:
            return total_reward
        log_probs = self.diverged_dialogues.get_log_probs()
        for content in log_probs:
            if len(content['ques_len']) == 0:
                gen_reward_index.append([])
            else:
                gen_reward_index.append(list(range(content['dialogue_length'])[-len(content['ques_len']):]))

        global_rewards = {name: content for name, content in self.history.rewards.items() if content['global']}
        local_rewards = {name: content for name, content in self.history.rewards.items() if not content['global']}

        for r_name, r in local_rewards.items():
            master_raw_r = r['master']
            diverged_raw_r = r['diverged_rewards']
            gen_r = [[r for j, r in enumerate(rs) if j in gen_reward_index[i]] for i, rs in enumerate(diverged_raw_r)]
            mean_gen_r = np.mean([r for conv in gen_r for r in conv])
            if not local_rewards_filters:
                local_rewards_filters = {"master": master_raw_r, "diverged": diverged_raw_r}
            else:
                for i, r in enumerate(master_raw_r):
                    local_rewards_filters["master"][i] = r * local_rewards_filters["master"][i]
                for i, conv in enumerate(diverged_raw_r):
                    for j, r in enumerate(conv):
                        local_rewards_filters["diverged"][i][j] = local_rewards_filters["diverged"][i][j] * r
            local_rewards_tracker[r_name] = mean_gen_r
            self.record_local_metric(r_name + '_mean', AverageMetric.many([float(mean_gen_r)], [1]))
        weight = 1 / len(global_rewards)
        for r_name, r in global_rewards.items():
            master_raw_r = r['master']
            diverged_raw_r = r['diverged_rewards']
            is_global = r['global']
            required_normalise = r['required_normalise']
            gen_r = [[r for j, r in enumerate(rs) if j in gen_reward_index[i]] for i, rs in enumerate(diverged_raw_r)]
            if is_global:
                for i, conv in enumerate(gen_r):
                    if len(conv) > 0:
                        gen_r[i] = forward_average_discount(np.array([conv]))[0].tolist()
                    else:
                        gen_r[i] = np.array([])
            if local_rewards_filters['diverged']:
                for i, conv in enumerate(gen_r):
                    if len(conv) > 0:
                        gen_r[i] = [d*local_rewards_filters['diverged'][i][j] for j, d in enumerate(conv)]
            if required_normalise:
                flat_gen_r = [r for conv in gen_r for r in conv]
                mean_gen_r = np.mean(flat_gen_r)
                std_gen_r = np.std(flat_gen_r)
            rewards = []
            reward_tracker[r_name] = []
            for step in range(len(master_raw_r)):
                diverged_lineages = [(l.dialogues, gen_r[i], log_probs[i])
                                    for i, l in enumerate(self.diverged_dialogues.lineages)
                                    if log_probs[i]['log_probs'] and l.gen_start_index == step]
                if not diverged_lineages:
                    continue
                turns_count = len(diverged_lineages[0][0])
                dl_logprobs = [dl[2]['log_probs'] for dl in diverged_lineages]
                dl_ques_len = [torch.tensor(dl[2]['ques_len']) for dl in diverged_lineages]
                num_generated_turns = len(dl_logprobs[-1])
                dl_rewards = [dl[1] for dl in diverged_lineages]
                master_rs = master_raw_r[turns_count-num_generated_turns:turns_count]
                master_filter = local_rewards_filters['master'][turns_count-num_generated_turns:turns_count]
                _rewards = np.array([master_rs] + dl_rewards)
                if required_normalise:
                    _rewards = (_rewards - mean_gen_r) / std_gen_r
                master_rs = _rewards[0]
                diverged_rs = _rewards[1:]
                #use master reward as baseline
                if self.use_master_baseline:
                    _reward = torch.tensor(diverged_rs - master_rs)
                else:
                    _reward = torch.tensor(diverged_rs)
                # relative_reward = torch.tensor(dl_rewards)
                if self.use_cuda:
                    _reward = _reward.cuda()
                    dl_ques_len = [dql.cuda() for dql in dl_ques_len]
                r = (_reward * torch.stack([torch.stack(dl_logprobs[i])/dl_ques_len[i] for i in range(len(dl_logprobs))])).mean()
                rewards.append(r)
                reward_tracker[r_name].append(r.data)
            reward = torch.stack(rewards).mean()
            total_reward.append(reward*weight)
        reinforcement_loss = torch.stack(total_reward).sum() * -10
        total_loss = self.reinforcement_lambda * reinforcement_loss + (1-self.reinforcement_lambda)*torch.stack(self.history.dialogues_nll_loss).mean()
        self.backward(total_loss)
        self.update_params()
        self.record_local_metric('total_loss', AverageMetric.many([float(total_loss.detach().cpu())], [1]))
        self.record_local_metric('reward_loss', AverageMetric.many([float(reinforcement_loss.detach().cpu())], [1]))

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        retval = None
        if batch.batchsize <= 0 and batch.episode_done:
            if self.rl_mode:
                self.rl_eval_final_step(batch)
            return retval
        else:
            if self.rl_mode:
                retval = self.rl_eval_step(batch)
            else:
                retval = super.eval_step(batch)
        return retval


    def rl_eval_step(self, batch):
        div_batch = batch.get('diverged_batch', None)
        if not div_batch:
            div_batch = batch
        token_losses = None
        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )
        preds = None
        maxlen = self.question_truncate or 30
        beam_preds_scores, beams = self._generate(div_batch, self.beam_size, maxlen)
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
        retval = Output(text[:1], log_probs=scores[:1], episode_end=[batch.episode_end], ques_len=[len(preds[0])-1],  diverged_outputs=[[(t, scores[i], len(preds[i])-1) for i, t in enumerate(text[1:])]])
        return retval

    def rl_eval_final_step(self, batch):
        gen_reward_index = []
        if not self.history.rewards:
            return
        log_probs = self.diverged_dialogues.get_log_probs()
        for content in log_probs:
            if len(content['ques_len']) == 0:
                gen_reward_index.append([])
            else:
                gen_reward_index.append(list(range(content['dialogue_length'])[-len(content['ques_len']):]))
        reward_tracker = {}
        reward_step_tracker = {}
        reward_step_d_tracker = {}
        reward_full_lineage_tracker = {}
        for r_name, r in self.history.rewards.items():
            master_raw_r = r['master']
            diverged_raw_r = r['diverged_rewards']
            is_global = r['global']
            required_normalise = r['required_normalise']

            gen_r = [[r for j, r in enumerate(rs) if j in gen_reward_index[i]] for i, rs in enumerate(diverged_raw_r)]
            if required_normalise:
                gen_r = normalize_rewards(gen_r)
            mean_gen_r = np.mean([r for conv in gen_r for r in conv])
            reward_tracker[r_name] = mean_gen_r
            reward_step_tracker[r_name] = []
            reward_step_d_tracker[r_name] = []
            reward_full_lineage_tracker[r_name] = {'reward': [], 'diff': []}
            for step in range(len(master_raw_r)):
                diverged_lineages = [(l.dialogues, diverged_raw_r[i], log_probs[i])
                                    for i, l in enumerate(self.diverged_dialogues.lineages)
                                    if log_probs[i]['log_probs'] and l.gen_start_index == step]
                if not diverged_lineages:
                    continue
                turns_count = len(diverged_lineages[0][0])
                dl_logprobs = [dl[2]['log_probs'] for dl in diverged_lineages]
                num_generated_turns = len(dl_logprobs[-1])
                dl_rewards = [dl[1][-num_generated_turns:] for dl in diverged_lineages]
                master_turns = master_raw_r[turns_count-num_generated_turns:turns_count]
                _rewards = np.array([master_turns] + dl_rewards)
                # if is_global:
                #     _rewards = forward_average_discount(_rewards, self.reinforcement_gemma)
                # if required_normalise:
                #     _rewards = normalizeZ(_rewards)
                #use master reward as baseline
                d_reward = _rewards[1:] - _rewards[0]
                _rewards = _rewards[1:]
                if num_generated_turns == len(master_raw_r):
                    reward_full_lineage_tracker[r_name]['reward'].append(_rewards.mean(0))
                    reward_full_lineage_tracker[r_name]['diff'].append(d_reward.mean(0))
                else:
                    reward_step_tracker[r_name].append(_rewards.mean(0))
                    reward_step_d_tracker[r_name].append(d_reward.mean(0))

            avg_diff = np.mean([r for conv in reward_step_d_tracker[r_name] for r in conv])
            self.record_local_metric(r_name+'_mean', AverageMetric.many([float(mean_gen_r)], [1]))
            self.record_local_metric(r_name+'_avg_diff', AverageMetric.many([float(avg_diff)], [1]))

    def tokenize_from_history(self, item, dialogues=None):
        history = self.history
        if not dialogues:
            dialogues = history.dialogues
        strings_to_tokenize = [history.title, history.section_title, history.context, history.background]
        qas = []
        if len(dialogues) > 0:
            for turn in dialogues:
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
