
import sys
import os
import re
from collections import defaultdict
import spacy
import torch
import numpy as np
import parlai.utils.logging as logging
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.torch_agent import Optional, Batch
from parlai_internal.agents.torch_span_agent.torch_span_agent import TorchSpanAgent, DialogueHistory
from parlai_internal.agents.interviewee.models.seq2seq import TeacherModel
from parlai_internal.agents.interviewee import constants
from parlai_internal.agents.interviewee.models.trainer import unpack_batch
import pickle
from parlai.core.opt import Opt

class IntervieweeHistory(DialogueHistory):

    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        super().__init__(opt, **kwargs)
        self.title = None
        self.background = None
        self.section_title = None

    def reset(self):
        """
        Clear the history.
        """
        self.history_raw_strings = []
        self.history_dialogues = []
        self.history_strings = []
        self.history_vecs = []
        self.context = None
        self.title = None
        self.background = None
        self.section_title = None

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
        self.temp_history = temp_history


class IntervieweeDictionaryAgent(DictionaryAgent):

    def __init__(self, opt: Opt, shared=None):
        """
        Get the pretrained teacher model vocabs
        """
        if 'dict_file' not in opt or not opt['dict_file']:
            opt['dict_file'] = os.path.join(opt['datapath'], constants.VOCAB_FILE)
        self.vocab_size = 0
        self.char_vocab_size = 0
        self.vocab = None
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
        self.WHITESPACE = re.compile('\s+')
        super().__init__(opt)
        self.override_special_tokens(opt)

    def __len__(self):
        return self.vocab_size

    def _index_lookup(self, key):
        # return token from index, or unk_token
        if key > self.vocab_size:
            return self.unk_token
        else:
            return self.ind2tok[key]

    def load(self, filename):
        with open(filename, 'rb') as f:
            vocab = pickle.load(f)
        self.vocab_size = len(vocab['word2id'])
        self.char_vocab_size = len(vocab['char2id'])
        self.tok2ind = vocab['word2id']
        self.ind2tok = vocab['id2word']
        self.char2id = vocab['char2id']
        self.id2char = vocab['id2char']
        self.wordid2chars = vocab['wordid2chars']
        self.wordid2docfreq = vocab['wordid2docfreq']
        logging.info(f'num words = {len(self)}')

    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.
        """
        return True

    def _define_special_tokens(self, opt):
        self.unk_token = constants.UNK
        self.pad_token = constants.PAD
        self.null_token = constants.PAD
        self.end_token = constants.EOS
        self.start_token = constants.SOS
        self.sep_token = constants.SEP

    def override_special_tokens(self, opt):
        # Note that contants.SEP does not exist in the original teacher model.
        # It is a temporary measure ro fulfull the history delimiator requirement
        # , and will require to be replaced or removed before vectorization.
        # define special tokens
        self._define_special_tokens(opt)
        # now override
        self.unk_idx = constants.UNK_ID
        self.pad_idx = constants.PAD_ID
        self.null_idx = constants.PAD_ID
        self.end_idx = constants.EOS_ID
        self.start_idx = constants.SOS_ID
        self.sep_idx = len(self.tok2ind)
        self.tok2ind[self.sep_token] = self.sep_idx
        self.ind2tok.append(self.sep_token)

    def bulk_tokenize(self, text, return_offsets=False):
        ann = list(self.nlp.pipe(text))
        if return_offsets:
            return [[w.text for w in s if not self.WHITESPACE.match(w.text)] for s in ann], [
                [(w.idx, w.idx + len(w.text)) for w in s if not self.WHITESPACE.match(w.text)] for s in ann]
        else:
            return [[w.text for w in s if not self.WHITESPACE.match(w.text)] for s in ann]
        return ann

    def tokenize(self, text, building=False):
        return self.bulk_tokenize([text])[0]

    def vec2txt(self, vector, delimiter=' '):
        """
        Convert a vector of IDs to a string.

        Converts a vector (iterable of ints) into a string, with each token separated by
        the delimiter (default ``' '``).
        """
        tokens = [self[int(idx)] for idx in vector]
        if self.tokenizer in ['gpt2', 'bpe', 'slow_bytelevel_bpe']:
            # if we used a BPE tokenizer we need to rejoin the encodings
            text = self.bpe.decode(tokens, vector, delimiter)
        elif self.tokenizer == 'bytelevelbpe':
            # We add special tokens in the beginning of ParlAI dict but in the
            # end of Hugging Face dict, there is an offset of #(extra tokens) between them.
            extra_tokens = 4  # length of special tokens
            vector = [
                self.bpe.special_tok_map[idx]
                if idx in self.bpe.special_tok_map
                else idx - extra_tokens
                for idx in vector
            ]
            tokens = [self[int(idx)] for idx in vector]
            text = self.bpe.decode(tokens, vector, delimiter)
        else:
            text = delimiter.join(self[int(idx)] for idx in vector)

        return text

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
        return IntervieweeDictionaryAgent

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
            model = TeacherModel(config, use_cuda=self.use_cuda)
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
            maxlen=self.text_truncate,
            size=self.histsz,
            p1_token=self.P1_TOKEN,
            p2_token=self.P2_TOKEN,
            dict_agent=self.dict,
        )
        history.delimiter_tok = self.dict.sep_idx
        return history

    def _set_text_vec(self, obs, history, truncate, is_training=True):
        tokenized_data = self.tokenize_one(obs)
        vectorized_data = self.map_data(tokenized_data)
        features = self.generate_features(tokenized_data, vectorized_data)
        obs['text_vec'] = features
        labels_with_special_tokens = []
        for l in obs.get('labels', obs.get('eval_labels', [])):
            if l == "":
                labels_with_special_tokens.append(self.dict.cls_token)
            else:
                labels_with_special_tokens.append(l)
        if 'labels' in obs:
            obs.force_set('labels', labels_with_special_tokens)
        elif 'eval_labels' in obs:
            obs.force_set('eval_labels', labels_with_special_tokens)
        return obs

    def pad_char_start_end(self, char_ids):
        return [[constants.CHAR_START_ID] + w + [constants.CHAR_END_ID] for w in char_ids]

    def pad_lists(self, x, fill_val=0, dtype=np.int64):
        size = [len(x)]
        y = x
        while isinstance(y[0], list) or isinstance(y[0], np.ndarray):
            yy = []
            mx = 0
            for t in y:
                mx = max(len(t), mx)
                yy.extend(t)
            size.append(mx)
            y = yy

        res = np.full(size, fill_val, dtype=dtype)
        assert len(size) <= 4
        if len(size) == 1:
            res = np.array(x, dtype=dtype)
        elif len(size) == 2:
            for i in range(len(x)):
                res[i, :len(x[i])] = x[i]
        elif len(size) == 3:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    res[i, j, :len(x[i][j])] = x[i][j]
        elif len(size) == 4:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    for k in range(len(x[i][j])):
                        res[i, j, k, :len(x[i][j][k])] = x[i][j][k]

        return res

    def generate_features(self, tok_data, idx_data):
        max_turns = self.model.args['max_turns']
        tpara, ipara = tok_data, idx_data
        tsrc = [constants.TITLEST] + tpara['title'] + [constants.TITLEEN, constants.BGST] + tpara['background'][:constants.MAX_BACKGROUND] + [constants.BGEN]
        isrc = [constants.TITLEST_ID] + ipara['title'] + [constants.TITLEEN_ID, constants.BGST_ID] + ipara['background'][:constants.MAX_BACKGROUND] + [constants.BGEN_ID]
        csrc = self.pad_char_start_end([[constants.TITLEST_ID]] + ipara['title_char'] + [[constants.TITLEEN_ID],
               [constants.BGST_ID]] + ipara['background_char'][:constants.MAX_BACKGROUND] + [[constants.BGEN_ID]])

        tbg = [] + tsrc
        ibg = [] + isrc
        cbg = [] + csrc

        turn_ids = [-1] * len(tsrc)

        # clear src variables for dialogue history
        tsrc = []
        isrc = []
        csrc = []
        turn_ids = []

        tsrc += [constants.SECST] + tpara['section_title'] + [constants.SECEN]
        isrc += [constants.SECST_ID] + ipara['section_title'] + [constants.SECEN_ID]
        csrc += self.pad_char_start_end([[constants.SECST_ID]] + ipara['section_title_char'] + [[constants.SECEN_ID]])
        src_idf = [0] + ipara['section_title_idf'] + [0]
        turn_ids += [0] * (len(ipara['section_title']) + 2)

        ans_mask = np.zeros((len(ipara['context'][:constants.MAX_CONTEXT]), constants.MAX_TURNS), dtype=np.int64)

        this_para = defaultdict(list)
        for turnid, (tqa, iqa) in enumerate(zip(tpara['qas'], ipara['qas'])):
            ttgt_in = [constants.SOS] + tqa['question']
            ttgt_out = tqa['question'] + [constants.EOS]
            itgt_in = np.array([constants.SOS_ID] + iqa['question'], dtype=np.int64)
            itgt_out = np.array(iqa['question'] + [constants.EOS_ID], dtype=np.int64)
            ctgt_in = self.pad_char_start_end([[constants.SOS_ID]] + iqa['question_char'])
            ctgt_out = self.pad_char_start_end(iqa['question_char'] + [[constants.EOS_ID]])

            this_para['src_text'].append([] + tsrc)
            this_para['src_idx'].append(self.pad_lists(isrc, dtype=np.int64))
            this_para['src_char'].append(self.pad_lists(csrc, dtype=np.int64))
            this_para['src_idf'].append(self.pad_lists(src_idf, dtype=np.float32))
            this_para['tgt_text'].append([x.lower() for x in tqa['question']])
            this_para['tgt_in_idx'].append(self.pad_lists(itgt_in, dtype=np.int64))
            this_para['tgt_in_char'].append(self.pad_lists(ctgt_in, dtype=np.int64))
            this_para['tgt_out_idx'].append(self.pad_lists(itgt_out, dtype=np.int64))
            this_para['tgt_out_char'].append(self.pad_lists(ctgt_out, dtype=np.int64))
            this_para['turn_ids'].append(self.pad_lists(turn_ids, fill_val=-1, dtype=np.int64))
            this_para['start'].append(iqa['start'])
            this_para['end'].append(iqa['end'])
            this_para['yesno'].append(iqa['yesno'])
            this_para['followup'].append(iqa['followup'])
            this_para['ans_mask'].append(np.array(ans_mask))
            this_para['this_turnid'].append(turnid)

            ans_mask[iqa['start']:iqa['end'], turnid] += 1 # in span
            ans_mask[iqa['start']:iqa['start']+1, turnid] += 1 # beginning of span
            ans_mask[iqa['end']-1:iqa['end'], turnid] += 2 # end of span

            # append Q and A with separators
            tsrc = [constants.QUESST] + [x.lower() for x in tqa['question']] + [constants.QUESEN, constants.ANSST] + tqa['answer'] + [constants.ANSEN]
            isrc = [constants.QUESST_ID] + iqa['question'] + [constants.QUESEN_ID, constants.ANSST_ID] + iqa['answer'] + [constants.ANSEN_ID]
            csrc = self.pad_char_start_end([[constants.QUESST_ID]] + iqa['question_char'] + [[constants.QUESEN_ID], [constants.ANSST_ID]] + iqa['answer_char'] + [[constants.ANSEN_ID]])
            turn_ids = [turnid + 1] * (len(tqa['question']) + len(tqa['answer']) + 4)

            ques_count = sum(1 for x in isrc if x == constants.QUESST_ID) - 1
            if max_turns >= 0 and ques_count > max_turns:
                idx = len(isrc) - 1
                count = 0
                while idx > 0:
                    if isrc[idx] == constants.QUESST_ID:
                        count += 1
                        if count > self.max_turns:
                            break
                    idx -= 1
                tsrc = tsrc[idx:]
                isrc = isrc[idx:]
                csrc = csrc[idx:]
                turn_ids = turn_ids[idx:]

        datum = dict()
        datum['ctx_text'] = tpara['context'][:constants.MAX_CONTEXT]
        datum['ctx_idx'] = ipara['context'][:constants.MAX_CONTEXT]
        datum['ctx_char'] = self.pad_lists(self.pad_char_start_end(ipara['context_char'][:constants.MAX_CONTEXT]), dtype=np.int64)

        datum['bg_text'] = tbg
        datum['bg_idx'] = ibg
        datum['bg_char'] = cbg

        for k in ['src_text', 'tgt_text', 'start', 'end', 'yesno', 'followup', 'this_turnid']:
            datum[k] = this_para[k]

        for k in ['src_idx', 'src_char', 'tgt_in_idx', 'tgt_in_char', 'tgt_out_idx', 'tgt_out_char', 'ans_mask']:
            datum[k] = self.pad_lists(this_para[k], dtype=np.int64)

        datum['turn_ids'] = self.pad_lists(this_para['turn_ids'], fill_val=-1, dtype=np.int64)
        return datum

    def tokenize_one(self, item):
        strings_to_tokenize = [item['title'], item['section_title'], item['context'], item['background']]
        qas = []
        if len(self.history.history_dialogues) > 1:
            for qa in self.history.history_dialogues[:-1]:
                strings_to_tokenize.append(qa[0])
                strings_to_tokenize.append(qa[1])
                qas.append((qa[0], qa[1]))
        strings_to_tokenize.append(item['text'])
        strings_to_tokenize.append(item['single_label_text'])
        qas.append((item['text'], item['single_label_text']))
        tokenized, offsets = self.dict.bulk_tokenize(strings_to_tokenize, return_offsets=True)
        retval = {'title': tokenized[0], 'section_title': tokenized[1], 'context': tokenized[2], 'background': tokenized[3], 'qas':[]}
        tokenized = tokenized[4:]
        ctx_offsets = [(st-offsets[2][0][0], en-offsets[2][0][0]) for st, en in offsets[2]]
        parsed_idx = 0
        for idx, qa in enumerate(qas):
            yesno = None
            followup = None
            ans_st = -1
            ans_en = -1
            if idx == len(qas) -1:
                ans = tokenized[1]
                if item['yesno'] == '__YES__':
                    ans = ['Yes', ','] + tokenized[1]
                    yesno = constants.YESNO_TO_ID['y']
                elif item['yesno'] == '__NO__':
                    ans = ['No', ','] + tokenized[1]
                    yesno = constants.YESNO_TO_ID['n']
                else:
                    yesno = constants.YESNO_TO_ID['x']
                if item['followup'] == '__SHOULDNOT__':
                    followup = constants.FOLLOWUP_TO_ID['n']
                elif item['followup'] == '__SHOULD__':
                    followup = constants.FOLLOWUP_TO_ID['f']
                else:
                    followup = constants.FOLLOWUP_TO_ID['m']

                char_st = item['character_start_end'][0]
                char_en = item['character_start_end'][-1]
                for idx, (st, en) in enumerate(ctx_offsets):
                    if en > char_st and ans_st < 0:
                        ans_st = idx
                    if st >= char_en and ans_en < 0:
                        ans_en = idx
                if ans_en < 0:
                    ans_en = len(ctx_offsets)
                assert ''.join(tokenized[1]) in ''.join(retval['context'][ans_st:ans_en]), '{} {}'.format(str(retval['context'][ans_st:ans_en]), str(tokenized[1]))
            else:
                ans = tokenized[1]
            retval['qas'].append({'question': tokenized[0], 'answer': ans,
            'start': ans_st, 'end': ans_en, 'yesno': yesno, 'followup': followup})
            tokenized = tokenized[2:]
            offsets = offsets[2:]
            parsed_idx += 2
        return retval

    def map_data(self, tokenized_data):
        def map_field(field):
            return [self.dict.tok2ind.get(x.lower(), self.dict.unk_idx) for x in field]

        def map_char(field, do_lower=False):
            if do_lower:
                return [[self.dict.char2id.get(c, self.dict.unk_idx) for c in w.lower()] if w not in constants.VOCAB_PREFIX else [self.dict.char2id[w]] for w in field]
            else:
                return [[self.dict.char2id.get(c, self.dict.unk_idx) for c in w] if w not in constants.VOCAB_PREFIX else [self.dict.char2id[w]] for w in field]

        def map_idf(field):
            return [1 / self.dict.wordid2docfreq[self.dict.tok2ind.get(x.lower(), self.dict.unk_idx)] for x in field]

        def copy_mask(src, dst):
            return [[1 if w1.lower() == w2.lower() else 0 for w2 in dst] for w1 in src]

        def map_one(item):
            retval = {'title': map_field(item['title']),
                    'title_char': map_char(item['title']),
                    'section_title': map_field(item['section_title']),
                    'section_title_idf': map_idf(item['section_title']),
                    'section_title_char': map_char(item['section_title']),
                    'background': map_field(item['background']),
                    'background_char': map_char(item['background']),
                    'context': map_field(item['context']),
                    'context_char': map_char(item['context']),
                    'qas': [{'question': map_field(x['question']), 'answer': map_field(x['answer']),
                        'question_char': map_char(x['question'], do_lower=True), 'answer_char': map_char(x['answer']),
                        'question_idf': map_idf(x['question']), 'answer_idf': map_idf(x['answer']),
                        'start': x['start'],
                        'end': x['end'],
                        'followup': x['followup'],
                        'yesno': x['yesno']} for x in item['qas']]}
            return retval

        return map_one(tokenized_data)
    
    def batchify(self, obs_batch, sort=False):
        is_training = self.is_training
        batch = Batch(batchsize=0)
        if len(obs_batch) == 0:
            return batch
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]
        if len(valid_obs) == 0:
            return batch
        retval = self._collate_fn(obs_batch)
        batch = Batch(
            batchsize=len(valid_obs),
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
        
    def _collate_fn(self, obs_batch):
        batch_data = [x['text_vec'] for x in obs_batch]
        src = torch.from_numpy(self.pad_lists([x['src_idx'] for x in batch_data]))
        src_char = torch.from_numpy(self.pad_lists([x['src_char'] for x in batch_data]))
        ctx = torch.from_numpy(self.pad_lists([x['ctx_idx'] for x in batch_data]))
        ctx_char = torch.from_numpy(self.pad_lists([x['ctx_char'] for x in batch_data]))
        tgt_in = torch.from_numpy(self.pad_lists([x['tgt_in_idx'] for x in batch_data]))
        tgt_in_char = torch.from_numpy(self.pad_lists([x['tgt_in_char'] for x in batch_data]))
        tgt_out = torch.from_numpy(self.pad_lists([x['tgt_out_idx'] for x in batch_data]))
        tgt_out_char = torch.from_numpy(self.pad_lists([x['tgt_out_char'] for x in batch_data]))
        # neg_out = torch.from_numpy(self.pad_lists([x['neg_out_idx'] for x in batch_data]))
        # neg_out_char = torch.from_numpy(self.pad_lists([x['neg_out_char'] for x in batch_data]))

        this_turn = torch.from_numpy(self.pad_lists([x['this_turnid'] for x in batch_data]))
        ans_mask = torch.from_numpy(self.pad_lists([x['ans_mask'] for x in batch_data]))

        turn_ids = torch.from_numpy(self.pad_lists([x['turn_ids'] for x in batch_data], fill_val=-1))
        start = torch.from_numpy(self.pad_lists([x['start'] for x in batch_data], fill_val=-1))
        end = torch.from_numpy(self.pad_lists([x['end'] for x in batch_data], fill_val=-1))
        yesno = torch.from_numpy(self.pad_lists([x['yesno'] for x in batch_data], fill_val=-1))
        followup = torch.from_numpy(self.pad_lists([x['followup'] for x in batch_data], fill_val=-1))

        retval = {'src': src,
                  'src_char': src_char,
                  'src_text': [x['src_text'] for x in batch_data],
                  'tgt_in': tgt_in,
                  'tgt_out': tgt_out,
                  'tgt_out_char': tgt_out_char,
                  'tgt_text': [x['tgt_text'] for x in batch_data],
                  # 'neg_out': neg_out,
                  # 'neg_out_char': neg_out_char,
                  # 'neg_text': [x['neg_text'] for x in batch_data],
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
            bg = torch.from_numpy(self.pad_lists([x['bg_idx'] for x in batch_data]))
            bg_char = torch.from_numpy(self.pad_lists([x['bg_char'] for x in batch_data]))
            retval['bg'] = bg
            retval['bg_char'] = bg_char
            retval['bg_text'] = [x['bg_text'] for x in batch_data]
        return retval

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
        teacher_loss, teacher_acc = self.model(**self._model_input(batch))[:2]