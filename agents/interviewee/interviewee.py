
import sys
import os
import re
import spacy
import torch
import parlai.utils.logging as logging
from parlai.core.dict import DictionaryAgent
from parlai_internal.agents.torch_span_agent.torch_span_agent import TorchSpanAgent
from parlai_internal.agents.interviewee.models.seq2seq import TeacherModel
from parlai_internal.agents.interviewee import constants
import pickle
import numpy as np
from parlai.core.opt import Opt



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
        return model

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

    # def batchify(self, obs_batch, sort=False):
    #     pass
    #
    # def _model_input(self, batch):
    #     pass

