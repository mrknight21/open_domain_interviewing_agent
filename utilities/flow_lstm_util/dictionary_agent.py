import re
import os
import pickle
import spacy
import parlai.utils.logging as logging
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai_internal.utilities.flow_lstm_util import constants
import numpy as np


class InterviewDictionaryAgent(DictionaryAgent):

    def __init__(self, opt: Opt, shared=None):
        """
        Get the pretrained teacher model vocabs
        """
        # if 'dict_file' not in opt or not opt['dict_file']:
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
        self.vocab = vocab
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
        self._unk_token_idx = self.unk_token

    def override_special_tokens(self, opt):
        # Note that contants.SEP does not exist in the original teacher model.
        # It is a temporary measure ro fulfull the history delimiator requirement
        # , and will require to be replaced or removed before vectorization.
        # define special tokens
        self._define_special_tokens(opt)
        # now override
        self.unk_idx = constants.UNK_ID
        self._unk_token_idx = self.unk_idx
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
