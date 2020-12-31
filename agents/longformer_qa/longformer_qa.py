import os


from parlai_internal.agents.torch_span_agent.torch_span_agent import TorchSpanAgent, TorchExtractiveModel
from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent
from parlai.utils.misc import warn_once
from transformers import AutoTokenizer

class HFLongformerQAModel(TorchExtractiveModel):
    """
        Bert Encoder.

        This encoder is initialized with the pretrained model from Hugging Face.
        """

    def __init__(self, opt):

        # init the model
        super().__init__(opt)
        self.model_type = 'longformer'

    def get_config_key(self, opt):
        # load model
        # check if datapath has the files that hugging face code looks for
        if all(
            os.path.isfile(
                os.path.join(opt["datapath"], "models", "longformer_hf", file_name)
            )
            for file_name in ["pytorch_model.bin", "config.json"]
        ):
            fle_key = os.path.join(opt["datapath"], "models", "longformer_hf")
        else:
            fle_key = opt["longformer_type"]
        return fle_key

class LongformerDictionaryAgent(HuggingFaceDictionaryAgent):

    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.
        """
        return True

    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        fle_key = opt["longformer_type"]
        return AutoTokenizer.from_pretrained(fle_key, use_fast=False)

    def _define_special_tokens(self, opt):
        if opt["add_special_tokens"]:
            # Add addtional start/end/pad tokens
            self.unk_token = self.tokenizer.unk_token
            self.sep_token = self.tokenizer.sep_token
            self.pad_token = self.tokenizer.pad_token
            self.cls_token = self.tokenizer.cls_token
            self.mask_token = self.tokenizer.mask_token
            self.no_answer_token = self.tokenizer.unk_token

    def override_special_tokens(self, opt):
        # define special tokens
        self._define_special_tokens(opt)
        # now override
        self.unk_idx = self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]
        self.sep_idx = self.tokenizer.convert_tokens_to_ids([self.sep_token])[0]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]
        self.cls_idx = self.tokenizer.convert_tokens_to_ids([self.cls_token])[0]
        self.mask_idx = self.tokenizer.convert_tokens_to_ids([self.mask_token])[0]
        # set tok2ind for special tokens
        self.tok2ind[self.unk_token] = self.unk_idx
        self.tok2ind[self.sep_token] = self.sep_idx
        self.tok2ind[self.pad_token] = self.pad_idx
        self.tok2ind[self.cls_token] = self.cls_idx
        self.tok2ind[self.mask_token] = self.mask_idx
        # set ind2tok for special tokens
        self.ind2tok[self.unk_idx] = self.unk_token
        self.ind2tok[self.sep_idx] = self.sep_token
        self.ind2tok[self.pad_idx] = self.pad_token
        self.ind2tok[self.cls_idx] = self.cls_token
        self.ind2tok[self.mask_idx] = self.mask_token


class LongformerQaAgent(TorchSpanAgent):
    """
    Hugging Face Bert Agent.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(LongformerQaAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group("LongformerQA Args")
        # query maximum length
        agent.add_argument(
            '--history-maximum-length',
            type=int,
            default=2046,
            help='maximum number of tokens allowed for history string',
        )
        # query maximum length
        agent.add_argument(
            '--query-maximum-length',
            type=int,
            default=512,
            help='maximum number of tokens allowed for quenry string',
        )
        # context maximum length
        agent.add_argument(
            '--context-maximum-length',
            type=int,
            default=1534,
            help='maximum number of tokens allowed for context string',
        )
        agent.add_argument(
            "--longformer-type",
            type=str,
            default="allenai/longformer-base-4096",
            choices=["allenai/longformer-base-4096", "allenai/longformer-large-4096", "mrm8488/longformer-base-4096-finetuned-squadv2"],
            help="Which size model to initialize.",
        )
        agent.add_argument(
            "--add-special-tokens",
            type="bool",
            default=True,
            help="Add special tokens (like PAD, etc.). If False, "
            "Can only use with batch size 1.",
        )
        agent.add_argument(
            "--add-global-query-attention",
            type="bool",
            default=True,
            help="Add global attention to query tokens "
            "Can only use with batch size 1.",
        )
        argparser.set_defaults(
            text_truncate=4096,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        warn_once("WARNING: this model is in beta and the API is subject to change.")
        return agent

    def __init__(self, opt, shared=None):
        self.add_cls_token = opt.get('add_cls_token', False)
        self.sep_last_utt = opt.get('sep_last_utt', False)
        if not opt["add_special_tokens"] and opt["batchsize"] > 1:
            raise RuntimeError(
                "If using batchsize > 1, --add-special-tokens must be True."
            )
        super().__init__(opt, shared)
        self.query_truncate = opt['query_maximum_length']
        self.context_truncate = opt['context_maximum_length']
        self.history_truncate = opt['history_maximum_length']
        self.global_query_attention = opt.get('--add-global-query-attention', False)
        self.truncate = self.dict.tokenizer.model_max_length
        self.doc_stride = self.context_truncate

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return LongformerDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return HFLongformerQAModel(self.opt)

    # Tokenize our training dataset
    def convert_to_features(self, example):
        # Tokenize contexts and questions (as pairs of inputs)
        input_pairs = [example['question'], example['context']]
        encodings = self.dict.tokenizer.encode_plus(input_pairs, pad_to_max_length=True, max_length=512)
        context_encodings = self.dict.tokenizer.encode_plus(example['context'])

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text
        start_idx, end_idx = self.get_correct_alignement(example['context'], example['answers'])
        start_positions_context = context_encodings.char_to_token(start_idx)
        end_positions_context = context_encodings.char_to_token(end_idx - 1)

        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example
        sep_idx = encodings['input_ids'].index(self.tokenizer.sep_token_id)
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        if end_positions > self.context_truncate:
            start_positions, end_positions = 0, 0

        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions,
                          'attention_mask': encodings['attention_mask']})
        return encodings