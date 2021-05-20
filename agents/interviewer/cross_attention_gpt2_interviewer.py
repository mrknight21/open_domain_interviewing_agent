
from parlai_internal.agents.interviewer.interviewer import InterviewerAgent
from parlai_internal.agents.interviewer.gpt_interviewer import GptInterviewerDictionaryAgent
from parlai.agents.hugging_face.gpt2 import Gpt2DictionaryAgent,  HFGPT2Model
from parlai_internal.utilities.flow_lstm_util import constants


GPT2_SPECIAL_TOKENS = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                  "additional_special_tokens": [constants.QUESST, constants.QUESEN,
                                                constants.TITLEST, constants.TITLEEN,
                                                constants.SECST, constants.SECEN,
                                                constants.BGST, constants.BGEN,
                                                constants.ANSST, constants.ANSEN]}

NO_OP = "x"
#
# class CrossAttentionGptInterviewerAgent(InterviewerAgent):
#
