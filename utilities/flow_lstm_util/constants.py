PAD = '<PAD>'
UNK = '<UNK>'
SOS = '<SOS>'
EOS = '<EOS>'
QUESST = '<QUES>'
QUESEN = '</QUES>'
ANSST = '<ANS>'
ANSEN = '</ANS>'
TITLEST = '<WIKITITLE>'
TITLEEN = '</WIKITITLE>'
SECST = '<SECTITLE>'
SECEN = '</SECTITLE>'
BGST = '<BG>'
BGEN = '</BG>'
CHAR_START = '<CHARSTART>'
CHAR_END = '<CHAREND>'
SEP ='<SEP>'
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
QUESST_ID = 4
QUESEN_ID = 5
ANSST_ID = 6
ANSEN_ID = 7
TITLEST_ID = 8
TITLEEN_ID = 9
SECST_ID = 10
SECEN_ID = 11
BGST_ID = 12
BGEN_ID = 13
CHAR_START_ID = 14
CHAR_END_ID = 15

VOCAB_PREFIX = [PAD, UNK, SOS, EOS, QUESST, QUESEN, ANSST, ANSEN, TITLEST, TITLEEN, SECST, SECEN, BGST, BGEN, CHAR_START, CHAR_END]

MAX_CONTEXT = 2000
MAX_BACKGROUND = 300
MAX_TURNS = 12

FOLLOWUP_TO_ID = { 'y': 0, 'm': 1, 'n': 2 }
YESNO_TO_ID = { 'y': 0, 'n': 1, 'x': 2 }

VOCAB_FILE = 'Interviewee/vocab.pkl'
TEACHER_FILE = 'Interviewee/teacher_model.pt'
FINE_TUNE_FILE = 'Interviewee/finetuned_model.pt'
BASE_MODEL_FILE = 'Interviewee/model.pt'
