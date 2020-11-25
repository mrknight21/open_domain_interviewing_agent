
from parlai.tasks.squad2.agents import IndexTeacher
from parlai_internal.utilities import util

import copy

NO_ANSWER_REPLY = "[CLS]"

class DefaultTeacher(IndexTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        paragraph_text = paragraph["context"]
        doc_tokens, char_to_word_offset = util.build_char_word_offset_list(paragraph)
        qa = paragraph["qas"][qa_idx]
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        is_impossible = qa["is_impossible"]
        if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length -
                                               1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                util.whitespace_tokenize(orig_answer_text))
            eva_labels_text = [ans['text'] for ans in qa["answers"]]
            if actual_text.find(cleaned_answer_text) == -1:
                print("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                return None
        else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""
            eva_labels_text = [NO_ANSWER_REPLY]



        action = {
            'id': 'squad',
            'qas_id': qas_id,
            'text': paragraph_text + ' [SEP] ' + question_text,
            'doc_tokens': doc_tokens,
            'context': paragraph_text,
            'question_text': question_text,
            'labels': eva_labels_text,
            'single_label_text': orig_answer_text,
            'episode_done': True,
            'start_position': start_position,
            'end_position': end_position,
            'is_impossible': is_impossible
        }
        return action


