
from parlai.tasks.squad2.agents import IndexTeacher

import copy

NO_ANSWER_REPLY = "I don't know"

class DefaultTeacher(IndexTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if self.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

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
                self.whitespace_tokenize(orig_answer_text))
            eva_labels_text = [ans['text'] for ans in qa["answers"]]
            if actual_text.find(cleaned_answer_text) == -1:
                print("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                return None
        else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""
            eva_labels_text = [""]



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

    def get_start_end_idx(self, gold_text, start_idx, context):
        end_idx = start_idx + len(gold_text)
        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            start_idx = start_idx - 1
            end_idx = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            start_idx = start_idx - 2
            end_idx = end_idx - 2  # When the gold label is off by two characters
        return start_idx, end_idx

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
          return True
        return False

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens
