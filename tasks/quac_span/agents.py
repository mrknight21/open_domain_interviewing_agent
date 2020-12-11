from parlai.tasks.quac.agents import DefaultTeacher as quac_teacher,_path
from parlai_internal.utilities import util

NO_ANSWER_REPLY = "[NOANSWER]"

class DefaultTeacher(quac_teacher):

    def get(self, episode_idx, entry_idx=None):
        """
        Get a specific example from the dataset.
        """
        ex = self.episodes[episode_idx][entry_idx]
        start_episode_text = self.episodes[episode_idx][0]['text']
        article_end_index = start_episode_text.index(" CANNOTANSWER\n")
        context = start_episode_text[:article_end_index]
        doc_tokens, char_to_word_offset = util.build_char_word_offset_list(context)
        text = ex['text']
        if entry_idx == 0:
            question_text = text[article_end_index+14:]
        else:
            question_text = text
        labels = ex['labels']
        single_label_text = ex['labels'][0]
        is_impossible = single_label_text == "CANNOTANSWER"
        if not is_impossible:
            orig_answer_text = single_label_text
            answer_offset = int(ex["answer_starts"])
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]
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
            if actual_text.find(cleaned_answer_text) == -1:
                print("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                return None
        else:
            start_position = -1
            end_position = -1
            single_label_text = ""
            labels = [""]

        action = {
            'id': 'quac',
            'qas_id': str(episode_idx) + "_" + str(entry_idx),
            'text': context + ' ' + question_text,
            'doc_tokens': doc_tokens,
            'context': context,
            'question_text': question_text,
            'labels': labels,
            'single_label_text': single_label_text,
            'episode_done': ex['episode_done'],
            'start_position': start_position,
            'end_position': end_position,
            'char_answer_start': ex['answer_starts'],
            'is_impossible': is_impossible
        }
        return action
