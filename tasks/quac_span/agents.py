from parlai.tasks.quac.agents import DefaultTeacher as quac_teacher,_path
from parlai_internal.utilities import util
from transformers.data.processors.squad import SquadExample

NO_ANSWER_REPLY = "CANNOTANSWER"

class DefaultTeacher(quac_teacher):

    def get(self, episode_idx, entry_idx=None):
        """
        Get a specific example from the dataset.
        """
        ex = self.episodes[episode_idx][entry_idx]
        is_training = self.datatype == "train"
        qas_id = str(episode_idx) + "_" + str(entry_idx)
        start_episode_text = self.episodes[episode_idx][0]['text']
        article_end_index = start_episode_text.index(" CANNOTANSWER\n")
        context_text = start_episode_text[:article_end_index]
        # doc_tokens, char_to_word_offset = util.build_char_word_offset_list(context)
        text = ex['text']
        if entry_idx == 0:
            question_text = text[article_end_index+14:]
        else:
            question_text = text
        answer_text = ex['labels'][0]
        start_position_character = None
        is_impossible = answer_text == NO_ANSWER_REPLY
        if not is_impossible:
            answers = ex['labels']
            start_position_character = int(ex["answer_starts"])
        else:
            answers = [NO_ANSWER_REPLY]

        squad_example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title="unknown title",
                        is_impossible=is_impossible,
                        answers=answers,
                    )

        action = {
            'id': 'squad',
            'qas_id': qas_id,
            'labels': answers,
            'context': context_text,
            'squad_example': squad_example,
            'single_label_text': answer_text,
            'episode_done': ex['episode_done'],
            'is_impossible': is_impossible,
            'text': question_text,
            'no_answer_reply': NO_ANSWER_REPLY
        }
        return action

