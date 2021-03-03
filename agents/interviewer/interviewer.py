from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai_internal.agents.interviewee.interviewee import IntervieweeDictionaryAgent






class IntervieweeAgent(TorchGeneratorAgent):
    """
    Interviewer agent.
    """

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return IntervieweeDictionaryAgent

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return IntervieweeHistory





