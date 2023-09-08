"""
Base class for the Chatbots with context.
"""
from abc import ABCMeta, abstractmethod


class AbstractChatbot(metaclass=ABCMeta):
    @abstractmethod
    def ask(self, context_list: list[str], question: str) -> str:
        pass
