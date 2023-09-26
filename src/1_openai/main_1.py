"""
Entry point for the chatbot.
"""
from chatbot_1 import Chatbot


def main():
    chatbot = Chatbot()
    chatbot.ask("I need a large backpack. Which one do you recommend?")
    chatbot.ask("How much does it cost?")
    chatbot.ask("And how much for a donut?")


if __name__ == "__main__":
    main()
