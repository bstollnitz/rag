"""
Entry point for the chatbot.
"""
import asyncio

from chatbot_3 import Chatbot


async def main():
    chatbot = Chatbot()
    await chatbot.ask("I need a large backpack. Which one do you recommend?")
    await chatbot.ask("How much does it cost?")
    await chatbot.ask("And how much for a donut?")


if __name__ == "__main__":
    asyncio.run(main())
