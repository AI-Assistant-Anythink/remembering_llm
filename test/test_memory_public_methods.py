"""Самопроверка публичных методов работы с памятью:
add_short_term_message, add_long_term_memory, flush_short_term_memory."""

import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from remembering_llm import RememberingLLM
from remembering_llm.short_term_memory import InMemoryShortTermMemory


class FakeLongTermMemory:
    def __init__(self):
        self.added: list[tuple[list[dict], str]] = []

    async def add(self, messages, user_id):
        self.added.append((messages, user_id))


async def demo():
    long_term = FakeLongTermMemory()
    short_term = InMemoryShortTermMemory()

    llm = RememberingLLM(
        long_term_memory=long_term,
        short_term_memory=short_term,
        main_llm=FakeListChatModel(responses=["ok"]),
    )

    user_id = "u1"

    # 1. add_short_term_message
    await llm.add_short_term_message(user_id, HumanMessage(content="привет"))
    await llm.add_short_term_message(user_id, AIMessage(content="привет!"))
    dialog = await short_term.get_dialog(user_id)
    assert [m.message.content for m in dialog] == ["привет", "привет!"], dialog

    # 2. add_long_term_memory
    await llm.add_long_term_memory(user_id, "любит кофе")
    assert long_term.added == [([{"role": "user", "content": "любит кофе"}], user_id)]

    # 3. flush_short_term_memory — переносит ВСЁ и полностью чистит short-term
    await llm.flush_short_term_memory(user_id)
    assert await short_term.count_messages(user_id) == 0
    flushed_messages, flushed_user = long_term.added[-1]
    assert flushed_user == user_id
    assert len(flushed_messages) == 2
    assert flushed_messages[0]["role"] == "user"
    assert flushed_messages[1]["role"] == "assistant"

    # flush на пустой short-term ничего не должен слать в long-term
    calls_before = len(long_term.added)
    await llm.flush_short_term_memory(user_id)
    assert len(long_term.added) == calls_before

    print("OK")


if __name__ == "__main__":
    asyncio.run(demo())
