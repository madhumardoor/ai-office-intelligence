from __future__ import annotations

import os
import sys
import uuid

from app.memory.redis_memory import RedisMemory


def main() -> int:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    conversation_id = f"check-{uuid.uuid4()}"
    memory = RedisMemory(url, ttl_seconds=300, max_messages=20, max_turns=10)

    try:
        print("Redis URL:", url)
        print("PING:", memory.ping())

        memory.append_message(conversation_id, "user", "Hello")
        memory.append_message(conversation_id, "assistant", "Hello. How can I help?")
        memory.save_sources(
            conversation_id,
            [{"name": "sample.txt", "type": "document", "content": "Sample company information."}],
        )
        memory.append_turn(
            conversation_id,
            question="Tell me about the company",
            final_text="The company information was loaded.",
            result={"verified": True, "tool_results": {}, "evidence_ids": ["E1"]},
        )

        snapshot = memory.snapshot(conversation_id)
        print("Messages:", len(snapshot["messages"]))
        print("Sources:", len(snapshot["sources"]))
        print("Turns:", len(snapshot["turns"]))
        print("PASS Redis memory read/write")
        return 0
    finally:
        try:
            memory.clear(conversation_id)
        finally:
            memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
