from __future__ import annotations

import json
from typing import Any


class RedisMemoryError(RuntimeError):
    """Raised when the Redis memory backend cannot be used."""


class RedisMemory:
    """Persistent conversation memory backed by Redis.

    Stores:
      - chat messages
      - attached website/PDF/TXT/MD sources
      - completed agent turns (question, answer, tools, verification)
      - lightweight conversation metadata

    Each conversation is isolated by conversation_id and expires after ttl_seconds
    since the most recent write. Message and turn lists are bounded so one chat
    cannot grow without limit.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        ttl_seconds: int = 2_592_000,
        max_messages: int = 200,
        max_turns: int = 100,
    ) -> None:
        if not redis_url.strip():
            raise RedisMemoryError("REDIS_URL is empty")

        try:
            import redis
        except ImportError as exc:
            raise RedisMemoryError(
                "Redis support requires the redis package. Run: uv add redis"
            ) from exc

        if ttl_seconds < 60:
            raise ValueError("ttl_seconds must be at least 60")
        if max_messages < 10:
            raise ValueError("max_messages must be at least 10")
        if max_turns < 10:
            raise ValueError("max_turns must be at least 10")

        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            health_check_interval=30,
        )
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self.max_turns = max_turns

    def _prefix(self, conversation_id: str) -> str:
        safe = conversation_id.strip()
        if not safe or len(safe) > 128:
            raise ValueError("Invalid conversation_id")
        return f"aoi:conversation:{safe}"

    def _messages_key(self, conversation_id: str) -> str:
        return f"{self._prefix(conversation_id)}:messages"

    def _sources_key(self, conversation_id: str) -> str:
        return f"{self._prefix(conversation_id)}:sources"

    def _turns_key(self, conversation_id: str) -> str:
        return f"{self._prefix(conversation_id)}:turns"

    def _meta_key(self, conversation_id: str) -> str:
        return f"{self._prefix(conversation_id)}:meta"

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as exc:  # noqa: BLE001
            raise RedisMemoryError(f"Redis connection failed: {exc}") from exc

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def _touch(self, conversation_id: str) -> None:
        keys = (
            self._messages_key(conversation_id),
            self._sources_key(conversation_id),
            self._turns_key(conversation_id),
            self._meta_key(conversation_id),
        )
        pipe = self.client.pipeline(transaction=True)
        for key in keys:
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def load_messages(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        key = self._messages_key(conversation_id)
        raw = self.client.lrange(key, -(limit or self.max_messages), -1)
        messages: list[dict[str, str]] = []
        for item in raw:
            try:
                value = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                role = str(value.get("role", "user"))
                content = str(value.get("content", ""))
                messages.append({"role": role, "content": content})
        return messages

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        key = self._messages_key(conversation_id)
        payload = json.dumps(
            {"role": role, "content": content[:20000]},
            ensure_ascii=False,
        )
        pipe = self.client.pipeline(transaction=True)
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.execute()
        self._touch(conversation_id)

    def load_sources(self, conversation_id: str) -> list[dict[str, Any]]:
        raw = self.client.get(self._sources_key(conversation_id))
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

    def save_sources(
        self,
        conversation_id: str,
        sources: list[dict[str, Any]],
    ) -> None:
        cleaned: list[dict[str, Any]] = []
        for source in sources[-50:]:
            cleaned.append(
                {
                    "name": str(source.get("name", ""))[:500],
                    "type": str(source.get("type", "document"))[:50],
                    "content": str(source.get("content", ""))[:100000],
                }
            )
        self.client.set(
            self._sources_key(conversation_id),
            json.dumps(cleaned, ensure_ascii=False),
        )
        self._touch(conversation_id)

    def append_turn(
        self,
        conversation_id: str,
        *,
        question: str,
        final_text: str,
        result: dict[str, Any],
    ) -> None:
        turn = {
            "question": question[:10000],
            "final_text": final_text[:20000],
            "verified": bool(result.get("verified")),
            "verification_errors": result.get("verification_errors", []),
            "tool_names": sorted(
                str(name) for name in result.get("tool_results", {}).keys()
            ),
            "evidence_ids": result.get("evidence_ids", [])[:100],
        }
        key = self._turns_key(conversation_id)
        pipe = self.client.pipeline(transaction=True)
        pipe.rpush(
            key,
            json.dumps(turn, ensure_ascii=False, default=str),
        )
        pipe.ltrim(key, -self.max_turns, -1)
        pipe.execute()
        self._touch(conversation_id)

    def load_turns(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        raw = self.client.lrange(
            self._turns_key(conversation_id),
            -limit,
            -1,
        )
        turns: list[dict[str, Any]] = []
        for item in raw:
            try:
                value = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                turns.append(value)
        return turns

    def get_meta(self, conversation_id: str) -> dict[str, str]:
        return {
            str(k): str(v)
            for k, v in self.client.hgetall(self._meta_key(conversation_id)).items()
        }

    def set_meta(
        self,
        conversation_id: str,
        **values: str,
    ) -> None:
        if values:
            self.client.hset(
                self._meta_key(conversation_id),
                mapping={k: v[:2000] for k, v in values.items()},
            )
            self._touch(conversation_id)

    def snapshot(self, conversation_id: str) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "messages": self.load_messages(conversation_id),
            "sources": self.load_sources(conversation_id),
            "turns": self.load_turns(conversation_id),
            "meta": self.get_meta(conversation_id),
        }

    def clear(self, conversation_id: str) -> None:
        self.client.delete(
            self._messages_key(conversation_id),
            self._sources_key(conversation_id),
            self._turns_key(conversation_id),
            self._meta_key(conversation_id),
        )
