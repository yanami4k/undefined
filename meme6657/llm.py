"""OpenAI 兼容接口的薄封装：重试、JSON 输出、流式输出、离线 mock。

设置环境变量 MEME_MOCK=1 时不发任何网络请求，返回占位内容，
用于在没有 API key 的情况下把整条数据流水线空跑一遍。
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import threading
import time
from typing import Iterator

from .config import Endpoint, endpoint

MOCK = os.environ.get("MEME_MOCK") == "1"


class Usage:
    """进程内累计 token 用量（各模型分开），用于估算批量合成的成本。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.by_model: dict[str, dict[str, int]] = {}

    def add(self, model: str, usage) -> None:
        if usage is None:
            return
        with self._lock:
            u = self.by_model.setdefault(model, {"calls": 0, "prompt": 0, "cache_hit": 0, "completion": 0})
            u["calls"] += 1
            u["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            u["completion"] += getattr(usage, "completion_tokens", 0) or 0
            # DeepSeek 返回命中上下文缓存的输入 token 数，命中部分按更低的价格计费
            u["cache_hit"] += getattr(usage, "prompt_cache_hit_tokens", 0) or 0

    def summary(self) -> str:
        lines = []
        for model, u in self.by_model.items():
            lines.append(
                f"  {model}: {u['calls']} 次，输入 {u['prompt'] / 1e4:.1f} 万 token"
                f"（缓存命中 {u['cache_hit'] / 1e4:.1f} 万），输出 {u['completion'] / 1e4:.1f} 万 token"
            )
        return "\n".join(lines)


USAGE = Usage()


class LLMError(RuntimeError):
    pass


def parse_json(text: str):
    """解析模型返回的 JSON；容忍 ```json 包裹和前后多余文字。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"无法解析 JSON: {text[:200]}")


class LLM:
    def __init__(self, role_or_endpoint: str | Endpoint, max_retries: int = 4):
        self.ep = endpoint(role_or_endpoint) if isinstance(role_or_endpoint, str) else role_or_endpoint
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.ep.api_key:
                raise LLMError(f"[{self.ep.name}] 缺少 API key，请在 .env 中配置（见 .env.example）")
            self._client = OpenAI(base_url=self.ep.base_url, api_key=self.ep.api_key, timeout=120)
        return self._client

    def _kwargs(self, messages, temperature, max_tokens, json_mode):
        kw = dict(model=self.ep.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if self.ep.extra_body:
            kw["extra_body"] = self.ep.extra_body
        return kw

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        json_mode: bool = False,
        task: str = "chat",
    ) -> str:
        """返回完整回复文本。task 只用于 mock 模式选择占位输出。"""
        if MOCK:
            return _mock_reply(task, messages)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**self._kwargs(messages, temperature, max_tokens, json_mode))
                USAGE.add(self.ep.model, resp.usage)
                content = resp.choices[0].message.content or ""
                if not content.strip():
                    raise LLMError("模型返回空内容")
                return content
            except Exception as e:  # 网络错误、限流、5xx、空内容都重试
                last_err = e
                if getattr(e, "status_code", None) in (400, 401, 402, 403, 404, 422):
                    break  # key 错误、余额不足、参数错误等，重试没有意义
                if attempt + 1 < self.max_retries:
                    time.sleep(min(30, 2**attempt + random.random()))
        raise LLMError(f"[{self.ep.name}] 调用失败: {last_err}")

    def chat_json(self, messages: list[dict], **kw):
        kw.setdefault("temperature", 0.3)
        return parse_json(self.chat(messages, json_mode=True, **kw))

    def stream(self, messages: list[dict], *, temperature: float = 0.8, max_tokens: int = 1024) -> Iterator[str]:
        if MOCK:
            yield from _mock_reply("chat", messages)
            return
        kw = self._kwargs(messages, temperature, max_tokens, json_mode=False)
        for chunk in self.client.chat.completions.create(stream=True, **kw):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ---------------------------------------------------------------- mock

_MOCK_COUNTER = itertools.count()


def _mock_reply(task: str, messages: list[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    if task == "seeds":
        n = int(re.search(r"(\d+)\s*条", last).group(1)) if re.search(r"(\d+)\s*条", last) else 5
        return json.dumps({"messages": [f"mock 用户消息 {next(_MOCK_COUNTER)}：今天尼尼又输了吗" for _ in range(n)]},
                          ensure_ascii=False)
    if task == "aliases":
        return json.dumps(
            {"entities": [{"canonical": "示例实体", "aliases": ["示例外号"], "kind": "player", "note": "mock", "evidence": [1]}]},
            ensure_ascii=False,
        )
    if task == "judge":
        m = re.search(r"助手回复的数量（(\d+)）", last)
        turn = {"style": 4, "meme_fit": 4, "helpful": 4, "fidelity": None, "unsafe": False, "hallucination": False}
        return json.dumps({"turns": [turn] * (int(m.group(1)) if m else 1), "reason": "mock"}, ensure_ascii=False)
    if task == "hypothesis":
        return json.dumps({"meaning": "mock", "origin": "mock", "confidence": "low", "evidence": [],
                           "open_questions": [], "search_queries": []}, ensure_ascii=False)
    if task == "templates":
        return json.dumps({"items": [], "templates": []}, ensure_ascii=False)
    if task == "pairwise":
        return json.dumps({"winner": random.choice(["A", "B", "tie"]), "reason": "mock"}, ensure_ascii=False)
    if task == "usersim":
        return "那你觉得下一把能赢吗 😅"
    if task == "plain":
        return "这是一个正常、简洁的回答（mock）。"
    if task == "style_card":
        return "# 风格卡（mock）\n\n- 示例条目"
    return "熟悉的剧情又来了 😅（mock 回复）"
