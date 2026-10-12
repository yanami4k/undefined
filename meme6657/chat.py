"""命令行聊天。

    python -m meme6657.chat                      # 学生模型（训练后部署的 vLLM 服务）
    python -m meme6657.chat --backend teacher    # 老师 = DeepSeek + 风格卡（工程原型）
    python -m meme6657.chat --rewrite            # 方案 A：老师出内容，学生改写

聊天中输入：/ctx 切换是否显示检索结果，/reset 清空对话，/quit 退出。
每轮对话记录到 logs/，供后续打分、补数据使用。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

from .config import LOG_DIR
from .llm import LLM
from .prompts import format_context, format_user, plain_messages, student_messages, teacher_messages
from .retrieval import Retriever

MAX_HISTORY_TURNS = 6  # 与训练数据的对话长度大致对齐


class ChatSession:
    def __init__(self, backend: str = "student", rewrite: bool = False, retriever: Retriever | None = None):
        self.backend = backend
        self.rewrite = rewrite
        self.retriever = retriever or Retriever.from_files()
        self.speaker = LLM("teacher" if backend == "teacher" else "student")
        self.content_llm = LLM("teacher") if rewrite else None
        self.history: list[dict] = []  # 学生视角：用户消息已含 [参考梗]/[改写] 块

    def reset(self) -> None:
        self.history.clear()

    def _messages(self) -> list[dict]:
        recent = self.history[-MAX_HISTORY_TURNS * 2 :]
        return teacher_messages(recent) if self.backend == "teacher" else student_messages(recent)

    def turn(self, user_text: str):
        """生成一轮回复，逐段 yield 文本；结束后写入历史。"""
        ctx = self.retriever.retrieve(user_text)
        source = None
        if self.rewrite:
            source = self.content_llm.chat(plain_messages(user_text), temperature=0.3, task="plain")
        self.history.append({"role": "user", "content": format_user(user_text, ctx, source)})
        self.last_ctx, self.last_source = ctx, source

        parts = []
        for piece in self.speaker.stream(self._messages()):
            parts.append(piece)
            yield piece
        self.history.append({"role": "assistant", "content": "".join(parts)})


def _log(record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"chat-{datetime.now():%Y%m%d}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["student", "teacher"], default="student")
    ap.add_argument("--rewrite", action="store_true", help="老师先生成正常回答，再由说话模型改写")
    ap.add_argument("--show-ctx", action="store_true", help="显示每轮检索到的参考梗")
    args = ap.parse_args()

    session = ChatSession(args.backend, args.rewrite)
    show_ctx = args.show_ctx
    print(f"[{args.backend}{' + rewrite' if args.rewrite else ''}] 已就绪，/ctx 显示检索，/reset 清空，/quit 退出\n")

    while True:
        try:
            user_text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/reset":
            session.reset()
            print("（对话已清空）\n")
            continue
        if user_text == "/ctx":
            show_ctx = not show_ctx
            print(f"（检索显示：{'开' if show_ctx else '关'}）\n")
            continue

        t0 = time.perf_counter()
        first = None
        stream = session.turn(user_text)
        try:
            piece = next(stream)
        except StopIteration:
            piece = ""
        if show_ctx:
            print(format_context(session.last_ctx) or "（无参考梗）")
            if session.last_source:
                print(f"[内容] {session.last_source}")
        print("它> ", end="", flush=True)
        reply = []
        while True:
            if piece:
                if first is None:
                    first = time.perf_counter() - t0
                print(piece, end="", flush=True)
                reply.append(piece)
            try:
                piece = next(stream)
            except StopIteration:
                break
        total = time.perf_counter() - t0
        print(f"\n  （首字 {first or 0:.2f}s，总计 {total:.2f}s）\n")
        _log({
            "time": datetime.now().isoformat(timespec="seconds"),
            "backend": args.backend,
            "rewrite": args.rewrite,
            "user": user_text,
            "aliases": [e.canonical for e in session.last_ctx.aliases],
            "meme_ids": [m.id for m in session.last_ctx.memes],
            "source": session.last_source,
            "reply": "".join(reply),
            "first_token_s": round(first or 0, 3),
            "total_s": round(total, 3),
        })


if __name__ == "__main__":
    main()
