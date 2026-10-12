"""用老师模型合成训练对话（上下文蒸馏）。

    python -m scripts.synth_dialogues --split train
    python -m scripts.synth_dialogues --split train --limit 50   # 先小批量看效果

每条种子生成一段对话：
- chat 模式（默认 80%）：1~4 轮。每轮先检索参考梗，老师看「风格卡 + 参考梗」回复；
  后续轮次的用户消息由用户模拟器生成。
- rewrite 模式（20%）：先让老师用普通助手口吻回答，再改写成 6657 风格（单轮）。

训练样本里只保存学生视角：系统提示词 = STUDENT_SYSTEM，不含风格卡。
输出 data/synth/dialogues.jsonl。
"""

from __future__ import annotations

import argparse
import random
import sys

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import (
    STUDENT_SYSTEM,
    format_user,
    load_style_card,
    plain_messages,
    teacher_messages,
    usersim_messages,
)
from meme6657.retrieval import Retriever

SYNTH_DIR = DATA_DIR / "synth"
SEEDS_PATH = SYNTH_DIR / "seeds.jsonl"
OUT_PATH = SYNTH_DIR / "dialogues.jsonl"

# 正经求助、闲聊类更适合做改写样本；挑衅类不做改写
REWRITE_CATEGORIES = {"help", "daily", "cs_match", "vent", "streamer", "anime", "meme_ask_known"}
TURN_WEIGHTS = {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.15}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "eval", "all"])
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条种子，0 为全部")
    ap.add_argument("--per-category", type=int, default=0, help="每个类别只取 N 条，用于小批量试点")
    ap.add_argument("--rewrite-ratio", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    seeds = [s for s in read_jsonl(SEEDS_PATH) if args.split == "all" or s["split"] == args.split]
    if args.per_category:
        counts: dict[str, int] = {}
        picked = []
        for s in seeds:
            if counts.get(s["category"], 0) < args.per_category:
                counts[s["category"]] = counts.get(s["category"], 0) + 1
                picked.append(s)
        seeds = picked
    if args.limit:
        seeds = seeds[: args.limit]
    retriever = Retriever.from_files()
    style_card = load_style_card()
    teacher, usersim = LLM("teacher"), LLM("usersim")

    def reply(history: list[dict]) -> str:
        return teacher.chat(teacher_messages(history, style_card), temperature=args.temperature,
                            max_tokens=800, task="chat").strip()

    def work(seed: dict) -> dict:
        rng = random.Random(seed["id"])
        do_rewrite = seed["category"] in REWRITE_CATEGORIES and rng.random() < args.rewrite_ratio

        if do_rewrite:
            msg = seed["message"]
            source = teacher.chat(plain_messages(msg), temperature=0.3, max_tokens=600, task="plain").strip()
            history = [{"role": "user", "content": format_user(msg, retriever.retrieve(msg), source)}]
            history.append({"role": "assistant", "content": reply(history)})
            mode, plain = "rewrite", [{"role": "user", "content": msg}, {"role": "assistant", "content": source}]
        else:
            n_turns = rng.choices(list(TURN_WEIGHTS), weights=list(TURN_WEIGHTS.values()))[0]
            if seed["category"] in ("meme_ask_unknown", "provoke"):
                n_turns = min(n_turns, 2)
            history, plain = [], []  # plain：不含 [参考梗] 的原始对话，给用户模拟器看
            msg = seed["message"]
            for t in range(n_turns):
                if t > 0:
                    msg = usersim.chat(usersim_messages(seed["persona"], plain), temperature=1.0,
                                       max_tokens=120, task="usersim").strip().strip('"“”')
                history.append({"role": "user", "content": format_user(msg, retriever.retrieve(msg))})
                plain.append({"role": "user", "content": msg})
                answer = reply(history)
                history.append({"role": "assistant", "content": answer})
                plain.append({"role": "assistant", "content": answer})
            mode = "chat"

        return {
            "id": seed["id"],
            "category": seed["category"],
            "split": seed["split"],
            "mode": mode,
            "messages": [{"role": "system", "content": STUDENT_SYSTEM}, *history],
            "plain": plain,
        }

    run_parallel(seeds, work, OUT_PATH, workers=args.workers)


if __name__ == "__main__":
    main()
