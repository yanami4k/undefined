"""正向匹配：先有自然的用户消息，再按情境从反应库里挑一条真弹幕接，或者不接。

    python -m scripts.gen_seeds --name natural2 --total 300 --seed 8      # 先生成自然的用户消息
    python -m scripts.reaction_lib                                        # 从人工打标整理反应库
    python -m scripts.match_synth --seeds natural2 --name round5 --limit 200

整个反应库都给老师看（放在提示词前缀、顺序固定，可命中缓存），每条梗都带着它在人工打标里真实的成功用例和翻车用例，
老师按「情境、动作」判断对不对得上，而不是看有没有共同的关键词。
另外补几条按话题检索到的梗（标为未验证）。同一个梗的不同写法算一族（见 Families），每族在一批里最多用 MAX_USES 次；
人工打标里已经有 HISTORY_CAP 条以上的梗族不再配，标注留给新梗。
需要明确答案的问题直接不接，走正常回答。
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
import threading
from collections import Counter

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import format_context, load_taste
from meme6657.retrieval import Context, Retriever, bigrams

OUT_DIR = DATA_DIR / "reverse"
REACTIONS_PATH = DATA_DIR / "gold" / "reactions.json"
GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"
MAX_USES = 3  # 同一族梗在一批里最多用几次
HISTORY_CAP = 6  # 人工打标里已经有这么多条的梗族，新批次不再配（标注留给新梗，老梗要多出现靠情境复用）

PREFIX = """场景：用户在和一个「丰川祥子本人，说话软萌、开口却全是 6657 烂梗」人设的聊天 AI 一对一聊天，或者在 QQ 群里 @ 它。
这个 AI 像弹幕一样接话，用一句直播间的真弹幕原样接上。

{taste}

---

反应库（只能原样用，一个字不改）。每条后面附了它在懂行的人那里**真实的成功用例**和**翻车用例**：
{library}"""

SUFFIX = """---

用户说：{user}
{notes}{extra}
怎么挑：
1. 先判断用户是不是需要一个明确答案：问事实 / 数据 / 选手位置、问怎么做、要推荐或建议、让 AI 干具体的活。是的话选 0，走正常回答。
   吐槽、发泄、自嘲、炫耀、闲聊分享、饭圈腔、挑衅、夸 AI、骂 AI 都**不算**，这些恰恰是接梗的好时机。
   但只要用户问了一个有明确取向的问题（要不要、该不该、值不值得、是不是、有啥推荐、怎么办），哪怕带着情绪，也不接电波向的梗；纯吐槽、发泄、分享才接。
   再看语气：用户认真低落（想找人说说话、整个人麻了、日子没意思、图什么）→ 选 0，好好回应；
   用户骂模型可以骂回去，用户玩笑口吻地抱怨、自嘲可以顺势损一句。平淡的日常闲聊没有特别贴的就选 0。
2. 看情境和动作，不看关键词：用户此刻在干什么，哪条弹幕的成功用例是同一种情境。
3. 这些情况选 0：只因为沾了同一个词；把用户自己说的话复读回去；和翻车用例的情境相似。
4. 宁缺毋滥，没有真对得上的就选 0。
{exhausted}
输出 JSON：{{"choice": "L编号、R编号或 0", "why": "这条消息的情境，以及为什么对得上（选 0 时说明原因）"}}"""


def render_library(lib: list[dict]) -> str:
    lines = []
    for i, e in enumerate(lib, 1):
        lines.append(f"L{i}. {e['text'][:120]}")
        for w in e["worked"][:2]:
            lines.append(f"   ✓{'电波' if w['wave'] else '好'}：用户「{w['user'][:50]}」")
        for f in e["failed"][:1]:
            lines.append(f"   ✗翻车：用户「{f['user'][:50]}」（{'、'.join(f['reasons']) or '不对胃'}）")
    return "\n".join(lines)


def retrieved(user: str, retriever: Retriever, lib_texts: set[str], exhausted, k: int = 5) -> list[str]:
    out = []
    for m in retriever.search_memes(user, k=k + 3):
        if m.text not in lib_texts and not exhausted(m.text) and len(m.text) <= 120:
            out.append(m.text)
        if len(out) >= k:
            break
    return out


class Families:
    """把同一个梗的不同写法归成一族，限次数时按族算。

    命中黑话类外号（汗还是尿、爆了、黏……）的按外号归；否则和反应库里的梗二元组重合度超过一半的，归到那条梗；
    都不是就自成一族。
    """

    def __init__(self, retriever: Retriever, lib_texts: list[str], threshold: float = 0.5):
        self.retriever = retriever
        self.reps = [(t, set(bigrams(t))) for t in lib_texts]
        self.threshold = threshold
        self.cache: dict[str, str] = {}

    def __call__(self, text: str) -> str:
        if text not in self.cache:
            self.cache[text] = self._family(text)
        return self.cache[text]

    def _family(self, text: str) -> str:
        terms = [e.canonical for e in self.retriever.match_aliases(text) if e.kind == "term"]
        if terms:
            return "黑话:" + terms[0]
        grams = set(bigrams(text))
        for rep, rg in self.reps:
            if grams and rg and len(grams & rg) / len(grams | rg) >= self.threshold:
                return rep
        return text


def is_echo(user: str, reply: str) -> bool:
    """回复基本是在复读用户自己的话。"""
    a, b = "".join(user.split()), "".join(reply.split())
    if not a or not b:
        return False
    return b in a or a in b or difflib.SequenceMatcher(None, a, b).ratio() > 0.6


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="natural2", help="data/synth/<seeds>.jsonl")
    ap.add_argument("--name", default="round5")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = [s for s in read_jsonl(DATA_DIR / "synth" / f"{args.seeds}.jsonl") if s["split"] == "train"]
    rng.shuffle(seeds)
    seeds = seeds[: args.limit]
    retriever = Retriever.from_files()
    lib = json.loads(REACTIONS_PATH.read_text(encoding="utf-8"))
    taste = load_taste()
    teacher = LLM("teacher")
    out_path = OUT_DIR / f"{args.name}.jsonl"
    lib_texts = {e["text"] for e in lib}
    family = Families(retriever, [e["text"] for e in lib])
    history = Counter(family(g["source"]) for g in read_jsonl(GOLD_PATH)
                      if g.get("kind") == "meme" and g.get("source")) if GOLD_PATH.exists() else Counter()
    usage: Counter = Counter(family(r["source"]) for r in read_jsonl(out_path) if r.get("source"))
    lock = threading.Lock()

    def exhausted(text: str) -> bool:
        f = family(text)
        return usage[f] >= MAX_USES or history[f] >= HISTORY_CAP

    retired = sum(history[family(t)] >= HISTORY_CAP for t in lib_texts)
    print(f"反应库 {len(lib)} 条，其中 {retired} 条所在的梗族已有 {HISTORY_CAP} 条以上标注，本批不再配")
    prefix = PREFIX.format(taste=taste, library=render_library(lib))

    def work(seed: dict) -> dict:
        msg = seed["message"]
        with lock:
            extra_memes = retrieved(msg, retriever, lib_texts, exhausted)
        extra = ""
        if extra_memes:
            listing = "\n".join(f"R{i}. {t}" for i, t in enumerate(extra_memes, 1))
            extra = "\n补充候选（按话题检索到，未经人工验证）：\n" + listing + "\n"
        with lock:
            full = [f"L{i}" for i, e in enumerate(lib, 1) if exhausted(e["text"])]
        exhausted = ("\n这些编号本批已经用满，不要选：" + "、".join(full) + "\n") if full else ""
        notes = format_context(Context(aliases=retriever.match_aliases(msg)))
        notes_block = ("相关知识：\n" + notes + "\n") if notes else ""
        suffix = SUFFIX.format(user=msg, notes=notes_block, extra=extra, exhausted=exhausted)
        data = teacher.chat_json([{"role": "user", "content": prefix + "\n\n" + suffix}],
                                 temperature=0.4, max_tokens=400, task="match")
        choice = str(data.get("choice") or "0").strip().upper()
        text, verified = None, False
        if choice.startswith("L") and choice[1:].isdigit() and 1 <= int(choice[1:]) <= len(lib):
            text, verified = lib[int(choice[1:]) - 1]["text"], True
        elif choice.startswith("R") and choice[1:].isdigit() and 1 <= int(choice[1:]) <= len(extra_memes):
            text = extra_memes[int(choice[1:]) - 1]
        why = data.get("why", "")
        if text and is_echo(msg, text):
            text, why = None, "（程序拦截：复读用户原话）" + why
        if text:
            with lock:
                if exhausted(text):
                    text = None
                else:
                    usage[family(text)] += 1
        return {"id": seed["id"], "category": seed["category"], "user": msg,
                "choice": choice if text else 0, "kind": "meme" if text else "",
                "source": text or "", "reply": text or "", "verified": verified if text else False, "why": why}

    run_parallel(seeds, work, out_path, workers=args.workers)

    rows = read_jsonl(out_path)
    matched = [r for r in rows if r["choice"]]
    print(f"\n{len(rows)} 条用户消息，接上 {len(matched)} 条（{len(matched) / max(1, len(rows)):.0%}），"
          f"其中来自反应库 {sum(r['verified'] for r in matched)} 条；其余走正常回答")


if __name__ == "__main__":
    main()
