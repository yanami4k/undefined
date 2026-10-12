"""外号表：用 LLM 从梗库里抽候选，人工审核后合并。

    python -m scripts.build_aliases extract   # 生成 kb/aliases.candidates.json
    # 人工审核：把确认无误的条目 status 改成 "verified"，改正 note，删掉错的
    python -m scripts.build_aliases merge     # seed + 已审核候选 → kb/aliases.json

LLM 抽出来的解释可能是编的，不审核直接用会把错误知识教给模型，所以默认只合并 verified。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict

from meme6657.config import ALIASES_PATH, KB_DIR
from meme6657.jobs import run_parallel, read_jsonl
from meme6657.llm import LLM
from meme6657.retrieval import load_memes

SEED_PATH = KB_DIR / "aliases.seed.json"
CANDIDATES_PATH = KB_DIR / "aliases.candidates.json"
BATCH_OUT = KB_DIR / ".aliases_batches.jsonl"

PROMPT = """下面是斗鱼玩机器（6657）直播间的一批弹幕，每条前面是编号。
请找出其中出现的「外号 / 谐音 / 代称」，以及它们指代的对象（选手、战队、主播、动漫角色、圈内群体等）。

要求：
- 只收录在这些弹幕里有证据的外号，evidence 填支持它的弹幕编号
- canonical 用对象最通用的名字（选手用比赛 ID，如 NiKo、donk）
- note 用一两句话说明是谁、为什么这么叫；如果只能猜，note 以「待确认：」开头
- 已知的不用重复：{known}
- 普通词、单纯的脏话不算外号

输出 JSON：{{"entities": [{{"canonical": "...", "aliases": ["..."], "kind": "player|team|streamer|character|scene|other", "note": "...", "evidence": [编号]}}]}}

弹幕：
{memes}"""


def extract(batch_size: int, max_batches: int, workers: int) -> None:
    memes = load_memes()
    random.Random(0).shuffle(memes)
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    known = "、".join(t for e in seed for t in [e["canonical"], *e["aliases"]][:4])

    batches = []
    for b in range(min(max_batches, (len(memes) + batch_size - 1) // batch_size)):
        chunk = memes[b * batch_size : (b + 1) * batch_size]
        batches.append({"id": f"b{b}", "memes": [{"id": m.id, "text": m.text[:150]} for m in chunk]})

    llm = LLM("teacher")

    def work(batch: dict) -> dict:
        lines = "\n".join(f"{m['id']}. {m['text']}" for m in batch["memes"])
        data = llm.chat_json(
            [{"role": "user", "content": PROMPT.format(known=known, memes=lines)}],
            max_tokens=2048,
            task="aliases",
        )
        return {"id": batch["id"], "entities": data.get("entities", [])}

    run_parallel(batches, work, BATCH_OUT, workers=workers)

    # 跨批次合并：同一 canonical 的外号取并集，证据计数
    merged: dict[str, dict] = {}
    support = defaultdict(int)
    seed_names = {t.lower() for e in seed for t in [e["canonical"], *e["aliases"]]}
    for row in read_jsonl(BATCH_OUT):
        for ent in row["entities"]:
            name = str(ent.get("canonical", "")).strip()
            if not name:
                continue
            k = name.lower()
            cur = merged.setdefault(k, {"canonical": name, "aliases": [], "kind": ent.get("kind", ""),
                                        "note": ent.get("note", ""), "tags": [], "evidence": [],
                                        "status": "candidate"})
            for a in ent.get("aliases", []):
                if a and a not in cur["aliases"] and a.lower() not in seed_names:
                    cur["aliases"].append(a)
            cur["evidence"].extend(ent.get("evidence", []))
            support[k] += 1
    rows = [v for k, v in merged.items() if v["aliases"]]
    for v in rows:
        v["evidence"] = sorted(set(v["evidence"]))[:10]
        v["support"] = support[v["canonical"].lower()]
    rows.sort(key=lambda v: (-v["support"], -len(v["evidence"])))
    CANDIDATES_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"候选 {len(rows)} 个 → {CANDIDATES_PATH}，请人工审核后运行 merge")


def load_working_aliases() -> list[dict]:
    """当前外号表：已有 aliases.json 时在其基础上增量合并，并补上 seed 里新增的条目。"""
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    if not ALIASES_PATH.exists():
        return seed
    entries = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    names = {e["canonical"].lower() for e in entries}
    return entries + [e for e in seed if e["canonical"].lower() not in names]


def merge() -> None:
    entries = load_working_aliases()
    cands = []
    for path in (CANDIDATES_PATH, KB_DIR / "glossary.candidates.json"):  # 外号候选 + 考据候选
        if path.exists():
            cands += json.loads(path.read_text(encoding="utf-8"))
    extra = ("evidence", "support", "origin", "confidence", "open_questions", "search_queries")
    approved = [{k: v for k, v in c.items() if k not in extra} for c in cands if c.get("status") == "verified"]
    n_before = len(entries)
    by_name = {e["canonical"].lower(): e for e in entries}
    for c in approved:
        cur = by_name.get(c["canonical"].lower())
        if cur:
            cur["aliases"] += [a for a in c["aliases"] if a not in cur["aliases"]]
        else:
            entries.append(c)
            by_name[c["canonical"].lower()] = c
    ALIASES_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原有 {n_before} 条 + 审核通过 {len(approved)} 条，合并后 {len(entries)} 条 → {ALIASES_PATH}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "merge"])
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.batch_size, args.max_batches, args.workers)
    else:
        merge()


if __name__ == "__main__":
    main()
