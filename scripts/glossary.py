"""圈内黑话待确认清单：机器找词，人来解释。

    python -m scripts.glossary list    # 生成 kb/glossary.todo.md
    # 在表格「含义」一列填上解释（不确定的留空），「变体」列可填其他写法，用 / 分隔
    python -m scripts.glossary merge   # 把填了含义的词并入 kb/aliases.json（作为 term 条目）

LLM 只负责找出「外人看不懂的词」，不负责解释：它对这些梗的理解不可靠，
解释必须由懂梗的人来写，否则模型会学会一本正经地编。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter

from meme6657.config import ALIASES_PATH, KB_DIR
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.retrieval import load_aliases, load_memes, normalize
from scripts.build_aliases import load_working_aliases

TODO_PATH = KB_DIR / "glossary.todo.md"
BATCH_OUT = KB_DIR / ".glossary_batches.jsonl"

PROMPT = """下面是斗鱼玩机器（6657）直播间的一批弹幕。
请列出其中「圈外人看不懂」的词：圈内黑话、外号、谐音、名场面代称、固定梗词。
不要解释，只列词本身（2~10 个字，或一个英文词/emoji 组合）。不要列普通词和单纯的脏话。

输出 JSON：{{"terms": ["...", "..."]}}

弹幕：
{memes}"""

# 圈外人也懂的普通词，LLM 偶尔会列进来
COMMON_WORDS = {"弹幕", "主播", "直播间", "决赛", "major", "比赛", "选手", "战队", "冠军", "亚军", "加时", "狙击手",
                "cs", "csgo", "cs2", "mvp", "rating", "top", "bro", "ok"}


def _is_emoji(t: str) -> bool:
    return any(ord(ch) > 0x2000 and not ("一" <= ch <= "鿿") for ch in t)


ROW_RE = re.compile(r"^\|\s*(?P<term>[^|]+?)\s*\|\s*(?P<count>\d+)\s*\|(?P<examples>[^|]*)\|(?P<meaning>[^|]*)\|(?P<variants>[^|]*)\|\s*$")


def list_terms(batch_size: int, max_batches: int, top: int, workers: int) -> None:
    memes = load_memes()
    rng = random.Random(1)
    sample = sorted(memes, key=lambda m: -m.cnt)[: batch_size * max_batches * 2]
    rng.shuffle(sample)
    batches = [
        {"id": f"g{b}", "lines": "\n".join(f"- {m.text[:150]}" for m in sample[b * batch_size : (b + 1) * batch_size])}
        for b in range(max_batches)
    ]
    llm = LLM("usersim")

    def work(batch: dict) -> dict:
        data = llm.chat_json([{"role": "user", "content": PROMPT.format(memes=batch["lines"])}],
                             max_tokens=2500, task="aliases")
        return {"id": batch["id"], "terms": [str(t).strip() for t in data.get("terms", []) if str(t).strip()]}

    run_parallel(batches, work, BATCH_OUT, workers=workers)

    known = {t.lower() for e in load_aliases(include_candidates=True) for t in e.terms}
    proposed: dict[str, str] = {}  # 小写 → 首次出现的写法，大小写变体只保留一条
    for row in read_jsonl(BATCH_OUT):
        for t in row["terms"]:
            k = t.lower()
            if k in known or k in COMMON_WORDS or k in proposed:
                continue
            # 单个字母 / 单个汉字子串匹配会命中大量无关弹幕，emoji 除外
            if len(normalize(t)) < 2 and not _is_emoji(t):
                continue
            proposed[k] = t
    lowered = [m.text.lower() for m in memes]
    stats = []
    for term in proposed.values():
        if re.fullmatch(r"[A-Za-z0-9 ]+", term):
            # 英文词按整词匹配，避免 op 命中 top
            pat = re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")
            hits = [i for i, low in enumerate(lowered) if pat.search(low)]
        else:
            hits = [i for i, low in enumerate(lowered) if term.lower() in low]
        if len(hits) >= 2:
            examples = sorted(hits, key=lambda i: -memes[i].cnt)[:2]
            stats.append((term, len(hits), [memes[i].text.replace("|", "｜").replace("\n", " ")[:60] for i in examples]))
    stats.sort(key=lambda s: -s[1])

    lines = [
        "# 圈内黑话待确认清单",
        "",
        "在「含义」列填写解释（一两句话：指什么、怎么来的）。不确定就留空，留空的词不会被合并。",
        "「变体」列可填其他写法，用 / 分隔。填完运行 `python -m scripts.glossary merge`。",
        "",
        "| 词 | 出现次数 | 例句 | 含义 | 变体 |",
        "|---|---|---|---|---|",
    ]
    for term, n, ex in stats[:top]:
        lines.append(f"| {term} | {n} | {' ／ '.join(ex)} |  |  |")
    TODO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{min(top, len(stats))} 个词 → {TODO_PATH}")


def merge() -> None:
    if not TODO_PATH.exists():
        sys.exit("没有 kb/glossary.todo.md，先运行 list")
    entries = load_working_aliases()
    by_name = {e["canonical"].lower(): e for e in entries}
    added = updated = 0
    for line in TODO_PATH.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m or m["term"] in ("词", "---") or not m["meaning"].strip():
            continue
        term, meaning = m["term"].strip(), m["meaning"].strip()
        variants = [v.strip() for v in m["variants"].split("/") if v.strip()]
        cur = by_name.get(term.lower())
        if cur:
            cur["note"] = meaning
            cur["aliases"] += [v for v in variants if v not in cur["aliases"]]
            updated += 1
        else:
            entry = {"canonical": term, "aliases": variants, "kind": "term", "note": meaning,
                     "tags": [], "status": "verified"}
            entries.append(entry)
            by_name[term.lower()] = entry
            added += 1
    ALIASES_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"新增 {added}，更新 {updated} → {ALIASES_PATH}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "merge"])
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--max-batches", type=int, default=15)
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "list":
        list_terms(args.batch_size, args.max_batches, args.top, args.workers)
    else:
        merge()


if __name__ == "__main__":
    main()
