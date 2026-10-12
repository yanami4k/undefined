"""梗考据：给一个词，收集证据包，可选让模型按考据范例推断含义。

    python -m scripts.investigate 波哈 宗宗              # 只收集证据，打印摘要
    python -m scripts.investigate 波哈 --hypothesize     # 再让老师模型给出带依据和把握程度的假设

证据来自 sb6657 站内搜索（全量梗库，低速请求并缓存）：
- 最早出现的几条：起源语境（注意 00:00:00 的时间多为早期批量导入，不精确）
- 复制最多的几条：最流行的用法
- 按月出现次数：爆发时间点，可以对照那段时间的比赛 / 直播事件
- 标签分布、同时出现的外号：指向谁

推断结果写入 kb/glossary.candidates.json（status=candidate），必须人工审核后才能并入知识库。
推理方式由 kb/decode_examples.md 里的考据范例来教。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from meme6657.config import DATA_DIR, KB_DIR
from meme6657.llm import LLM
from meme6657.retrieval import load_aliases

API = "https://api.hguofichp.cn/machine/pageSearch"
USER_AGENT = "sb6657-style-prototype/0.1 (personal research; low-rate)"
CACHE_DIR = DATA_DIR / "investigate"
EXAMPLES_PATH = KB_DIR / "decode_examples.md"
CANDIDATES_PATH = KB_DIR / "glossary.candidates.json"
DELAY = 4.0


def _search(term: str, sort: int, page: int, size: int = 50) -> dict:
    body = json.dumps({"barrage": term, "sort": sort, "pageSize": size, "pageNum": page}).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8-sig"))
    if str(data.get("code")) != "200":
        raise RuntimeError(f"搜索失败: {data.get('msg')}")
    return data["data"]


def fetch_all(term: str, max_items: int = 300) -> list[dict]:
    """取某个词在全量梗库里的全部出现记录（最多 max_items 条），结果缓存到 data/investigate/。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{term}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows, page = [], 1
    while len(rows) < max_items:
        data = _search(term, sort=0, page=page)
        batch = data.get("list") or []
        rows.extend(batch)
        if not batch or data.get("lastPage") or len(rows) >= (data.get("total") or 0):
            break
        page += 1
        time.sleep(DELAY)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    time.sleep(DELAY)
    return rows


def evidence(term: str, rows: list[dict], tag_names: dict[str, str]) -> dict:
    def brief(r):
        t = r["submitTime"]
        imported = t.endswith("T00:00:00")
        return {"time": t[:10] + ("（早期导入，时间不精确）" if imported else ""), "cnt": int(r.get("cnt") or 0),
                "text": r["barrage"].replace("\n", " ")[:120]}

    by_time = sorted(rows, key=lambda r: r["submitTime"])
    by_cnt = sorted(rows, key=lambda r: -int(r.get("cnt") or 0))
    months = Counter(r["submitTime"][:7] for r in rows)
    tags = Counter(tag_names.get(t, t) for r in rows for t in (r.get("tags") or "").split(",") if t)
    co = Counter()
    for e in load_aliases():
        if term.lower() in (x.lower() for x in e.terms):
            continue
        n = sum(any(x.lower() in r["barrage"].lower() for x in e.terms) for r in rows)
        if n:
            co[e.canonical] = n
    return {
        "term": term,
        "total": len(rows),
        "earliest": [brief(r) for r in by_time[:8]],
        "most_copied": [brief(r) for r in by_cnt[:8]],
        "latest": [brief(r) for r in by_time[-5:]],
        "by_month": dict(sorted(months.items())),
        "tags": dict(tags.most_common(8)),
        "co_occurring": dict(co.most_common(8)),
    }


def render(ev: dict) -> str:
    out = [f"## {ev['term']}（共 {ev['total']} 条）"]
    for key, label in [("earliest", "最早出现"), ("most_copied", "复制最多"), ("latest", "最近")]:
        out.append(f"\n{label}：")
        out += [f"- {r['time']}｜复制 {r['cnt']}｜{r['text']}" for r in ev[key]]
    peak = sorted(ev["by_month"].items(), key=lambda kv: -kv[1])[:5]
    out.append("\n出现最多的月份：" + "，".join(f"{m}（{n}）" for m, n in peak))
    out.append("标签：" + "，".join(f"{k}（{v}）" for k, v in ev["tags"].items()))
    if ev["co_occurring"]:
        out.append("同时出现的外号：" + "，".join(f"{k}（{v}）" for k, v in ev["co_occurring"].items()))
    return "\n".join(out)


HYPOTHESIZE_PROMPT = """你在考据一个直播间（斗鱼玩机器 6657，CS 主播）的弹幕梗。
下面先给出考据范例，示范怎么从证据推理；然后是待考据词的证据包。

推理要求：
- 先看最早出现的弹幕判断起源语境，再看复制最多的弹幕判断主流用法，再看爆发月份和同时出现的人物
- 区分「证据直接支持」和「推测」；证据不够就说不够，不要编
- 这个直播间的梗普遍是「解构 + 移植」：同一个形式被搬到不同语境，注意区分原始含义和后来的移植用法

{examples}

---

待考据：
{evidence}

输出 JSON：
{{"meaning": "现在的主流含义（一两句话）",
  "origin": "起源推测（一两句话）",
  "confidence": "high / medium / low",
  "evidence": ["支持结论的具体弹幕或数据"],
  "open_questions": ["还不确定、需要人确认的点"],
  "search_queries": ["建议用来上网核实的搜索词"]}}"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--hypothesize", action="store_true", help="让老师模型按考据范例推断含义")
    ap.add_argument("--max-items", type=int, default=300)
    args = ap.parse_args()

    tag_path = DATA_DIR / "raw" / "tags.json"
    tag_names = json.loads(tag_path.read_text(encoding="utf-8")) if tag_path.exists() else {}
    examples = EXAMPLES_PATH.read_text(encoding="utf-8") if EXAMPLES_PATH.exists() else "（暂无范例）"
    cands = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8")) if CANDIDATES_PATH.exists() else []

    for term in args.terms:
        rows = fetch_all(term, args.max_items)
        if not rows:
            print(f"## {term}：站内没有搜到\n")
            continue
        ev = evidence(term, rows, tag_names)
        text = render(ev)
        print(text + "\n")
        if not args.hypothesize:
            continue
        hyp = LLM("teacher").chat_json(
            [{"role": "user", "content": HYPOTHESIZE_PROMPT.format(examples=examples, evidence=text)}],
            max_tokens=1500, task="hypothesis",
        )
        print("推断：" + json.dumps(hyp, ensure_ascii=False, indent=2) + "\n")
        cands = [c for c in cands if c.get("canonical") != term]
        cands.append({"canonical": term, "aliases": [], "kind": "term", "note": hyp.get("meaning", ""),
                      "origin": hyp.get("origin", ""), "confidence": hyp.get("confidence", "low"),
                      "evidence": hyp.get("evidence", []), "open_questions": hyp.get("open_questions", []),
                      "search_queries": hyp.get("search_queries", []), "status": "candidate"})
    if args.hypothesize:
        CANDIDATES_PATH.write_text(json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"候选已写入 {CANDIDATES_PATH}，审核后把 status 改为 verified 再合并")


if __name__ == "__main__":
    main()
