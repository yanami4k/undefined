"""用固定的留出消息评估学生模型，结果做成打标卡片，由人来打分。

    python -m scripts.eval_student --name eval-v0                 # 学生端点（STUDENT_BASE_URL）
    python -m scripts.eval_student --name eval-base --model base  # 同一端点上的另一个模型名

评估集 data/eval/eval_set.jsonl 第一次运行时生成并固定下来，之后每个版本都用同一批消息，好坏比例才能横向比较：
- 种子里的 eval 划分（生成时就留出、从没进过匹配和训练），去掉已删除的「问现编的梗」类别
- 再补几条从没用过的求助类消息（留出集里需要明确答案的太少）
学生只看 STUDENT_SYSTEM + [参考梗]，和训练格式一致。
输出 data/eval/<name>.jsonl，卡片写到 data/label_cards/<name>/，可以直接发到打标页。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace

from meme6657.config import DATA_DIR, endpoint
from meme6657.jobs import read_jsonl, run_parallel, write_jsonl
from meme6657.llm import LLM
from meme6657.prompts import format_user, student_messages
from meme6657.retrieval import Retriever

EVAL_DIR = DATA_DIR / "eval"
EVAL_SET = EVAL_DIR / "eval_set.jsonl"
GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"
DROPPED_CATEGORIES = {"meme_ask_unknown"}
EXTRA_HELP = 8


def build_eval_set(seed: int = 7) -> list[dict]:
    seen = {r["user"] for r in read_jsonl(GOLD_PATH)}
    rows = []
    for name in ["natural", "natural2"]:
        for s in read_jsonl(DATA_DIR / "synth" / f"{name}.jsonl"):
            if s["split"] == "eval" and s["category"] not in DROPPED_CATEGORIES and s["message"] not in seen:
                seen.add(s["message"])
                rows.append({"id": f"{name}-{s['id']}", "category": s["category"], "user": s["message"]})
    # 匹配阶段判为「不接」、但从没生成过回复的求助类消息
    used = {r["user"] for p in (DATA_DIR / "reverse").glob("normal*.jsonl") for r in read_jsonl(p)}
    pool = [r for name in ["round4", "round5b"] for r in read_jsonl(DATA_DIR / "reverse" / f"{name}.jsonl")
            if not r["choice"] and r["category"] in ("help", "tech", "ai_self")
            and r["user"] not in used and r["user"] not in seen]
    random.Random(seed).shuffle(pool)
    for r in pool[:EXTRA_HELP]:
        seen.add(r["user"])
        rows.append({"id": f"pool-{r['category']}-{len(rows)}", "category": r["category"], "user": r["user"]})
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="这次评估的名字，例如 eval-v0")
    ap.add_argument("--model", default="", help="学生端点上的模型名，默认用 STUDENT_MODEL")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not EVAL_SET.exists():
        write_jsonl(EVAL_SET, build_eval_set())
    rows = read_jsonl(EVAL_SET)
    print(f"评估集 {len(rows)} 条（{EVAL_SET}）")

    retriever = Retriever.from_files()
    ep = endpoint("student")
    student = LLM(replace(ep, model=args.model) if args.model else ep)

    def work(r: dict) -> dict:
        user_content = format_user(r["user"], retriever.retrieve(r["user"]))
        reply = student.chat(student_messages([{"role": "user", "content": user_content}]),
                             temperature=args.temperature, max_tokens=400).strip()
        return {**r, "user_content": user_content, "reply": reply}

    out_path = EVAL_DIR / f"{args.name}.jsonl"
    run_parallel(rows, work, out_path, workers=args.workers)

    card_dir = DATA_DIR / "label_cards" / args.name
    card_dir.mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(read_jsonl(out_path), 1):
        card = {"batch": args.name, "order": 9000 + n, "kind": "eval", "source": "", "user": r["user"], "reply": r["reply"]}
        (card_dir / f"{args.name}-{r['id']}.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    print(f"卡片写到 {card_dir}")


if __name__ == "__main__":
    main()
