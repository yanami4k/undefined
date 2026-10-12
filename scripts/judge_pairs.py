"""配对评委：按口味校准文件给「用户说 → 它接」打 好 / 不好 / 电波，并和人工打标对比一致率。

    python -m scripts.judge_pairs predict data/reverse/round3.jsonl --out data/judge/round3.jsonl
    python -m scripts.judge_pairs predict data/gold/labels.jsonl --out data/judge/gold.jsonl
    python -m scripts.judge_pairs agree data/judge/gold.jsonl data/gold/labels.jsonl

判定类别和不好的原因与打标页面完全一致，结果可以直接对比。
一致率够高时，就让评委预筛候选，只把它判为好 / 电波的交给人看。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import format_context, load_taste
from meme6657.retrieval import Context, Retriever

REASONS = ["仿造梗感", "不懂梗义", "通用段子", "硬扔无关", "照抄套路", "太长加戏"]

PROMPT = """{taste}

---

以上是懂行的人的口味标准和他亲手评过的例子。请站在他的角度，评判下面这个接法。

用户说：{user}
它接：{reply}
{source_line}
{notes}

判定只有三种：
- good：对胃
- wave：电波加分（原样的梗意外地正好撞上情境，荒诞又贴，或者撞上 AI 自身处境）
- bad：不对胃。原因从这些里选（可多选）：{reasons}

输出 JSON：{{"verdict": "good|wave|bad", "reasons": [], "why": "一句话"}}"""


def pairs_from(path: Path) -> list[dict]:
    """支持两种输入：反推结果（每行一个来源 + 多个 pairs）和金标（每行一对）。"""
    items = []
    for i, r in enumerate(read_jsonl(path)):
        if "pairs" in r:
            for k, p in enumerate(r["pairs"]):
                items.append({"id": f"{path.stem}-{i:02d}-{k}", "kind": r["kind"], "source": r["source"],
                              "user": p.get("user", ""), "reply": p.get("reply", "")})
        else:
            items.append({"id": r["id"], "kind": r.get("kind", "meme"), "source": r.get("source", ""),
                          "user": r["user"], "reply": r["reply"]})
    return items


def predict(src: Path, out: Path, workers: int) -> None:
    retriever = Retriever.from_files()
    taste = load_taste()
    judge = LLM("teacher")

    def work(it: dict) -> dict:
        notes = format_context(Context(aliases=retriever.match_aliases(it["source"] + it["user"])))
        label = "原句式" if it["kind"] == "template" else "原弹幕"
        prompt = PROMPT.format(taste=taste, user=it["user"], reply=it["reply"],
                               source_line=f"{label}：{it['source']}" if it["source"] else "",
                               notes=f"相关知识：\n{notes}" if notes else "", reasons="、".join(REASONS))
        data = judge.chat_json([{"role": "user", "content": prompt}], max_tokens=300, task="pairwise")
        verdict = data.get("verdict", "bad")
        if verdict not in ("good", "wave", "bad"):
            verdict = "bad"
        return {**it, "verdict": verdict, "reasons": [r for r in data.get("reasons", []) if r in REASONS],
                "why": data.get("why", "")}

    run_parallel(pairs_from(src), work, out, workers=workers)


def agree(pred_path: Path, gold_path: Path, exclude_taste: bool = True) -> None:
    pred = {r["id"]: r for r in read_jsonl(pred_path)}
    gold = {r["id"]: r for r in read_jsonl(gold_path)}
    taste = load_taste() if exclude_taste else ""
    ids = [i for i in gold if i in pred and not (taste and gold[i]["reply"][:20] in taste)]
    if not ids:
        sys.exit("没有可对比的样本")
    exact = sum(pred[i]["verdict"] == gold[i]["verdict"] for i in ids)
    passed = lambda v: v in ("good", "wave")
    binary = sum(passed(pred[i]["verdict"]) == passed(gold[i]["verdict"]) for i in ids)
    conf = Counter((gold[i]["verdict"], pred[i]["verdict"]) for i in ids)
    print(f"对比 {len(ids)} 条（已排除口味校准文件里引用过的例子）")
    print(f"三分类一致率：{exact / len(ids):.0%}")
    print(f"过关 / 不过关一致率：{binary / len(ids):.0%}")
    print("混淆（人工 → 评委）：")
    for g in ("good", "wave", "bad"):
        print(f"  {g:5s} → " + "  ".join(f"{p}:{conf[(g, p)]}" for p in ("good", "wave", "bad")))
    # 预筛的价值：评委判过关的样本里，人工也判过关的比例
    kept = [i for i in ids if passed(pred[i]["verdict"])]
    if kept:
        prec = sum(passed(gold[i]["verdict"]) for i in kept) / len(kept)
        rec = sum(passed(gold[i]["verdict"]) for i in kept) / max(1, sum(passed(gold[i]["verdict"]) for i in ids))
        print(f"预筛效果：评委放行 {len(kept)} 条，其中人工也认可 {prec:.0%}；人工认可的里被评委放行 {rec:.0%}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("predict")
    p1.add_argument("src")
    p1.add_argument("--out", required=True)
    p1.add_argument("--workers", type=int, default=8)
    p2 = sub.add_parser("agree")
    p2.add_argument("pred")
    p2.add_argument("gold")
    args = ap.parse_args()
    if args.cmd == "predict":
        predict(Path(args.src), Path(args.out), args.workers)
    else:
        agree(Path(args.pred), Path(args.gold))


if __name__ == "__main__":
    main()
