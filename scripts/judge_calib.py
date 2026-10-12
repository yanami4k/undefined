"""评委校准：把人工打标当作评委的参考例子，看它能不能预测人对没见过的样本的判定。

    python -m scripts.judge_calib                      # 默认用老师模型（deepseek-v4-pro）
    python -m scripts.judge_calib --role judge         # 换成便宜的 flash 对比

金标按用户消息稳定划分：约 80% 作为例子放进提示词前缀（顺序固定，可命中缓存），其余 20% 让评委判。
每次判 --batch 条，输出 好 / 电波 / 坏 和 1~5 的把握。
报告三分类一致率，以及「保留」（好或电波）的精确率和召回率——用评委预筛时最关心精确率：
它说保留的，人有多大比例也会保留。把握分档看精确率，决定能不能只收高把握的样本。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel, stable_hash
from meme6657.llm import LLM
from meme6657.prompts import load_taste

GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"
OUT_DIR = DATA_DIR / "judge"
VERDICT_ZH = {"good": "好", "wave": "电波", "bad": "坏"}
ZH_VERDICT = {v: k for k, v in VERDICT_ZH.items()}
KIND_ZH = {"meme": "整条甩梗", "template": "套句式", "normal": "自己开口"}

PREFIX = """{taste}

---

下面是懂行的人亲手打过的标注，是判断好坏的唯一标准。每条格式：[回复类型] 用户 → 回复 ⇒ 判定（坏的原因）。
- 好：短冷准、贴情境、懂梗义；电波：荒诞但贴着当下、大加分；坏：仿造梗感、不懂梗义、通用段子、硬扔无关、照抄套路、太长加戏，或者用户消息本身不自然
- 「整条甩梗」是整条原样甩一句弹幕；「自己开口」是祥子用自己的话回复

{examples}"""

SUFFIX = """---

请完全站在这个人的角度，按上面的标注口味，判断下面每一条。不要按「一般意义上的好回复」判，要按他的口味：
{items}

输出 JSON：{{"results": [{{"i": 编号, "verdict": "好 / 电波 / 坏", "confidence": 1到5, "why": "一句话"}}]}}，每条都要有。"""


def render(r: dict, reply_chars: int = 90) -> str:
    reply = r["reply"].replace("\n", " ")
    if len(reply) > reply_chars:
        reply = reply[:reply_chars] + "…"
    return f"[{KIND_ZH.get(r['kind'], r['kind'])}] 用户：{r['user']} → 回复：{reply}"


def split(gold: list[dict], holdout: float) -> tuple[list[dict], list[dict]]:
    """按用户消息划分，同一条消息的不同回复不会一边当例子一边被考。"""
    test = [r for r in gold if stable_hash(r["user"] + "#calib") < holdout]
    train = [r for r in gold if stable_hash(r["user"] + "#calib") >= holdout]
    return train, test


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="teacher", help="teacher（v4-pro）或 judge（flash）")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    gold = read_jsonl(GOLD_PATH)
    train, test = split(gold, args.holdout)
    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(test)
    examples = "\n".join(
        f"{render(r)} ⇒ {VERDICT_ZH[r['verdict']]}" + (f"（{'、'.join(r['reasons'])}）" if r.get("reasons") else "")
        for r in train)
    prefix = PREFIX.format(taste=load_taste(), examples=examples)
    print(f"例子 {len(train)} 条，考题 {len(test)} 条；考题里人判保留的比例 "
          f"{sum(r['verdict'] != 'bad' for r in test) / len(test):.0%}")

    llm = LLM(args.role)
    batches = [{"id": f"b{i}", "rows": test[i : i + args.batch]} for i in range(0, len(test), args.batch)]

    def work(b: dict) -> dict:
        items = "\n".join(f"{i}. {render(r, 200)}" for i, r in enumerate(b["rows"], 1))
        data = llm.chat_json([{"role": "user", "content": prefix + "\n\n" + SUFFIX.format(items=items)}],
                             temperature=0.2, max_tokens=1500, task="judge")
        by_i = {int(x.get("i", 0)): x for x in data.get("results", [])}
        out = []
        for i, r in enumerate(b["rows"], 1):
            x = by_i.get(i, {})
            out.append({"id": r["id"], "kind": r["kind"], "user": r["user"], "reply": r["reply"],
                        "human": r["verdict"], "pred": ZH_VERDICT.get(str(x.get("verdict", "")).strip(), ""),
                        "confidence": int(x.get("confidence") or 0), "why": x.get("why", "")})
        return {"id": b["id"], "results": out}

    out_path = OUT_DIR / f"calib-{args.role}.jsonl"
    run_parallel(batches, work, out_path, workers=args.workers)

    rows = [x for b in read_jsonl(out_path) for x in b["results"] if x["pred"]]
    keep = lambda v: v in ("good", "wave")  # noqa: E731
    acc = sum(x["pred"] == x["human"] for x in rows) / len(rows)
    agree2 = sum(keep(x["pred"]) == keep(x["human"]) for x in rows) / len(rows)
    print(f"\n有效判定 {len(rows)} 条；三分类一致率 {acc:.0%}；保留 / 不保留一致率 {agree2:.0%}")
    print("混淆（行=人，列=评委）：")
    for h in ("good", "wave", "bad"):
        c = Counter(x["pred"] for x in rows if x["human"] == h)
        print(f"  人={VERDICT_ZH[h]:2s} " + "  ".join(f"{VERDICT_ZH[p]}:{c[p]:3d}" for p in ("good", "wave", "bad")))
    print("按把握分档，评委判保留的精确率（人也保留的比例）：")
    for lo in (1, 3, 4, 5):
        sel = [x for x in rows if keep(x["pred"]) and x["confidence"] >= lo]
        if sel:
            prec = sum(keep(x["human"]) for x in sel) / len(sel)
            rec = len([x for x in sel if keep(x["human"])]) / max(1, sum(keep(x["human"]) for x in rows))
            print(f"  把握≥{lo}：{len(sel):3d} 条，精确率 {prec:.0%}，召回率 {rec:.0%}")
    for kind in ("meme", "template", "normal"):
        sub = [x for x in rows if x["kind"] == kind]
        if sub:
            sel = [x for x in sub if keep(x["pred"])]
            prec = sum(keep(x["human"]) for x in sel) / max(1, len(sel))
            print(f"  {KIND_ZH[kind]}：{len(sub)} 条，人保留 {sum(keep(x['human']) for x in sub)}，"
                  f"评委保留 {len(sel)}，精确率 {prec:.0%}")


if __name__ == "__main__":
    main()
