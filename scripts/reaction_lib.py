"""反应库：从人工打标里整理「每条梗在什么情境下好用、在什么情境下翻车」。

    python -m scripts.reaction_lib

匹配时按情境配梗而不是按关键词：每个候选梗都带着它真实的成功用例和翻车用例，
老师据此判断当前用户消息的情境是否对得上。
输出 data/gold/reactions.json。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl

GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"
OUT_PATH = DATA_DIR / "gold" / "reactions.json"


def build(min_rate: float = 0.4, min_count_for_drop: int = 3) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for g in read_jsonl(GOLD_PATH):
        if g["kind"] == "meme":
            by_source[g["source"]].append(g)
    lib = []
    for source, gs in by_source.items():
        ok = [g for g in gs if g["verdict"] in ("good", "wave")]
        bad = [g for g in gs if g["verdict"] == "bad"]
        if not ok:
            continue
        rate = len(ok) / len(gs)
        if len(gs) >= min_count_for_drop and rate < min_rate:
            continue
        ok.sort(key=lambda g: g["verdict"] != "wave")  # 电波排前面
        lib.append({
            "text": source,
            "worked": [{"user": g["user"], "reply": g["reply"], "wave": g["verdict"] == "wave"} for g in ok[:3]],
            "failed": [{"user": g["user"], "reasons": g["reasons"]} for g in bad[:2]],
            "n_ok": len(ok), "n_wave": sum(g["verdict"] == "wave" for g in ok), "n_bad": len(bad),
        })
    lib.sort(key=lambda e: (-e["n_wave"], -e["n_ok"]))
    return lib


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    lib = build()
    OUT_PATH.write_text(json.dumps(lib, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"反应库 {len(lib)} 条梗（其中出过电波的 {sum(e['n_wave'] > 0 for e in lib)} 条）→ {OUT_PATH}")


if __name__ == "__main__":
    main()
