"""把打标页导出的标注和卡片合并成金标文件。

    python -m scripts.sync_labels --labels data/label_results_v2/labels

卡片在 data/label_cards/<批次目录>/<id>.json（发到打标页的原样内容），
标注由 ArtifactData 导出为 <labels>/<id>.json（{"data": {"verdict", "reasons", "at"}}）。
输出 data/gold/labels.jsonl：每条 {id, batch, kind, source, user, reply, verdict, reasons}。
这是 data/ 下唯一提交到仓库的文件，每次打标合并后一起提交。
合并到已有的金标上：同一张卡以这次导出为准，其余的原样保留、顺序不变，新卡追加在后面。
打标页换过（旧页面的标注只在金标里），所以不能用一次导出整份重写。没打标的卡片不写入。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, write_jsonl

CARDS_DIR = DATA_DIR / "label_cards"
GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"


def load_cards() -> dict[str, dict]:
    cards = {}
    for p in sorted(CARDS_DIR.glob("*/*.json")):
        if p.parent.name.startswith("_"):
            continue
        card = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(card, dict) and "user" in card and "reply" in card:
            cards[p.stem] = card
    return cards


def load_labels(labels_dir: Path) -> dict[str, dict]:
    out = {}
    for p in labels_dir.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = d.get("data", d)
    return out


def merge(gold: list[dict], cards: dict[str, dict], labels: dict[str, dict]) -> tuple[list[dict], int, list[str]]:
    """把这次导出的标注合并进已有金标，返回 (金标, 更新的条数, 有标注但找不到卡片的 id)。"""
    rows = list(gold)
    where = {r["id"]: i for i, r in enumerate(rows)}
    updated, missing = 0, []
    for cid, lab in sorted(labels.items(), key=lambda kv: (cards.get(kv[0], {}).get("order", 0), kv[0])):
        if lab.get("verdict") not in ("good", "bad", "wave"):
            continue
        card = cards.get(cid)
        if card is None:
            missing.append(cid)
            continue
        row = {"id": cid, "batch": card.get("batch", ""), "kind": card.get("kind", ""),
               "source": card.get("source", ""), "user": card["user"], "reply": card["reply"],
               "verdict": lab["verdict"], "reasons": lab.get("reasons", [])}
        if cid in where:
            updated += rows[where[cid]] != row
            rows[where[cid]] = row
        else:
            where[cid] = len(rows)
            rows.append(row)
    return rows, updated, missing


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ArtifactData 导出的 labels 目录")
    args = ap.parse_args()

    gold = read_jsonl(GOLD_PATH)
    rows, updated, missing = merge(gold, load_cards(), load_labels(Path(args.labels)))
    write_jsonl(GOLD_PATH, rows)
    print(f"写入 {len(rows)} 条 → {GOLD_PATH}（原有 {len(gold)}，新增 {len(rows) - len(gold)}，改动 {updated}）")
    for (kind, verdict), n in sorted(Counter((r["kind"], r["verdict"]) for r in rows).items()):
        print(f"  {kind:9s} {verdict:5s} {n}")
    if missing:
        print(f"有标注但找不到卡片的 {len(missing)} 条（未写入）：", missing[:10])


if __name__ == "__main__":
    main()
