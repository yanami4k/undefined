"""把打标页导出的标注和卡片合并成金标文件。

    python -m scripts.sync_labels --labels data/label_cards/_labels_final/labels

卡片在 data/label_cards/<批次目录>/<id>.json（发到打标页的原样内容），
标注由 ArtifactData 导出为 <labels>/<id>.json（{"data": {"verdict", "reasons", "at"}}）。
输出 data/gold/labels.jsonl：每条 {id, batch, kind, source, user, reply, verdict, reasons}。
这是 data/ 下唯一提交到仓库的文件，每次打标合并后一起提交。
没打标的卡片不写入；已经从打标页删掉、但留有标注的卡片照样保留。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from meme6657.config import DATA_DIR
from meme6657.jobs import write_jsonl

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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ArtifactData 导出的 labels 目录")
    args = ap.parse_args()

    cards, labels = load_cards(), load_labels(Path(args.labels))
    rows, missing = [], []
    for cid, lab in sorted(labels.items(), key=lambda kv: cards.get(kv[0], {}).get("order", 0)):
        if lab.get("verdict") not in ("good", "bad", "wave"):
            continue
        card = cards.get(cid)
        if card is None:
            missing.append(cid)
            continue
        rows.append({"id": cid, "batch": card.get("batch", ""), "kind": card.get("kind", ""),
                     "source": card.get("source", ""), "user": card["user"], "reply": card["reply"],
                     "verdict": lab["verdict"], "reasons": lab.get("reasons", [])})
    write_jsonl(GOLD_PATH, rows)
    print(f"写入 {len(rows)} 条 → {GOLD_PATH}")
    for (kind, verdict), n in sorted(Counter((r["kind"], r["verdict"]) for r in rows).items()):
        print(f"  {kind:9s} {verdict:5s} {n}")
    if missing:
        print(f"有标注但找不到卡片的 {len(missing)} 条（未写入）：", missing[:10])


if __name__ == "__main__":
    main()
