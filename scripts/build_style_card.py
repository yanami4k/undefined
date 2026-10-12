"""用 LLM 从梗库归纳风格，生成风格卡草稿。

    python -m scripts.build_style_card

输出 kb/style_card.generated.md，不会覆盖人工维护的 kb/style_card.md。
对比两份文件，把有价值的新观察手工合并进 style_card.md。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

from meme6657.config import KB_DIR
from meme6657.llm import LLM
from meme6657.retrieval import load_blocklist, load_memes

OUT_PATH = KB_DIR / "style_card.generated.md"

OBSERVE_PROMPT = """下面是斗鱼玩机器（6657）直播间的一批高频弹幕。请从「说话风格」角度做观察笔记，不要逐条解释：
1. 常见句式模板（抽象成「X 怎么你们了？是 A 了还是 B 了？」这样的形式）
2. 语气手法（阴阳、复读、模仿解说、饭圈腔等），各举一个改写后的短例
3. emoji 的用法和含义
4. 称呼和外号
5. 哪些内容是低俗 / 冒犯的，模仿时应该避开

弹幕：
{memes}"""

MERGE_PROMPT = """下面是对同一个直播间弹幕风格的多份观察笔记。请合并去重，写成一份给 AI 看的「风格卡」，
用 Markdown，包含这些小节：人设、抖梗的分寸、语气与常见手法（表格：手法 / 说明 / 改写后的短例）、
emoji 语义、称呼、红线、回复形态。

要求：例子用改写过的短句，不要大段照抄原弹幕；红线一节要明确禁止色情、歧视、攻击用户。

笔记：
{notes}"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tag", type=int, default=40, help="每个标签抽样条数")
    ap.add_argument("--chunk", type=int, default=120)
    args = ap.parse_args()

    blocklist = load_blocklist()
    memes = [m for m in load_memes() if not any(b in m.text.lower() for b in blocklist)]
    by_tag = defaultdict(list)
    for m in memes:
        for t in m.tags or ["无标签"]:
            by_tag[t].append(m)
    rng = random.Random(0)
    picked, seen = [], set()
    for tag, group in by_tag.items():
        group = sorted(group, key=lambda m: -m.cnt)[: args.per_tag * 3]
        for m in rng.sample(group, min(args.per_tag, len(group))):
            if m.id not in seen:
                seen.add(m.id)
                picked.append(m)
    rng.shuffle(picked)
    print(f"抽样 {len(picked)} 条，分 {(len(picked) + args.chunk - 1) // args.chunk} 批观察")

    llm = LLM("teacher")
    notes = []
    for i in range(0, len(picked), args.chunk):
        lines = "\n".join(f"- {m.text[:150]}" for m in picked[i : i + args.chunk])
        notes.append(llm.chat([{"role": "user", "content": OBSERVE_PROMPT.format(memes=lines)}],
                              temperature=0.3, max_tokens=2048, task="style_card"))
        print(f"  第 {len(notes)} 批完成")
    card = llm.chat([{"role": "user", "content": MERGE_PROMPT.format(notes="\n\n---\n\n".join(notes))}],
                    temperature=0.3, max_tokens=4096, task="style_card")
    OUT_PATH.write_text(card, encoding="utf-8")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
