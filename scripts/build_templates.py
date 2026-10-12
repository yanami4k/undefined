"""可移植模板库：从梗库里抽取「固定形式 + 可变槽位」，给老师模型当移植素材。

    python -m scripts.build_templates

1. 逐批让模型识别每条梗的模板和手法（便宜模型）
2. 按模板骨架聚合，统计变体数、跨了几个领域
3. 让老师模型合并近似模板，挑出最适合移植到日常聊天的若干个，附一个移植示范

输出 kb/templates.json（可人工删改）和 kb/templates.md（自动拼进老师的系统提示词）。
学生模型不看这份目录，而是从老师的输出里学会移植手法。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

from meme6657.config import KB_DIR, TEMPLATES_JSON, TEMPLATES_MD
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM, parse_json
from meme6657.retrieval import load_blocklist, load_memes

BATCH_OUT = KB_DIR / ".templates_batches.jsonl"
RAW_OUT = KB_DIR / ".templates_raw.txt"
BANNED_PATH = KB_DIR / "templates_banned.json"  # 人工判定为不好用的句式，重建时跳过
MOVES = ["模板移植", "语域错置", "反讽反转", "反向辩护", "语义场扩张", "元梗", "复读变异", "身份扮演"]
SITUATIONS = ["比赛输赢", "崩溃破防", "吐槽抱怨", "夸赞起哄", "日常闲聊", "学习工作", "二次元", "主播直播间"]

EXTRACT_PROMPT = """下面是一批直播间弹幕。很多弹幕是「眼熟的固定形式 + 换进去的新内容」。
对每条**有可复用形式**的弹幕，抽出它的模板：固定部分原样保留，可替换部分用 {{槽位名}} 表示。
没有可复用形式的（纯复读某个词、纯骂人、纯人名堆砌）跳过。

手法从这些里选一个：{moves}
domain 是这条弹幕当前套用的领域：CS比赛 / 主播 / 木柜子二次元 / 体育 / 游戏 / 生活 / 其他

输出 JSON：{{"items": [{{"id": 编号, "template": "...", "move": "...", "domain": "..."}}]}}

弹幕：
{memes}"""

SELECT_PROMPT = """下面是从直播间弹幕里抽出的模板候选（已按骨架聚合，附变体数、覆盖领域和原弹幕例子）。
请合并意思相同的模板，挑出**最多 {top} 个**最适合「移植到日常聊天」的：形式眼熟、换个语境依然好笑、不依赖低俗内容。

每个模板输出：
- name：简短名字
- template：模板，槽位用 {{槽位名}}
- move：手法（{moves} 之一）
- examples：1~2 条原弹幕例子，尽量来自不同领域，每条不超过 60 字（可截取）
- transplant：把这个模板移植到一个日常聊天场景的示范（你自己写，一句话）

不要选：纯复读某个词、纯 emoji 堆砌、需要替换一整首歌词的；性化动漫角色的不要选。
- situations：适用场景，从 {situations} 中选 1~3 个

输出 JSON：{{"templates": [...]}}

候选：
{candidates}"""


def skeleton(template: str) -> str:
    """模板骨架：去掉槽位内容、标点、emoji、空白，用于聚合同一模板的不同写法。"""
    s = re.sub(r"\{[^}]*\}", "#", template.lower())
    return "".join(re.findall(r"[#0-9a-z一-鿿]+", s))


def render_md(templates: list[dict]) -> str:
    lines = ["## 可移植模板", "", "下面是这个直播间常见的梗模板。聊天时觉得合适，可以把其中一个的形式套进当前话题；"
             "不要照搬例子原句，不要每次都用。", ""]
    for t in templates:
        lines.append(f"- **{t['name']}**（{t.get('move', '')}；适用：{'、'.join(t.get('situations', []))}）")
        lines.append(f"  模板：{t['template']}")
        if t.get("bound_to"):
            lines.append(f"  强绑定：{t['bound_to']}。原样用，不要拆开替换")
        for ex in t.get("examples", [])[:2]:
            lines.append(f"  原例：{ex}")
        if t.get("transplant"):
            lines.append(f"  移植示范：{t['transplant']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2000, help="按复制数取前 N 条梗")
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--candidates", type=int, default=150, help="交给老师模型筛选的候选数")
    ap.add_argument("--top", type=int, default=36)
    ap.add_argument("--chunk", type=int, default=50, help="每次交给老师筛选的候选数")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    blocklist = load_blocklist()
    memes = [m for m in load_memes() if len(m.text) <= 150 and not any(b in m.text.lower() for b in blocklist)]
    memes = sorted(memes, key=lambda m: -m.cnt)[: args.pool]
    by_id = {m.id: m for m in memes}
    batches = [
        {"id": f"t{i // args.batch_size}", "memes": [m.id for m in memes[i : i + args.batch_size]]}
        for i in range(0, len(memes), args.batch_size)
    ]

    extractor = LLM("usersim")

    def work(batch: dict) -> dict:
        lines = "\n".join(f"{mid}. {by_id[mid].text}" for mid in batch["memes"])
        data = extractor.chat_json(
            [{"role": "user", "content": EXTRACT_PROMPT.format(moves="、".join(MOVES), memes=lines)}],
            max_tokens=4000, task="templates",
        )
        return {"id": batch["id"], "items": data.get("items", [])}

    run_parallel(batches, work, BATCH_OUT, workers=args.workers)

    groups: dict[str, dict] = defaultdict(lambda: {"templates": [], "domains": set(), "examples": [], "cnt": 0})
    for row in read_jsonl(BATCH_OUT):
        for it in row["items"]:
            mid, tpl = it.get("id"), str(it.get("template", "")).strip()
            if mid not in by_id or "{" not in tpl:
                continue
            key = skeleton(tpl)
            # 固定部分太少（纯复读、纯堆 emoji）或槽位太多（歌词替换）的不算可移植模板
            fixed = key.replace("#", "")
            if len(fixed) < 4 or tpl.count("{") > 4:
                continue
            g = groups[key]
            g["templates"].append(tpl)
            g["domains"].add(it.get("domain", "其他"))
            g["examples"].append(by_id[mid].text)
            g["cnt"] += by_id[mid].cnt
            g["move"] = it.get("move", "")
    ranked = sorted(groups.values(), key=lambda g: (-len(g["domains"]), -len(g["examples"]), -g["cnt"]))
    print(f"聚合出 {len(groups)} 个模板骨架，取前 {args.candidates} 个交给老师筛选")

    cand_lines = []
    for i, g in enumerate(ranked[: args.candidates], 1):
        exs = " ／ ".join(e[:80] for e in g["examples"][:3])
        cand_lines.append(f"{i}. {g['templates'][0]}（{g.get('move', '')}；变体 {len(g['examples'])}；"
                          f"领域 {'、'.join(sorted(g['domains']))}）例：{exs}")

    # 分批挑选：一次输出太多会被截断
    teacher = LLM("teacher")
    n_chunks = max(1, (len(cand_lines) + args.chunk - 1) // args.chunk)
    per_chunk = max(1, (args.top + n_chunks - 1) // n_chunks)
    templates, seen, raws = [], set(), []
    banned = {skeleton(t) for t in json.loads(BANNED_PATH.read_text(encoding="utf-8"))} if BANNED_PATH.exists() else set()
    for c in range(n_chunks):
        chunk = cand_lines[c * args.chunk : (c + 1) * args.chunk]
        raw = teacher.chat(
            [{"role": "user", "content": SELECT_PROMPT.format(
                top=per_chunk, moves="、".join(MOVES), situations="、".join(SITUATIONS),
                candidates="\n".join(chunk))}],
            max_tokens=8000, json_mode=True, temperature=0.3, task="templates",
        )
        raws.append(raw)
        RAW_OUT.write_text("\n\n=====\n\n".join(raws), encoding="utf-8")  # 解析失败时方便排查
        for t in parse_json(raw).get("templates", [])[:per_chunk]:
            key = skeleton(str(t.get("template", "")))
            if key and key not in seen and key not in banned:
                seen.add(key)
                templates.append(t)
        print(f"  第 {c + 1}/{n_chunks} 批：累计 {len(templates)} 个模板")
    TEMPLATES_JSON.write_text(json.dumps(templates, ensure_ascii=False, indent=2), encoding="utf-8")
    TEMPLATES_MD.write_text(render_md(templates), encoding="utf-8")
    print(f"{len(templates)} 个模板 → {TEMPLATES_JSON}、{TEMPLATES_MD}")
    print("可以直接删改 templates.json，然后运行 python -m scripts.build_templates --render-only 重新生成 md")


if __name__ == "__main__":
    if "--render-only" in sys.argv:
        sys.stdout.reconfigure(encoding="utf-8")
        TEMPLATES_MD.write_text(render_md(json.loads(TEMPLATES_JSON.read_text(encoding="utf-8"))), encoding="utf-8")
        print(f"→ {TEMPLATES_MD}")
    else:
        main()
