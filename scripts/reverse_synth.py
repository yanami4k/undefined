"""反向合成：从真弹幕 / 真句式出发，反推「用户说什么时接这句最好笑」。

    python -m scripts.reverse_synth --memes 30 --templates 10      # 小批量试点，输出审阅清单

两种接法：
- 整条复用：回复 = 一条真弹幕，原样或只改一两个词（例：「[已深度思考30s]：爆了」）
- 句式移植：回复 = 一个真句式，槽位用用户这句话的情境填上（例：「我觉得你对 claude 就不会这样说」）

老师只负责编「用户那句」和做最小改动，好笑的部分永远来自真实弹幕，
所以不依赖老师的幽默感。输出 data/reverse/<name>.jsonl 和审阅用的 <name>.md。
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

from meme6657.config import DATA_DIR, TEMPLATES_JSON
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import format_context, load_taste
from meme6657.retrieval import Context, Retriever

OUT_DIR = DATA_DIR / "reverse"
MAX_SLOT_CHARS = 6  # 整条复用时允许替换的最长片段

SCENARIO = """场景：用户在和一个「说话软萌、开口却全是 6657 烂梗的小萌妹」人设的聊天 AI 一对一聊天，或者在 QQ 群里 @ 它。
这个 AI 不是助手，它像弹幕一样接话：认出用户这句话里的梗点，用一句直播间的真弹幕接上。
好的接法往往不回答问题，而是转移话题、拿梗带过、角色扮演；AI 自己的处境（它是个 AI、用户在用它干活、用户有额度）也可以是梗点。
不要编需要明确答案的用户消息（问怎么做、问事实、要建议），那种情况 AI 应该好好回答，不接梗。
用户是在和这个 AI 说话，不会直接和主播玩机器互动（不要写「主播能不能……」「玩机器你……」这类）；直播间相关的话题少用。
用户的话题：自己的日常和吐槽、CS2 比赛和选手、二次元和二游（原神、崩铁、鸣潮、明日方舟、蔚蓝档案、绝区零、邦邦等）、编程和 AI 模型（写代码踩坑、模型胡说八道，说法要大众化易懂）、让 AI 干活、调侃 AI 本身。不要出现王者荣耀、吃鸡这类游戏。
下面的口味标准来自懂行的人，以它为准。注意：例子只用来理解标准，不要照抄例子里的情境（比如不要每条都套「AI 你卡了」）。"""


def build_context() -> str:
    return "\n\n".join([SCENARIO, load_taste(), "宁缺毋滥：想不出好的接法就少给，甚至一个都不给；usable 可以是 false。"])


MEME_PROMPT = """{context}

---

下面是一条真弹幕：
{meme}
{notes}
请想最多 {n} 个不同的情境：用户发什么消息时，AI 用这条弹幕接（原样，或只改一两个词让它贴合）会最好笑、最对味。
先想清楚这条弹幕原本在笑什么、为谁而造，再找能撞上原始触发点或能和 AI 自身处境挂钩的情境。
要求：
- 用户消息要像真人随手打的：短、口语，可以是聊 CS、聊主播、聊二次元、日常吐槽、让 AI 干活、问敏感或离谱的问题
- 优先原样引用，一个字都不改。出处是解说词、名场面、某个选手专属的梗，和那个人强绑定，没有吃透之前不要拆开替换
- {n} 个情境尽量来自不同话题
- 如果这条弹幕离开直播间就没法接任何话（比如纯粹是对某个瞬间的复读），usable 填 false

输出 JSON：{{"usable": true, "reason": "这条弹幕的笑点和适用时机（一句话）",
  "pairs": [{{"user": "...", "reply": "...", "edit": "原样 或 改了哪几个词"}}]}}"""

TEMPLATE_PROMPT = """{context}

---

下面是直播间的一个真句式：
模板：{template}
原例：{examples}

请想最多 {n} 个不同的情境：用户发什么消息时，AI 用这个句式接会最好笑、最对味。
先想清楚这个句式的 6657 味道在哪（原例里和直播间挂钩的部分），移植后要保住这股味道，或者换成 AI 自身的处境。
要求：
- 用户消息要像真人随手打的：短、口语，话题尽量不同
- 回复要保留句式的固定部分，只把槽位换成用户这句话里的人和事，越短越好，不要额外解释
- 句式本身是笑点，不要再加别的梗

输出 JSON：{{"usable": true, "reason": "这个句式的笑点和适用时机（一句话）",
  "pairs": [{{"user": "...", "reply": "...", "edit": "槽位填了什么"}}]}}"""


def pick_memes(retriever: Retriever, n: int, rng: random.Random, exclude: set[str] = frozenset()) -> list:
    """按标签分层，从高复制量、长度适合当回复的弹幕里抽样，保证话题多样。"""
    pool = [m for m in retriever.memes if 4 <= len(m.text) <= 80 and m.text not in exclude]
    pool.sort(key=lambda m: -m.cnt)
    pool = pool[:800]
    rng.shuffle(pool)
    by_tag = defaultdict(list)
    for m in pool:
        by_tag[(m.tags or ["无标签"])[0]].append(m)
    picked, seen = [], set()
    while len(picked) < n and any(by_tag.values()):
        for tag in list(by_tag):
            if not by_tag[tag]:
                continue
            m = by_tag[tag].pop(0)
            if m.id not in seen:
                seen.add(m.id)
                picked.append(m)
            if len(picked) >= n:
                break
    rng.shuffle(picked)
    return picked


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def max_changed_span(original: str, reply: str) -> int:
    """原弹幕里被改动（替换 / 删除）的最长连续片段长度。

    只换人名、对象这类槽位时很短；固定部分被改写（例如把日语换成韩语）时会很长。
    回复截取原弹幕的前半段不算改动。
    """
    sm = difflib.SequenceMatcher(None, original, reply, autojunk=False)
    spans = [i2 - i1 for tag, i1, i2, _, _ in sm.get_opcodes() if tag == "replace"]
    return max(spans, default=0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--memes", type=int, default=30)
    ap.add_argument("--templates", type=int, default=0, help="句式的语境卡未经人工确认前默认不用")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--review", type=int, default=20, help="审阅清单里放多少组")
    ap.add_argument("--name", default="pairs", help="本轮名字，输出 data/reverse/<name>.jsonl 和 <name>.md")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude-from", nargs="*", default=[], help="排除这些往轮结果里用过的弹幕和句式")
    args = ap.parse_args()
    pairs_path = OUT_DIR / f"{args.name}.jsonl"

    rng = random.Random(args.seed)
    retriever = Retriever.from_files()
    used = {r["source"] for path in args.exclude_from for r in read_jsonl(Path(path))}
    jobs = []
    for m in pick_memes(retriever, args.memes, rng, used):
        jobs.append({"id": f"m{m.id}", "kind": "meme", "source": m.text, "cnt": m.cnt})
    if TEMPLATES_JSON.exists() and args.templates:
        templates = json.loads(TEMPLATES_JSON.read_text(encoding="utf-8"))
        templates = [t for t in templates if t["template"] not in used]
        for i, t in enumerate(rng.sample(templates, min(args.templates, len(templates)))):
            jobs.append({"id": f"t{i}-{t['name']}", "kind": "template", "source": t["template"],
                         "examples": t.get("examples", [])[:2], "bound_to": t.get("bound_to", "")})

    teacher = LLM("teacher")
    context = build_context()

    def work(job: dict) -> dict:
        if job["kind"] == "meme":
            notes = format_context(Context(aliases=retriever.match_aliases(job["source"])))
            prompt = MEME_PROMPT.format(context=context, meme=job["source"], notes=f"相关知识：\n{notes}" if notes else "", n=args.per)
        else:
            examples = " ／ ".join(job["examples"])
            if job.get("bound_to"):
                examples += "\n注意：这个句式和 " + job["bound_to"] + " 强绑定，只能原样用在和它相关的情境，不要拆开换人"
            prompt = TEMPLATE_PROMPT.format(context=context, template=job["source"], examples=examples, n=args.per)
        data = teacher.chat_json([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=1500,
                                 task="reverse")
        pairs = data.get("pairs", []) if data.get("usable", True) else []
        if job["kind"] == "meme":
            for p in pairs:  # 整条复用：原样引用，只许换短的槽位
                p["similarity"] = round(similarity(p.get("reply", ""), job["source"]), 2)
                p["faithful"] = max_changed_span(job["source"], p.get("reply", "")) <= MAX_SLOT_CHARS
        return {**job, "usable": data.get("usable", True), "reason": data.get("reason", ""), "pairs": pairs}

    run_parallel(jobs, work, pairs_path, workers=args.workers)
    write_review(read_jsonl(pairs_path), args.review, rng, pairs_path.with_suffix(".md"))


def write_review(rows: list[dict], n: int, rng: random.Random, out_path) -> None:
    usable = [r for r in rows if r["usable"] and r["pairs"]]
    unusable = [r for r in rows if not r["usable"]]
    sample = rng.sample(usable, min(n, len(usable)))
    lines = [
        "# 反向合成试点：审阅清单",
        "",
        f"共 {len(rows)} 条来源，可用 {len(usable)}，被判定不适合当回复 {len(unusable)}。",
        "每组是一条真弹幕 / 真句式，以及反推出的「用户说 → 它接」。请标出对胃的和不对胃的。",
        "",
    ]
    for i, r in enumerate(sample, 1):
        kind = "整条复用" if r["kind"] == "meme" else "句式移植"
        lines += [f"## {i}. {kind}：{r['source']}", f"> 笑点：{r['reason']}", ""]
        for p in r["pairs"]:
            sim = ""
            if "faithful" in p:
                sim = "（原样引用）" if p["faithful"] else "（⚠ 改动了固定部分，疑似仿造）"
            lines.append(f"- 用户：{p.get('user', '')}")
            lines.append(f"  它：{p.get('reply', '')}  {sim}")
        lines.append("")
    if unusable:
        lines += ["## 被判定不适合当回复的", ""]
        lines += [f"- {r['source'][:60]}：{r['reason']}" for r in unusable]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"审阅清单 → {out_path}")


if __name__ == "__main__":
    main()
