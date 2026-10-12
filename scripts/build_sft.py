"""把人工打标的金标数据整理成 ms-swift 训练集。

    python -m scripts.sync_labels --labels <导出的 labels 目录>   # 先合并打标结果
    python -m scripts.build_sft

数据来源 data/gold/labels.jsonl，只取 good / wave，且只取下面三种类型（评估卡片 eval 打完分也会进金标，但不能进训练集）：
- 接梗（meme / template）：回复是原样的弹幕，不做任何清理；电波样本重复 --wave-weight 次加权
- 正常回复（normal）：用当前清理规则再过一遍（去掉颜文字、结尾主动推话题的那句），引用的弹幕片段原样保留
用户消息按线上推理的格式附 [参考梗]（同一个检索器）。接梗样本以 --inject 的概率把目标弹幕放进 [参考梗]，
让学生既学会背梗，也学会从参考里原样取用；其余样本保持检索的自然结果。
原样搬来的弹幕回复命中屏蔽词的丢弃（安全兜底）；过期梗（kb/expired_memes.json）也剔掉。

输出 data/sft/train.jsonl 和 data/sft/val.jsonl，格式为 ms-swift 自定义数据集的 {"messages": [...]}，
按用户消息划分，同一条消息的样本不会同时出现在 train 和 val。
通用对话数据不在这里混入，训练时用 ms-swift 的 dataset#N 采样直接加（见 train/sft.sh）。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, stable_hash, write_jsonl
from meme6657.prompts import STUDENT_SYSTEM, format_user
from meme6657.retrieval import Context, Meme, Retriever, is_expired, load_blocklist, load_expired

GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"
SFT_DIR = DATA_DIR / "sft"
TRAIN_KINDS = ("meme", "template", "normal")


def training_rows(gold: list[dict]) -> list[dict]:
    """金标里能当训练样本的：判为好 / 电波，且是训练用的类型。

    评估卡片（kind=eval）是学生模型对固定留出消息的回复，进了训练集，之后的评估就不是留出的了。
    用白名单而不是排除 eval，以后加了新类型的卡片也不会悄悄混进来。
    """
    return [r for r in gold if r["verdict"] in ("good", "wave") and r["kind"] in TRAIN_KINDS]


def _phrases(row: dict, n: int) -> set[str]:
    out = set()
    for m in row["messages"]:
        if m["role"] == "assistant":
            s = "".join(m["content"].split())
            out.update(s[i : i + n] for i in range(len(s) - n + 1))
    return out


def cap_repeated_phrases(rows: list[dict], n: int = 10, cap: int = 5) -> tuple[list[dict], list[tuple[str, int]]]:
    """限制套路话术：同一个 n 字片段最多出现在 cap 个样本的回复里，超出的样本丢弃。

    老师模型容易反复用同一句话（尤其是照搬风格卡里的示例），不限量的话学生会学成固定话术。
    返回 (保留的样本, 出现最频繁的片段及其样本数)。
    """
    df = Counter()
    phrases = [_phrases(r, n) for r in rows]
    for p in phrases:
        df.update(p)
    hot = {g for g, c in df.items() if c > cap}
    used = Counter()
    kept = []
    for r, p in zip(rows, phrases):
        mine = p & hot
        if any(used[g] >= cap for g in mine):
            continue
        used.update(mine)
        kept.append(r)
    return kept, df.most_common(10)


def with_target(ctx: Context, target: Meme | None, rng: random.Random, k: int = 3) -> Context:
    """把目标弹幕放进 [参考梗]（随机位置；已经满 k 条时随机替换掉一条）。"""
    if target is None or any(m.text == target.text for m in ctx.memes):
        return ctx
    memes = list(ctx.memes)
    if len(memes) >= k:
        memes.pop(rng.randrange(len(memes)))
    memes.insert(rng.randrange(len(memes) + 1), target)
    return Context(aliases=ctx.aliases, memes=memes)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--wave-weight", type=int, default=2, help="电波样本重复的次数")
    ap.add_argument("--inject", type=float, default=0.5, help="接梗样本把目标弹幕放进 [参考梗] 的概率")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.normal_replies import MemeIndex, problems, strip_followup, tidy_tics  # 正常回复的清理规则

    rng = random.Random(args.seed)
    retriever = Retriever.from_files()
    by_text = {m.text: m for m in retriever.memes}
    index = MemeIndex([m.text for m in retriever.memes])
    blocklist = load_blocklist()
    expired = load_expired()

    gold = training_rows(read_jsonl(GOLD_PATH))
    samples, stats, dropped, no_persona = [], Counter(), [], 0
    for r in sorted(gold, key=lambda r: r["id"]):
        user, reply = r["user"].strip(), r["reply"].strip()
        # 屏蔽词只查模型要说的话：用户骂人（对骂样本）不算；正常回复是老师按禁止清单写、人工逐条看过的，
        # 「袜子一湿」这种字面命中不该丢，所以只查原样搬来的弹幕
        if r["kind"] != "normal" and any(b in reply.lower() for b in blocklist):
            dropped.append(r["id"])
            continue
        if is_expired(reply + r.get("source", ""), expired):  # 梗的前提已经变了（见 kb/expired_memes.json）
            stats["过期梗剔除"] += 1
            continue
        ctx = retriever.retrieve(user)
        if r["kind"] == "normal":
            memes = [m.text for m in retriever.search_memes(user, k=8) + retriever.search_memes(reply, k=8)]
            reply = strip_followup(tidy_tics(user, reply, memes, keep=index.quotes(user, reply)))
            no_persona += bool(problems(user, reply, memes, "help", index.quotes(user, reply)))
        elif rng.random() < args.inject:
            target = by_text.get(r.get("source", "").strip())
            if target is not None:
                ctx = with_target(ctx, target, rng)
                stats["注入目标弹幕"] += 1
        messages = [{"role": "system", "content": STUDENT_SYSTEM},
                    {"role": "user", "content": format_user(user, ctx)},
                    {"role": "assistant", "content": reply}]
        times = args.wave_weight if r["verdict"] == "wave" else 1
        stats[f"{r['kind']}/{r['verdict']}"] += 1
        samples += [{"user": user, "messages": messages, "copy": i} for i in range(times)]

    train, val = [], []
    # 打乱但可复现；加权重复的样本带上副本序号，不会挨在一起
    for s in sorted(samples, key=lambda s: stable_hash(f"{s['user']}{s['messages'][2]['content']}#{s['copy']}")):
        (val if stable_hash(s["user"] + "#val") < args.val_ratio else train).append({"messages": s["messages"]})

    n_train = write_jsonl(SFT_DIR / "train.jsonl", train)
    n_val = write_jsonl(SFT_DIR / "val.jsonl", val)
    kinds = Counter("normal" if k.startswith("normal") else "接梗" for k in stats.elements() if "/" in k)
    weighted_meme = sum(v * (args.wave_weight if k.endswith("wave") else 1)
                        for k, v in stats.items() if "/" in k and not k.startswith("normal"))
    chars = sum(len(m["content"]) for s in train + val for m in s["messages"] if m["role"] != "system")
    print(f"train {n_train} / val {n_val} → {SFT_DIR}")
    print("来源：", {k: v for k, v in sorted(stats.items()) if "/" in k})
    print(f"去重前接梗 {kinds['接梗']} 条（电波加权后 {weighted_meme}），正常回复 {kinds['normal']} 条，"
          f"正常回复占比 {kinds['normal'] / max(1, weighted_meme + kinds['normal']):.0%}")
    print(f"过期梗剔除 {stats['过期梗剔除']} 条；接梗样本里注入目标弹幕 {stats['注入目标弹幕']} 条；屏蔽词丢弃 {len(dropped)} 条 {dropped[:5]}；"
          f"清理后缺人设标记的正常回复 {no_persona} 条")
    print(f"总字符约 {chars / 1e4:.1f} 万（不含系统提示词）")


if __name__ == "__main__":
    main()
