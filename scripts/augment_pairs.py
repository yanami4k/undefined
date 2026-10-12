"""情境复用：给人工评为好 / 电波的「整条甩梗」样本，写同一情境下的新用户消息，回复原样不变。

    python -m scripts.augment_pairs --name aug1 --sample 30 --n 2     # 小规模试点，先打标看命中率
    python -m scripts.augment_pairs --name aug2 --n 3                 # 命中率够高再全量

一条梗对上过某个情境，用户换个说法说同一件事，原样甩过去通常还对得上。新消息由老师写，要求：
情境和动作不变、换说法和细节、绑定的对象（点名的选手、队伍）保留、不提前用梗词往梗上撞。
同一族梗（含原有样本）最多 FAMILY_CAP 条，名额先给标注少的梗。只处理整条甩梗（meme）：套句式和自己开口的回复是贴着原消息写的，换了消息就不成立。
老师先写出梗靠哪个词接上（钩子），新消息必须保留钩子；程序再过滤丢了钩子、抄了梗原话、和已有消息重复、只是把原句换了一两个字的。输出 data/reverse/<name>.jsonl，卡片写到 data/label_cards/<name>/。
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from collections import Counter

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import load_taste
from meme6657.retrieval import Retriever
from scripts.match_synth import REACTIONS_PATH, Families, is_echo

GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"
OUT_DIR = DATA_DIR / "reverse"
VERDICT_ZH = {"good": "好", "wave": "电波"}
FAMILY_CAP = 8  # 同一族梗（含原有的好 / 电波样本）最多这么多条，免得打标时反复看到同一条

PROMPT = """{taste}

---

下面这条接法，懂行的人评为「{verdict}」：
用户：{user}
回复：{reply}

请写 {n} 条新的用户消息：真实用户在同一种情境下可能发的话，而且上面这条回复原样接上去同样成立。
- 情境和动作不变：用户此刻在干什么、什么情绪、冲着谁说。换说法、换细节、换长短和语气，{n} 条彼此也要不一样
- 回复的梗如果绑定了具体的人、队伍、事件（例如回复里点名的选手），新消息里这个对象必须保留，不能换
- 用户事先不知道会被怎么接：不要在消息里提前用回复里的梗词，不要往梗上撞
- 符合用户画像：单聊或 QQ 群里跟 AI 说话；游戏只聊 CS2、DOTA2、二游；说到的事实要对，不编不存在的梗
- 口语，像在聊天软件里打字，一般 5 到 40 个字

先想清楚这条回复是靠用户消息里的哪个词接上的（钩子），写进 hook（一个词，原样出现在用户消息里）。
钩子是让梗成立的触发点：状态、动作、情绪或话题，一般不是人名本身。例如「NiKo 又软了」配「世界上最软的几样东西」，钩子是「软」；
「玩机器最近是不是又胖了」配猪头肉梗，钩子是「胖」；「你是不是被降智了」配「爆了」，钩子是「降智」；「一起打二游吗」配「二刺螈来了」，钩子是「二游」。
每条新消息都必须原样包含这个钩子，换掉其余部分；钩子一丢，梗就接不上了。

输出 JSON：{{"hook": "钩子", "messages": ["……"]}}"""


def copies_meme(message: str, reply: str, min_len: int = 6) -> bool:
    """用户消息里抄了梗的原话（连续重合 min_len 字以上），等于替回复把梗说了。"""
    sm = difflib.SequenceMatcher(None, message, reply, autojunk=False)
    return sm.find_longest_match(0, len(message), 0, len(reply)).size >= min_len


def keeps_hook(message: str, hook: str, origin: str) -> bool:
    """新消息保留了让梗接得上的钩子。钩子不在原消息里（老师没按要求写）时不做这项检查。"""
    if not hook or hook not in origin:
        return True
    return hook.lower() in message.lower()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=2, help="每条样本写几条新消息")
    ap.add_argument("--sample", type=int, default=0, help="只抽这么多条样本（试点用），0 为全部")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    gold = read_jsonl(GOLD_PATH)
    seen_users = {r["user"] for r in gold}
    kept = [r for r in gold if r["kind"] == "meme" and r["verdict"] in VERDICT_ZH]
    # 只扩原始样本：变体（batch 以 aug 开头）不再拿来扩，已经扩过的原样本也跳过；变体照样计入梗族上限
    done = {r["id"].split("-", 1)[1].rsplit("-a", 1)[0] for r in gold if r["batch"].startswith("aug")}
    pairs = [r for r in kept if not r["batch"].startswith("aug") and r["id"] not in done
             and "meme_ask_unknown" not in r["id"]]  # 问编出来的梗这类用户消息本身是坏数据，不扩
    rng = random.Random(args.seed)
    if args.sample:
        pairs = rng.sample(pairs, min(args.sample, len(pairs)))
    # 按梗族限量：原有样本已经够多的梗少扩或不扩，名额先给标注少的梗
    lib = json.loads(REACTIONS_PATH.read_text(encoding="utf-8")) if REACTIONS_PATH.exists() else []
    family = Families(Retriever.from_files(), [e["text"] for e in lib])
    have = Counter(family(r["source"]) for r in kept)
    quota = {f: max(0, FAMILY_CAP - c) for f, c in have.items()}
    rng.shuffle(pairs)
    for r in pairs:
        f = family(r["source"])
        r["n"] = min(args.n, quota[f])
        quota[f] -= r["n"]
    skipped = sum(r["n"] == 0 for r in pairs)
    pairs = [r for r in pairs if r["n"] > 0]
    print(f"样本 {len(pairs)} 条（{skipped} 条所在梗族已满 {FAMILY_CAP} 条，不扩），每条最多写 {args.n} 条新消息")

    taste = load_taste()
    teacher = LLM("teacher")

    def work(r: dict) -> dict:
        prompt = PROMPT.format(taste=taste, verdict=VERDICT_ZH[r["verdict"]], user=r["user"], reply=r["reply"], n=r["n"])
        data = teacher.chat_json([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=600, task="augment")
        msgs = [str(m).strip() for m in data.get("messages", []) if str(m).strip()]
        return {"id": r["id"], "reply": r["reply"], "source": r.get("source", ""), "origin_user": r["user"],
                "origin_verdict": r["verdict"], "hook": str(data.get("hook", "")).strip(), "messages": msgs[: r["n"]]}

    out_path = OUT_DIR / f"{args.name}.raw.jsonl"
    run_parallel(pairs, work, out_path, workers=args.workers)

    rows, dropped = [], 0
    for x in read_jsonl(out_path):
        for k, m in enumerate(x["messages"]):
            too_close = difflib.SequenceMatcher(None, m, x["origin_user"]).ratio() > 0.75  # 只是原句换了一两个字
            if m in seen_users or is_echo(m, x["reply"]) or too_close or copies_meme(m, x["reply"])                     or not keeps_hook(m, x.get("hook", ""), x["origin_user"]):
                dropped += 1
                continue
            seen_users.add(m)
            rows.append({"id": f"{x['id']}-a{k}", "kind": "meme", "user": m, "reply": x["reply"], "source": x["source"],
                         "origin_id": x["id"], "origin_user": x["origin_user"], "origin_verdict": x["origin_verdict"]})
    final = OUT_DIR / f"{args.name}.jsonl"
    final.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    card_dir = DATA_DIR / "label_cards" / args.name
    card_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(len(rows)))
    rng.shuffle(order)  # 打乱，同一条梗的变体不挨在一起
    for n, i in enumerate(order, 1):
        r = rows[i]
        card = {"batch": args.name, "order": 9500 + n, "kind": "meme", "source": r["source"], "user": r["user"], "reply": r["reply"]}
        (card_dir / f"{args.name}-{r['id']}.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    print(f"得到 {len(rows)} 条（过滤 {dropped} 条）→ {final}；卡片 → {card_dir}")


if __name__ == "__main__":
    main()
