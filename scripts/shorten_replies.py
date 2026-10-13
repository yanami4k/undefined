"""把偏长的「祥子自己开口」回复压短，做成卡片让人重新打标。

    python -m scripts.shorten_replies --name short1               # 金标里判为好 / 电波、超过 60 字的正常回复
    python -m scripts.shorten_replies --name short1 --limit 5     # 先试几条

eval-v0 的打分显示回复越长越容易出错：30 字以内全部可接受，超过 100 字只有三分之一。
小模型写长回复既不讨喜又容易错，所以正常回复只留一个要点，压到 --target 字左右、最多 --max-chars 字。
老师只做删减和改写，不加原文没有的信息；人设检查沿用 normal_replies 的规则，不合格打回重写。
带代码块的回复（cfg、脚本）不压：删掉一行就跑不起来。
输出 data/reverse/<name>.jsonl，卡片写到 data/label_cards/<name>/（kind=normal），可以直接发到打标页。
"""

from __future__ import annotations

import argparse
import json
import sys

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import load_style_card, teacher_system
from meme6657.retrieval import Retriever
from scripts.normal_replies import MemeIndex, problems, soft_problems, strip_followup, tidy_tics

GOLD_PATH = DATA_DIR / "gold" / "labels.jsonl"
OUT_DIR = DATA_DIR / "reverse"

SHORTEN_HINT = """---

## 这一批的任务：把祥子的回复压短

下面每条是用户消息和祥子已经说过的一条回复，懂行的人评过，内容是对的。问题是太长：小模型学长回复，既不讨喜又容易出错。
把它压到 {target} 字左右，最多 {max_chars} 字，一两句话：
- 只留最核心的一个要点或结论；事实、数字不能改，也不能加原文没有的信息
- 需要大段专业知识才能说全的问题，只给一句结论；原文本身就含糊的地方，直说拿不准，不要硬凑
- 原回复里原样引用的弹幕片段（下面列出）要么一字不改地留下，要么整段去掉，不能截断或改写
- 人设照旧：自称祥子，不说「我」；句尾口癖（喵 / desuwa）最多一处；不用列表、颜文字和「～」
- 答完就收，结尾不要往下推话题

输出 JSON：{{"reply": "压短后的回复"}}"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="批次名，例如 short1")
    ap.add_argument("--min-chars", type=int, default=60, help="只处理超过这个长度的回复")
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条试跑，0 表示全部")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(GOLD_PATH)
            if r["kind"] == "normal" and r["verdict"] in ("good", "wave") and len(r["reply"]) > args.min_chars]
    # 代码、配置一删就跑不起来（试跑时 cfg 少了一行 alias），这类回复不压
    code = [r for r in rows if "```" in r["reply"]]
    rows = [r for r in rows if "```" not in r["reply"]]
    if args.limit:
        rows = rows[: args.limit]
    print(f"待压短 {len(rows)} 条（正常回复、判为好 / 电波、超过 {args.min_chars} 字）；带代码块跳过 {len(code)} 条")

    retriever = Retriever.from_files()
    index = MemeIndex([m.text for m in retriever.memes])
    system = teacher_system(load_style_card()) + "\n\n" + SHORTEN_HINT.format(target=args.target, max_chars=args.max_chars)
    teacher = LLM("teacher")

    def work(r: dict) -> dict:
        user, orig = r["user"], r["reply"]
        orig_quotes = index.quotes(user, orig)
        prompt = (f"用户说：{user}\n祥子原来的回复（{len(orig)} 字）：{orig}\n"
                  f"原回复里引用的弹幕片段：{'；'.join(orig_quotes) if orig_quotes else '无'}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        tries, asked = [], set()
        for _ in range(3):
            data = teacher.chat_json(messages, temperature=0.5, max_tokens=400, task="chat")
            raw = str(data.get("reply", "")).strip()
            memes = [m.text for m in retriever.search_memes(user, k=8) + retriever.search_memes(raw, k=8)]
            reply = strip_followup(tidy_tics(user, raw, memes, keep=index.quotes(user, raw)))
            quotes = index.quotes(user, reply)
            issues = problems(user, reply, memes, "chat", quotes)
            if len(reply) > args.max_chars:
                issues.append(f"还是太长（{len(reply)} 字）：压到 {args.target} 字左右，最多 {args.max_chars} 字，只留一个要点")
            soft = [s for s in soft_problems(user, reply, memes, quotes) if s not in asked]
            tries.append({"reply": reply, "raw": raw, "issues": issues})
            if not issues and not soft:
                break
            asked |= set(soft)
            feedback = "这条请再改一下，仍按 JSON 输出：\n- " + "\n- ".join(issues + soft)
            messages = messages + [{"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                                   {"role": "user", "content": feedback}]
        last = tries[-1]
        return {"id": f"{args.name}-{r['id']}", "orig_id": r["id"], "user": user, "orig_reply": orig,
                "reply": last["reply"], "raw_reply": last["raw"], "kind": "normal", "tries": len(tries),
                "passed": not last["issues"], "issues": last["issues"]}

    out_path = OUT_DIR / f"{args.name}.jsonl"
    run_parallel(rows, work, out_path, workers=args.workers)

    done = read_jsonl(out_path)
    card_dir = DATA_DIR / "label_cards" / args.name
    card_dir.mkdir(parents=True, exist_ok=True)
    for n, d in enumerate(done, 1):
        card = {"batch": args.name, "order": 8000 + n, "kind": "normal", "source": "", "user": d["user"], "reply": d["reply"]}
        (card_dir / f"{d['id']}.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    lens = sorted(len(d["reply"]) for d in done)
    print(f"合格 {sum(d['passed'] for d in done)}/{len(done)}；压短后中位 {lens[len(lens) // 2] if lens else 0} 字，"
          f"最长 {lens[-1] if lens else 0} 字；卡片写到 {card_dir}")


if __name__ == "__main__":
    main()
