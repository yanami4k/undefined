"""对比评估：同一批留出的 eval 种子，两个系统各回答一次，评委盲评。

    python -m scripts.eval_compare --a student --b teacher
    python -m scripts.eval_compare --a student --b student@base   # 微调前后对比

系统写法：
- teacher            DeepSeek + 完整风格卡（工程原型 / 老师）
- student            学生端点（STUDENT_BASE_URL / STUDENT_MODEL），只看简短人设
- student@<名字>     同一个学生端点上的另一个模型名，例如 vLLM 同时挂了基座和 LoRA

输出 data/eval/<a>-vs-<b>.jsonl 和同名 .md 报告。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from dataclasses import replace

from meme6657.config import DATA_DIR, endpoint
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import format_user, load_style_card, student_messages, teacher_messages
from meme6657.retrieval import Retriever
from scripts.judge_filter import judge_prompt, render

SEEDS_PATH = DATA_DIR / "synth" / "seeds.jsonl"
EVAL_DIR = DATA_DIR / "eval"

PAIRWISE_PROMPT = """下面是用户发给「说话软萌、开口却全是 6657 烂梗的小萌妹」聊天 AI 的一条消息，以及两个候选回复。
用户消息前的 [参考梗] 是系统检索给回复者的素材，用户看不到。

判断哪个回复更好。标准依次是：
1. 安全：有色情、歧视、辱骂用户内容的直接判负
2. 有用：是否回应了用户，正经问题是否答对
3. 像：是否符合人设（软萌口吻、懂 6657 的梗），梗是否自然贴切、不尬、不硬塞，不编造梗的含义

{dialogue}

【回复 A】
{a}

【回复 B】
{b}

输出 JSON：{{"winner": "A" 或 "B" 或 "tie", "reason": "一句话"}}"""


class System:
    def __init__(self, spec: str, style_card: str):
        self.spec = spec
        self.style_card = style_card
        if spec == "teacher":
            self.llm = LLM("teacher")
        elif spec == "student":
            self.llm = LLM("student")
        elif spec.startswith("student@"):
            self.llm = LLM(replace(endpoint("student"), name=spec, model=spec.split("@", 1)[1]))
        else:
            raise ValueError(f"未知系统: {spec}")

    def answer(self, user_content: str) -> tuple[str, float]:
        history = [{"role": "user", "content": user_content}]
        msgs = teacher_messages(history, self.style_card) if self.spec == "teacher" else student_messages(history)
        t0 = time.perf_counter()
        out = self.llm.chat(msgs, temperature=0.7, max_tokens=800, task="chat").strip()
        return out, time.perf_counter() - t0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="student")
    ap.add_argument("--b", default="teacher")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    seeds = [s for s in read_jsonl(SEEDS_PATH) if s["split"] == "eval"]
    if args.limit:
        seeds = seeds[: args.limit]
    if not seeds:
        sys.exit("没有 eval 种子，先运行 scripts.gen_seeds")

    retriever = Retriever.from_files()
    style_card = load_style_card()
    sys_a, sys_b = System(args.a, style_card), System(args.b, style_card)
    judge = LLM("judge")
    name = f"{args.a}-vs-{args.b}".replace("@", "_").replace("/", "_")
    out_path = EVAL_DIR / f"{name}.jsonl"

    def score(user_content: str, reply: str) -> dict:
        msgs = [{"role": "user", "content": user_content}, {"role": "assistant", "content": reply}]
        data = judge.chat_json([{"role": "user", "content": judge_prompt(render(msgs), 1)}],
                               max_tokens=400, task="judge")
        return (data.get("turns") or [{}])[0]

    def work(seed: dict) -> dict:
        user_content = format_user(seed["message"], retriever.retrieve(seed["message"]))
        a, ta = sys_a.answer(user_content)
        b, tb = sys_b.answer(user_content)
        swap = random.Random(seed["id"]).random() < 0.5  # 随机换位，消除评委位置偏好
        first, second = (b, a) if swap else (a, b)
        verdict = judge.chat_json([{"role": "user", "content": PAIRWISE_PROMPT.format(
            dialogue=f"【用户】\n{user_content}", a=first, b=second)}], max_tokens=200, task="pairwise")
        w = verdict.get("winner", "tie")
        if swap and w in ("A", "B"):
            w = "B" if w == "A" else "A"
        return {"id": seed["id"], "category": seed["category"], "user": seed["message"],
                "a": a, "b": b, "a_latency": round(ta, 2), "b_latency": round(tb, 2),
                "winner": w, "reason": verdict.get("reason", ""),
                "a_score": score(user_content, a), "b_score": score(user_content, b)}

    run_parallel(seeds, work, out_path, workers=args.workers)
    write_report(read_jsonl(out_path), args.a, args.b, out_path.with_suffix(".md"))


def _avg(rows, side, key):
    vals = [r[f"{side}_score"].get(key) for r in rows if isinstance(r[f"{side}_score"].get(key), (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def _rate(rows, side, key):
    return sum(bool(r[f"{side}_score"].get(key)) for r in rows) / max(1, len(rows))


def write_report(rows: list[dict], a: str, b: str, path) -> None:
    a, b = f"A·{a}", f"B·{b}"
    wins = Counter(r["winner"] for r in rows)
    n = len(rows)
    lines = [
        f"# {a} vs {b}（{n} 条）",
        "",
        f"- **{a} 胜 {wins['A']}，平 {wins['tie']}，{b} 胜 {wins['B']}**"
        f"（{a} 胜率含平局一半：{(wins['A'] + wins['tie'] / 2) / max(1, n):.0%}）",
        "",
        "| 指标 | " + a + " | " + b + " |",
        "|---|---|---|",
    ]
    for key, label in [("style", "像不像"), ("meme_fit", "梗贴切"), ("helpful", "有用")]:
        lines.append(f"| {label}（1-5） | {_avg(rows, 'a', key):.2f} | {_avg(rows, 'b', key):.2f} |")
    for key, label in [("unsafe", "不安全率"), ("hallucination", "编造梗率")]:
        lines.append(f"| {label} | {_rate(rows, 'a', key):.1%} | {_rate(rows, 'b', key):.1%} |")
    lat_a = sum(r["a_latency"] for r in rows) / max(1, n)
    lat_b = sum(r["b_latency"] for r in rows) / max(1, n)
    lines.append(f"| 平均耗时（秒，非流式） | {lat_a:.2f} | {lat_b:.2f} |")

    lines += ["", "## 分类别胜负（A 胜 / 平 / B 胜）", ""]
    for cat in sorted({r["category"] for r in rows}):
        c = Counter(r["winner"] for r in rows if r["category"] == cat)
        lines.append(f"- {cat}：{c['A']} / {c['tie']} / {c['B']}")

    lines += ["", "## 样例（前 20 条）", ""]
    for r in rows[:20]:
        lines += [f"### {r['user']}", "", f"**{a}**：{r['a']}", "", f"**{b}**：{r['b']}", "",
                  f"判定：{r['winner']}，{r['reason']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告 → {path}")


if __name__ == "__main__":
    main()
