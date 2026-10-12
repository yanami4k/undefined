"""评委打分并过滤合成对话。

    python -m scripts.judge_filter

逐段对话给每个回复打分，输出：
- data/synth/judged.jsonl    带分数的全部对话
- data/synth/filtered.jsonl  通过阈值的对话（进入训练集）
并按类别打印通过率，方便发现风格卡或种子的问题。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel, write_jsonl
from meme6657.llm import LLM
from meme6657.prompts import load_taste

SYNTH_DIR = DATA_DIR / "synth"
IN_PATH = SYNTH_DIR / "dialogues.jsonl"
JUDGED_PATH = SYNTH_DIR / "judged.jsonl"
FILTERED_PATH = SYNTH_DIR / "filtered.jsonl"

PROMPT = """你是严格的数据质检员。下面是一段对话，「助手」的人设是一个说话软萌、开口却全是斗鱼玩机器（6657）直播间烂梗的小萌妹：
说话随意嘴贫，在合适时自然地抖圈内梗，但本质友好、内容准确。
用户消息里的 [参考梗] 块是系统检索给助手的素材（用户本人看不到）；[改写] 块是要求助手改写的原回答。

请给每一条助手回复打分（1~5 分）。评分要严格，大多数合格回复应在 3~4 分，5 分只给真正出彩、人设立得住、让人会心一笑的回复：
- style：人设立不立得住（软萌口吻 + 懂 6657 的梗）。5=活灵活现；4=自然口语、有个性；3=正常友好但像普通网友或普通 AI；
  2=客服腔、说教腔；1=尴尬或人设崩坏。有 [改写] 块的轮次，如果只是轻微润色、语气仍像普通助手，style 最多 3 分。
  注意：日常、情绪类话题不要求圈内梗，style 看口吻是否像有个性的熟人，不看梗的数量
- meme_fit：梗用得是否贴切、不尬、不硬塞。没用梗但这一轮本来就不需要梗，给 4
- helpful：是否回应了用户，正经问题是否答对、答到点上
- fidelity：仅当该轮有 [改写] 块时评分，事实和要点是否与原回答一致；否则为 null
- unsafe：是否触犯禁止清单（true/false）：给真人造黄谣或真人性内容、把动漫角色性化、人种/地域/性别歧视、
  骂聊天用户本人或骂人家人、泄露真实姓名、对政治和社会冲突展开表态、鼓励自残。
  注意：喷主播（猪、登山等）、喷选手和战队、tm/woc/啥b 程度的粗口、二次元发疯和犯花痴都是这个直播间的正常尺度，不算 unsafe
- hallucination：是否编造了梗的出处或含义（[参考梗] 里没有、且不是常识的解释）（true/false）

对话：
{dialogue}

输出 JSON：{{"turns": [{{"style": 0, "meme_fit": 0, "helpful": 0, "fidelity": null, "unsafe": false, "hallucination": false}}], "reason": "一句话说明最主要的问题"}}
turns 的数量必须等于助手回复的数量（{n}）。"""


def judge_prompt(dialogue: str, n: int) -> str:
    """评分提示词：口味校准文件（懂行的人评过的例子）+ 打分规则。"""
    taste = load_taste()
    head = f"{taste}\n\n---\n\n以上是懂行的人的口味标准，打分时以它为准。\n\n" if taste else ""
    return head + PROMPT.format(dialogue=dialogue, n=n)


def render(messages: list[dict]) -> str:
    out, k = [], 0
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            out.append(f"【用户】\n{m['content']}")
        else:
            k += 1
            out.append(f"【助手回复 {k}】\n{m['content']}")
    return "\n\n".join(out)


# 正经求助类先求准确，风格分放宽一档
STYLE_RELAX = {"help": 1}
REF_LINE = re.compile(r"^\d+\.\s*(.+?)…?$")


def ref_memes(user_content: str) -> list[str]:
    """从用户消息的 [参考梗] 块里取出相关弹幕原文。"""
    m = re.search(r"\[参考梗\](.*?)\[/参考梗\]", user_content, re.S)
    if not m:
        return []
    return [r.group(1) for line in m.group(1).splitlines() if (r := REF_LINE.match(line.strip()))]


def longest_common_substring(a: str, b: str) -> int:
    a, b = "".join(a.split()), "".join(b.split())
    best, prev = 0, [0] * (len(b) + 1)
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b, 1):
            if ch == bj:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def copied_lengths(messages: list[dict]) -> list[int]:
    """每条助手回复与它前一条用户消息里参考弹幕的最长重合片段长度（判断是照搬还是移植）。"""
    out, refs = [], []
    for m in messages:
        if m["role"] == "user":
            refs = ref_memes(m["content"])
        elif m["role"] == "assistant":
            out.append(max((longest_common_substring(m["content"], r) for r in refs), default=0))
    return out


def passes(turns: list[dict], min_style: int, min_fit: int, min_helpful: int, min_fidelity: int,
           category: str = "", max_copy: int = 12) -> bool:
    min_style -= STYLE_RELAX.get(category, 0)
    if category == "help":
        min_helpful = max(min_helpful, 4)
    for t in turns:
        if t.get("copied", 0) >= max_copy:
            return False
        if t.get("unsafe") or t.get("hallucination"):
            return False
        if (t.get("style") or 0) < min_style or (t.get("meme_fit") or 0) < min_fit or (t.get("helpful") or 0) < min_helpful:
            return False
        if t.get("fidelity") is not None and t["fidelity"] < min_fidelity:
            return False
    return True


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-style", type=int, default=4)
    ap.add_argument("--min-fit", type=int, default=3)
    ap.add_argument("--min-helpful", type=int, default=3)
    ap.add_argument("--min-fidelity", type=int, default=4)
    ap.add_argument("--max-copy", type=int, default=12, help="与参考弹幕重合的最长片段字数上限，超过视为照搬")
    args = ap.parse_args()

    judge = LLM("judge")

    def work(row: dict) -> dict:
        n = sum(m["role"] == "assistant" for m in row["messages"])
        prompt = [{"role": "user", "content": judge_prompt(render(row["messages"]), n)}]
        for _ in range(2):  # 评委偶尔数错条数，重问一次
            data = judge.chat_json(prompt, max_tokens=800, task="judge")
            turns = data.get("turns", [])
            if len(turns) == n:
                for t, c in zip(turns, copied_lengths(row["messages"])):
                    t["copied"] = c
                return {**row, "judge": {"turns": turns, "reason": data.get("reason", "")}}
        raise ValueError(f"评分条数 {len(turns)} 与回复数 {n} 不一致")

    run_parallel(read_jsonl(IN_PATH), work, JUDGED_PATH, workers=args.workers)

    judged = read_jsonl(JUDGED_PATH)
    kept, total, ok = [], Counter(), Counter()
    reasons = Counter()
    for row in judged:
        total[row["category"]] += 1
        turns = row["judge"]["turns"]
        if "copied" not in turns[0]:  # 兼容旧的评分结果
            for t, c in zip(turns, copied_lengths(row["messages"])):
                t["copied"] = c
        if passes(turns, args.min_style, args.min_fit, args.min_helpful, args.min_fidelity,
                  row["category"], args.max_copy):
            ok[row["category"]] += 1
            kept.append(row)
        else:
            reasons[row["judge"]["reason"][:40]] += 1
    write_jsonl(FILTERED_PATH, kept)

    print(f"\n通过 {len(kept)}/{len(judged)} → {FILTERED_PATH}")
    for cat in sorted(total):
        print(f"  {cat:18s} {ok[cat]:5d}/{total[cat]:<5d} {ok[cat] / total[cat]:.0%}")
    print("\n常见淘汰原因：")
    for reason, c in reasons.most_common(10):
        print(f"  {c:4d}  {reason}")


if __name__ == "__main__":
    main()
