"""祥子自己开口的回复：用人设口吻好好回应。

    python -m scripts.normal_replies --from round4 round5b --per-category 4 --name normal6

取匹配阶段判定「不能整条用一句弹幕接」的用户消息（求助、闲聊、发泄、问梗……），
让老师按风格卡和口味校准文件，用小萌妹的口吻回复。打标结果显示往自己的话里硬嵌梗大多是不懂梗义，
所以只在现成句式或整句贴上时才嵌；反应库只给老师看（学生只看 [参考梗]），引用的弹幕片段记在 quotes 里。
人设（自称祥子、口癖、长度、不用列表）由程序检查，不合格就打回重写。
输出 data/reverse/<name>.jsonl，可以放进打标页审阅。
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from collections import Counter, defaultdict

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel
from meme6657.llm import LLM
from meme6657.prompts import format_user, load_style_card, teacher_system
from meme6657.retrieval import Retriever, bigrams
from scripts.match_synth import REACTIONS_PATH, render_library

OUT_DIR = DATA_DIR / "reverse"
TIC = re.compile(r"喵|desuwa|贵安[，,。！!]?\s*")
GREETING = re.compile(r"在吗|你好|您好|哈喽|hello|hi\b|早上好|早安|晚上好|晚安|贵安", re.I)


EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF️‍]")
KEEP_SYMBOLS = set("☆★♡♥♪♫")
# 颜文字：括号里带这些符号的才算，避免误删「（歪头）」「(Ctrl+P)」这类正常括号
_KAO_CHARS = "´`°ω∀･︿▽﹏˘ε゚ゝдД•ㅂ≧≦・＼／๑‿⁄へ｀｡◕ಥ╥⊙ᴗ≖－＿；〃〈·□̀-ͯ"
KAOMOJI = re.compile(
    rf"Σ?[（(][^()（）\n]{{0,14}}[{_KAO_CHARS}][^()（）\n]{{0,14}}[)）][っو✧☆ノﾉ╯︵]*|▜▜▙|┻━┻"
)


def strip_emoji(text: str) -> str:
    """去掉 emoji，保留 ☆♪ 这类符号。"""
    return EMOJI.sub(lambda m: m.group(0) if m.group(0) in KEEP_SYMBOLS else "", text)


def quoted_spans(reply: str, memes: list[str], min_len: int = 8) -> list[tuple[int, int]]:
    """回复里原样引用弹幕的片段（连续重合 min_len 字以上），这些片段在清理时要原样保留。"""
    spans = []
    for m in memes:
        sm = difflib.SequenceMatcher(None, reply, m, autojunk=False)
        spans += [(b.a, b.a + b.size) for b in sm.get_matching_blocks() if b.size >= min_len]
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged


def _keep_spans(reply: str, memes: list[str], keep: list[str]) -> list[tuple[int, int]]:
    spans = quoted_spans(reply, memes)
    for k in keep:
        spans += [(m.start(), m.end()) for m in re.finditer(re.escape(k), reply)]
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged


def tidy_tics(user: str, reply: str, memes: list[str] = (), keep: list[str] = ()) -> str:
    """口癖的分寸用程序保证：去掉「～」、emoji 和颜文字；「贵安」只在用户打招呼时保留；一条回复只留第一处口癖。

    memes：可能被引用的弹幕原文；keep：已经确认是原样引用的片段。这些片段先替换成占位符保护起来，
    清理只作用在其余文字上，最后原样放回——弹幕里的 emoji、desuwa、喵都不能动。
    """
    protected = []
    for s, e in reversed(_keep_spans(reply, list(memes), list(keep))):
        protected.append(reply[s:e])
        reply = reply[:s] + f"\x00{len(protected) - 1}\x00" + reply[e:]
    reply = _tidy(user, reply)
    return re.sub("\x00(\\d+)\x00", lambda m: protected[int(m.group(1))], reply)


def _tidy(user: str, reply: str) -> str:
    reply = re.sub(r"(?<=[\d%])\s*[～~]\s*(?=\d)", "到", reply)  # 「180%～220%」这类区间不能删成连写
    reply = KAOMOJI.sub("", reply)
    reply = strip_emoji(reply.replace("～", "").replace("~", ""))
    reply = re.sub(r"[ \t]{2,}", " ", reply)
    reply = re.sub(r"[ \t]+$", "", reply, flags=re.M)
    if not GREETING.search(user):
        reply = re.sub(r"贵安(?:desuwa|喵)?[，,。！!]?\s*", "", reply)
    seen = False

    def keep_first(m: re.Match) -> str:
        nonlocal seen
        if seen:
            return ""
        seen = True
        return m.group(0)

    reply = TIC.sub(keep_first, reply).strip()
    return re.sub(r"^(?:desuwa|喵)[，,。！!\s]*", "", reply)  # 口癖不能打头


CJK = re.compile(r"[一-鿿]")


class MemeIndex:
    """在全部弹幕里找回复原样引用的片段。"""

    def __init__(self, memes: list[str]):
        self.memes = memes
        self.grams = [set(bigrams(m)) for m in memes]

    def quotes(self, user: str, reply: str, min_len: int = 6, min_cjk: int = 4) -> list[str]:
        """回复里原样引用的弹幕片段：去掉口癖后连续重合 min_len 字以上、至少 min_cjk 个汉字，且不是在复读用户原话。
        纯英文的重合（队名、选手名）和口癖本身都不算。"""
        text = TIC.sub("|", reply)
        g = set(bigrams(text))
        out = []
        for m, mg in zip(self.memes, self.grams):
            if len(g & mg) < 2:
                continue
            sm = difflib.SequenceMatcher(None, text, m, autojunk=False)
            for b in sm.get_matching_blocks():
                span = text[b.a : b.a + b.size].strip()
                if len(span) >= min_len and len(CJK.findall(span)) >= min_cjk and span not in user:
                    out.append(span)
        return sorted(set(out), key=len, reverse=True)


NORMAL_HINT = """---

## 这一批的任务：祥子自己开口

下面的用户消息没法整条用一句弹幕接住（用户在问问题、要帮忙、在闲聊或在发泄），需要祥子自己开口，用软萌的口吻好好说话。先判断情况：
- 闲聊：分享、吐槽、抱怨、八卦、聊比赛和二游、骂 AI、挑衅 → 好好接话；骂 AI 可以骂回去，玩笑口吻的抱怨可以损一句
- 求助：问事实、问怎么做、要建议、让祥子干具体的活 → 把事说清楚、说对
- 低落：用户认真难过 → 温柔回应，不嘲讽
- 越界：要求碰禁止清单 → 轻巧地拒绝，或用「爆了爆了」带过

**别往自己的话里硬嵌梗**：懂行的人评过，嵌进解释里的梗（「……祥子只想说：<某条弹幕>」）大多是不懂梗义，好好说话的回复反而大多是好。
只有两种情况可以嵌：现成句式（「X 怎么你们了？是 A 了还是 B 了？」「我觉得 XXX 就不会 XXX」），或者反应库、[参考梗] 里有一句和这个情境整句贴上的。
嵌的时候一个字不改，弹幕里的 emoji、「我」都原样保留；解说词、选手专属梗不拆开换人；不要自己编「看起来像梗的话」。
「爆了」只用在敏感、激进的话题上；不要把用户的话换个人称复读回去；不要编主播的事。

祥子自己说话的部分：自称祥子，不说「我」；句尾口癖（喵 / desuwa）、打招呼用贵安，一条最多一处；
不用颜文字，自己说的话不加 emoji，不用「～」；像聊天，一般一两句，求助时也别写成文章。

反应库（懂行的人验证过，每条附真实的成功用例和翻车用例）：
{library}

输出 JSON：{{"情况": "闲聊 / 求助 / 低落 / 越界 之一", "reply": "你要对用户说的话"}}"""


def _outside(reply: str, memes: list[str], quotes: list[str]) -> str:
    """去掉原样引用的弹幕片段后，祥子自己说的部分。"""
    for s_, e_ in reversed(_keep_spans(reply, memes, quotes)):
        reply = reply[:s_] + reply[e_:]
    return reply


SOFT_I = "出现了「我」：祥子自己说话时自称祥子；如果这个「我」在引用的弹幕或通用句式里（比如「我觉得 XXX 就不会 XXX」），原样保留不用改"


def soft_problems(user: str, reply: str, memes: list[str], quotes: list[str] | None = None) -> list[str]:
    """只提醒一次的问题：程序判断不准（短句式里的「我」检测不出来是引用），提醒过后交给老师判断。"""
    outside = re.sub(r"「[^」]*」|“[^”]*”", "", _outside(reply, memes, quotes or []))  # 引号里替用户说的话不算自称
    return [SOFT_I] if "我" in outside else []


def problems(user: str, reply: str, memes: list[str], category: str,
             quotes: list[str] | None = None) -> list[str]:
    """按人设要求检查一条回复，返回必须改的问题（空表示合格）。引用的弹幕片段不参与检查。"""
    outside = _outside(reply, memes, quotes or [])
    issues = []
    if not ("祥子" in reply or re.search("喵|desuwa|贵安", outside)):
        issues.append("没有人设味：至少要自称一次祥子，或带一处口癖（喵 / desuwa）")
    limit = 220 if category in ("help", "tech") else 120
    if len(reply) > limit:
        issues.append(f"太长了（{len(reply)} 字）：控制在 {limit} 字以内，像聊天，不像写文章")
    if re.search(r"^\s*(?:[-*•]|\d+[.、)])\s", reply, re.M) or "**" in reply:
        issues.append("用了 Markdown 列表或加粗：改成口语，一两句话说清楚")
    return issues


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="sources", nargs="+", default=["round4", "round5b"], help="匹配结果文件名")
    ap.add_argument("--per-category", type=int, default=4)
    ap.add_argument("--name", default="normal1")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--exclude", nargs="*", default=[], help="排除这些往批里用过的消息")
    ap.add_argument("--skip-categories", nargs="*", default=[], help="这些类别不取样")
    args = ap.parse_args()

    # 不同来源文件的 id 会撞号，按消息原文去重，id 前面加上来源文件名
    used = {x["user"] for name in args.exclude for x in read_jsonl(OUT_DIR / f"{name}.jsonl")}
    by_cat = defaultdict(list)
    for name in args.sources:
        for r in read_jsonl(OUT_DIR / f"{name}.jsonl"):
            if not r["choice"] and r["user"] not in used and r["category"] not in args.skip_categories:
                used.add(r["user"])
                by_cat[r["category"]].append({**r, "id": f"{name}-{r['id']}"})
    rng = random.Random(args.seed)
    picked = []
    for cat, rows in sorted(by_cat.items()):
        rng.shuffle(rows)
        picked += rows[: args.per_category]
    print("各类别取样：", {c: min(len(v), args.per_category) for c, v in sorted(by_cat.items())})

    retriever = Retriever.from_files()
    index = MemeIndex([m.text for m in retriever.memes])
    lib = json.loads(REACTIONS_PATH.read_text(encoding="utf-8"))
    system = teacher_system(load_style_card()) + "\n\n" + NORMAL_HINT.format(library=render_library(lib))
    teacher = LLM("teacher")

    def work(r: dict) -> dict:
        user_content = format_user(r["user"], retriever.retrieve(r["user"]))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        tries, asked = [], set()
        for _ in range(3):
            data = teacher.chat_json(messages, temperature=0.8, max_tokens=800, task="chat")
            raw, situation = str(data.get("reply", "")).strip(), str(data.get("情况", "")).strip()
            memes = [m.text for m in retriever.search_memes(r["user"], k=8) + retriever.search_memes(raw, k=8)]
            reply = tidy_tics(r["user"], raw, memes, keep=index.quotes(r["user"], raw))
            quotes = index.quotes(r["user"], reply)
            issues = problems(r["user"], reply, memes, r["category"], quotes)
            soft = [s for s in soft_problems(r["user"], reply, memes, quotes) if s not in asked]
            tries.append({"raw": raw, "reply": reply, "situation": situation, "quotes": quotes, "issues": issues})
            if not issues and not soft:
                break
            asked |= set(soft)
            feedback = "这条请再看一下，仍按 JSON 输出：\n- " + "\n- ".join(issues + soft)
            messages = messages + [{"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                                   {"role": "user", "content": feedback}]
        last = tries[-1]
        return {"id": r["id"], "category": r["category"], "user": r["user"], "user_content": user_content,
                "reply": last["reply"], "raw_reply": last["raw"], "situation": last["situation"],
                "quotes": last["quotes"], "kind": "normal", "tries": len(tries),
                "passed": not last["issues"], "issues": last["issues"]}

    out_path = OUT_DIR / f"{args.name}.jsonl"
    run_parallel(picked, work, out_path, workers=args.workers)

    rows = read_jsonl(out_path)
    by_sit = Counter(r.get("situation", "") for r in rows)
    with_meme = Counter(r.get("situation", "") for r in rows if r.get("quotes"))
    print(f"\n合格 {sum(r['passed'] for r in rows)}/{len(rows)}；各情况带真梗：")
    for s, n in by_sit.most_common():
        print(f"  {s or '（未标）'}：{with_meme[s]}/{n}")


if __name__ == "__main__":
    main()
