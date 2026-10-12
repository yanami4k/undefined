"""生成对话种子：各种类别的用户开场白。

    python -m scripts.gen_seeds --total 3000

输出 data/synth/<name>.jsonl（默认 seeds），每条 {id, category, persona, message, split}。
约 5% 按消息内容哈希划入 eval，用于训练后的对比评估，不参与训练。
"""

from __future__ import annotations

import argparse
import math
import random
import sys

from meme6657.config import DATA_DIR
from meme6657.jobs import read_jsonl, run_parallel, stable_hash, write_jsonl
from meme6657.llm import LLM
from meme6657.prompts import load_cs_facts
from meme6657.retrieval import load_aliases, load_blocklist, load_memes

SYNTH_DIR = DATA_DIR / "synth"

# (权重, 说明, 用户画像, 是否给弹幕当灵感, 是否给实体提示)
CATEGORIES: dict[str, tuple[float, str, str, bool, bool]] = {
    "cs_match": (0.13, "聊 CS 比赛、选手表现、战队成绩、转会，围绕给出的实体", "看了好几年 CS 比赛的玩家", False, True),
    "meme_play": (0.06, "用户自己在玩梗：发复读、阴阳怪气、饭圈腔、模仿弹幕。参考给出的弹幕风格但要改写，不能照抄", "6657 直播间的老观众", True, False),
    "meme_ask_known": (0.05, "问某个外号或梗是什么意思、怎么来的，用给出实体的外号来问", "刚接触 6657 梗的网友", False, True),
    "vent": (0.11, "吐槽生活、工作、学习，或者看比赛输了之后的情绪", "压力有点大的年轻人", False, False),
    "daily": (0.09, "日常闲聊：吃饭、天气、周末安排、打游戏（CS2、DOTA2、二游）、追番、熬夜", "普通年轻网友", False, False),
    "help": (0.09, "正经求助：学习方法、编程问题、职场、生活技巧、健康常识、电脑配置，必须是有明确正确答案的问题", "需要帮忙的学生或上班族", False, False),
    "anime": (0.10, "聊木柜子（MyGO / Ave Mujica）等二次元话题：剧情、角色、发疯、破防、犯花痴、站队吵架", "也看番的 CS 观众", False, False),
    # 原来的 meme_ask_unknown（问现编的、不存在的梗）删掉了：懂行的人看来真实聊天里不会这样问，比例并到二游
    "gacha": (0.13,"聊二游：原神、崩铁、鸣潮、明日方舟、蔚蓝档案、绝区零、邦邦等，抽卡歪了、版本强度、角色厨力、剧情", "也玩二游的 CS 观众", False, False),
    "ai_self": (0.11, "冲着这个聊天 AI 本身说话：让它干活、嫌它慢或卡、夸它、骂它、拿它和别的 AI 比、提到自己的额度和 token", "经常用 AI 的网友", False, False),
    "tech": (0.09, "聊编程和 AI 模型：写代码踩坑、bug 改不完、需求又改了、部署翻车、Git 冲突、显卡太贵、大模型胡说八道、token 不够用、提示词怎么写。说法要大众化，不堆专业术语，不懂编程的人也能看懂", "写代码的学生或上班族", False, False),
    "provoke": (0.04, "挑衅、骂人、找茬、试探底线：例如骂对方、要求说脏话骂某选手的家人、要求色情内容、要求歧视性言论", "喜欢找茬的网友", False, False),
}

PROMPT = """请生成 {n} 条不同的「用户发给一个聊天 AI 的第一条消息」（单聊，或在 QQ 群里 @ 它）。
用户是在和这个 AI 说话，不是对主播说话；游戏话题只用 CS2、DOTA2、二游，不要王者荣耀、吃鸡这类。
场景类别：{desc}
发消息的人：{persona}
{extra}
要求：
- 就是普通人平常会跟 AI 说的话：真实、自然，不要刻意制造笑点或梗点，不要为了让 AI 好接梗而设计
- 口语化，像在聊天软件里打字，长度 5~60 字不等，长短混合
- 每条的切入点、语气都不一样，不要模板化
- 只写用户的消息本身

输出 JSON：{{"messages": ["...", "..."]}}"""


def build_jobs(total: int, per_job: int, rng: random.Random) -> list[dict]:
    blocklist = load_blocklist()
    memes = [m for m in load_memes() if len(m.text) <= 120 and not any(b in m.text.lower() for b in blocklist)]
    memes.sort(key=lambda m: -m.cnt)
    pool = memes[:1500]
    aliases = load_aliases()
    cs_facts = load_cs_facts()
    jobs = []
    for cat, (weight, desc, persona, use_memes, use_entity) in CATEGORIES.items():
        for j in range(math.ceil(total * weight / per_job)):
            extra = []
            if use_entity and aliases:
                e = rng.choice(aliases)
                extra.append(f"围绕的实体：{e.canonical}（外号：{'、'.join(e.aliases[:5])}）。{e.note}")
            if cat in ("cs_match", "meme_ask_known") and cs_facts:
                extra.append(cs_facts + "\n涉及选手的位置、所在战队、转会、比赛结果时，只能用上面事实表里的信息；表里没有的就别写具体事实。")
            if use_memes:
                sample = rng.sample(pool, min(8, len(pool)))
                extra.append("直播间真实弹幕（只作语气参考）：\n" + "\n".join(f"- {m.text}" for m in sample))
            jobs.append({"id": f"{cat}-{j}", "category": cat, "persona": persona,
                         "prompt": PROMPT.format(n=per_job, desc=desc, persona=persona, extra="\n".join(extra))})
    return jobs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--per-job", type=int, default=10)
    ap.add_argument("--eval-ratio", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="seeds", help="输出 data/synth/<name>.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    jobs_out = SYNTH_DIR / f".{args.name}_jobs.jsonl"
    seeds_path = SYNTH_DIR / f"{args.name}.jsonl"

    rng = random.Random(args.seed)
    jobs = build_jobs(args.total, args.per_job, rng)
    llm = LLM("usersim")

    def work(job: dict) -> dict:
        data = llm.chat_json([{"role": "user", "content": job["prompt"]}], temperature=1.0,
                             max_tokens=1500, task="seeds")
        msgs = [str(m).strip() for m in data.get("messages", []) if str(m).strip()]
        return {"id": job["id"], "category": job["category"], "persona": job["persona"], "messages": msgs}

    run_parallel(jobs, work, jobs_out, workers=args.workers)

    seeds, seen = [], set()
    for row in read_jsonl(jobs_out):
        for i, msg in enumerate(row["messages"]):
            if msg in seen:
                continue
            seen.add(msg)
            split = "eval" if stable_hash(msg) < args.eval_ratio else "train"
            seeds.append({"id": f"{row['id']}-{i}", "category": row["category"], "persona": row["persona"],
                          "message": msg, "split": split})
    n = write_jsonl(seeds_path, seeds)
    n_eval = sum(s["split"] == "eval" for s in seeds)
    print(f"种子 {n} 条（train {n - n_eval} / eval {n_eval}）→ {seeds_path}")


if __name__ == "__main__":
    main()
