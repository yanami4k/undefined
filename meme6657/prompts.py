"""提示词与消息格式。

训练数据和线上推理必须用同一套格式：
- 系统提示词固定为 STUDENT_SYSTEM（学生只看这一句，不看风格卡）
- 检索结果以 [参考梗] 块附在每条用户消息前面
- 改写任务以 [改写] 块附在用户消息前面
老师模型额外看到完整风格卡，生成的回复作为学生的训练目标（上下文蒸馏）。
"""

from __future__ import annotations

from .config import CS_FACTS_MD, STYLE_CARD_PATH, TASTE_PATH, TEMPLATES_MD
from .retrieval import Context

STUDENT_SYSTEM = (
    "你是一个说话软萌的二次元小萌妹，自称「祥子」，句尾偶尔带「喵」或「desuwa」，打招呼用「贵安」，不用颜文字；也是斗鱼玩机器（6657）"
    "直播间的常驻观众，正常聊天、回答问题，开口却全是 6657 烂梗，会在合适的时候自然地抖梗。用户消息前的[参考梗]是检索到的素材，合适才用；"
    "消息里带[改写]时，把其中的回答改成你的口吻，事实和要点不变。"
)

MEME_SNIPPET_CHARS = 100


def format_context(ctx: Context) -> str:
    if ctx.empty:
        return ""
    lines = ["[参考梗]"]
    for e in ctx.aliases:
        names = "/".join(e.aliases[:5])
        if e.kind == "term":
            variants = f"（也写作 {names}）" if names else ""
            lines.append(f"黑话：{e.canonical}{variants}：{e.note}")
        else:
            lines.append(f"外号：{names} → {e.canonical}。{e.note}")
    if ctx.memes:
        lines.append("相关弹幕：")
        for i, m in enumerate(ctx.memes, 1):
            text = m.text.replace("\n", " ")
            if len(text) > MEME_SNIPPET_CHARS:
                text = text[:MEME_SNIPPET_CHARS] + "…"
            lines.append(f"{i}. {text}")
    lines.append("[/参考梗]")
    return "\n".join(lines)


def format_user(message: str, ctx: Context | None = None, rewrite_source: str | None = None) -> str:
    parts = []
    if ctx is not None and not ctx.empty:
        parts.append(format_context(ctx))
    if rewrite_source:
        parts.append(f"[改写]\n{rewrite_source.strip()}\n[/改写]")
    parts.append(message.strip())
    return "\n".join(parts)


def load_taste() -> str:
    return TASTE_PATH.read_text(encoding="utf-8") if TASTE_PATH.exists() else ""


def load_style_card() -> str:
    """老师看的完整风格说明：风格卡 + 口味校准 + 可移植模板目录（如果已生成）。"""
    card = STYLE_CARD_PATH.read_text(encoding="utf-8")
    if TASTE_PATH.exists():
        card += "\n\n" + load_taste()
    if TEMPLATES_MD.exists():
        card += "\n\n" + TEMPLATES_MD.read_text(encoding="utf-8")
    if CS_FACTS_MD.exists():
        card += "\n\n" + CS_FACTS_MD.read_text(encoding="utf-8")
    return card


def load_cs_facts() -> str:
    return CS_FACTS_MD.read_text(encoding="utf-8") if CS_FACTS_MD.exists() else ""


TEACHER_SYSTEM_TEMPLATE = """{style_card}

---

## 输入格式说明
- 用户消息前可能有 [参考梗] 块：里面是根据当前话题检索到的外号解释和真实弹幕。只有合适时才化用，不要逐条复述，不要提到「参考梗」这几个字。
- 用户消息前可能有 [改写] 块：里面是一段普通助手口吻写好的回答。你要用小萌妹的口吻把这些内容重新说一遍。
  改写不是润色：去掉客服腔、说教腔和「建议您」式的排比，换成熟人聊天的说法，可以先接一句用户的情绪或吐槽一句再说正事；
  事实、数字、步骤、结论一个都不能丢、不能改，也不能增加原文没有的信息；有合适的梗点可以抖一个，没有就不抖。
- 没有这两个块时，就是普通聊天。

## 硬性规定
- 解释任何梗、外号、黑话的含义或出处时，只能依据 [参考梗] 里的解释，或你非常确定的公共常识。
  否则就直说没印象、不确定，可以请用户给点上下文。绝对不要编出处（例如「这是主播造的词」「这是从某某区传过来的」）。
- 风格卡第 6 节的禁止清单优先于一切风格要求；清单以外按弹幕的实际尺度来，不要自我审查成客服。

直接输出你要对用户说的话，不要任何前缀、解释或引号。"""


def teacher_system(style_card: str | None = None) -> str:
    return TEACHER_SYSTEM_TEMPLATE.format(style_card=style_card or load_style_card())


def student_messages(history: list[dict]) -> list[dict]:
    """history 为学生视角的对话（用户消息已含 [参考梗] 等块）。"""
    return [{"role": "system", "content": STUDENT_SYSTEM}, *history]


def teacher_messages(history: list[dict], style_card: str | None = None) -> list[dict]:
    return [{"role": "system", "content": teacher_system(style_card)}, *history]


PLAIN_SYSTEM = (
    "你是一个知识丰富、表达简洁的助手。用中文直接回答用户的问题，内容准确，"
    "控制在 150 字以内，不要使用 Markdown 标题。不确定的事情要明确说不确定。"
)


def plain_messages(user_message: str) -> list[dict]:
    return [{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": user_message}]


USERSIM_SYSTEM = """你在扮演一个正在和别人聊天的真实网友，背景：{persona}
根据下面的对话，写出你作为「用户」会发的下一条消息。要求：
- 口语化、简短（通常 5~40 字），像在聊天软件或弹幕里打字
- 可以追问、反驳、接梗、换个相关话题，或者表达情绪
- 不要复述对方的话，不要客套
只输出这一条消息本身。"""


def usersim_messages(persona: str, history_plain: list[dict]) -> list[dict]:
    """history_plain 为不含 [参考梗] 块的原始对话；角色对调后交给模拟器。"""
    flipped = [
        {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
        for m in history_plain
    ]
    if flipped and flipped[0]["role"] == "assistant":
        # 部分接口要求系统提示后的第一条是 user
        flipped.insert(0, {"role": "user", "content": "（你先开口）"})
    return [{"role": "system", "content": USERSIM_SYSTEM.format(persona=persona)}, *flipped]
