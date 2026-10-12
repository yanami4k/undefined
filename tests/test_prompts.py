from meme6657.prompts import (
    STUDENT_SYSTEM,
    format_context,
    format_user,
    student_messages,
    teacher_messages,
    usersim_messages,
)
from meme6657.retrieval import AliasEntry, Context, Meme


def test_format_user_plain_message_unchanged():
    assert format_user("你好", Context()) == "你好"


def test_format_user_with_context_and_rewrite_order():
    ctx = Context([AliasEntry("NiKo", ["尼尼"], note="波黑选手")], [Meme(1, "x" * 150)])
    out = format_user("尼尼呢", ctx, rewrite_source="原回答")
    assert out.index("[参考梗]") < out.index("[改写]") < out.index("尼尼呢")
    assert out.endswith("尼尼呢")
    assert "…" in out  # 长弹幕被截断


def test_term_entry_rendering():
    ctx = Context([AliasEntry("爆了", [], kind="term", note="复读梗")])
    assert "黑话：爆了：复读梗" in format_context(ctx)


def test_student_and_teacher_share_history_but_not_system():
    history = [{"role": "user", "content": "hi"}]
    s = student_messages(history)
    t = teacher_messages(history, style_card="风格卡内容")
    assert s[0]["content"] == STUDENT_SYSTEM
    assert "风格卡内容" in t[0]["content"]
    assert s[1:] == t[1:] == history


def test_usersim_flips_roles_and_starts_with_user():
    plain = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    msgs = usersim_messages("观众", plain)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "b"
