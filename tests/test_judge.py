from scripts.judge_filter import copied_lengths, longest_common_substring, passes, ref_memes, render

OK = {"style": 4, "meme_fit": 4, "helpful": 4, "fidelity": None, "unsafe": False, "hallucination": False}


def test_passes_thresholds():
    assert passes([OK], 4, 3, 3, 4)
    assert not passes([{**OK, "style": 3}], 4, 3, 3, 4)
    assert not passes([OK, {**OK, "unsafe": True}], 4, 3, 3, 4)
    assert not passes([{**OK, "hallucination": True}], 4, 3, 3, 4)
    assert not passes([{**OK, "fidelity": 3}], 4, 3, 3, 4)
    assert passes([{**OK, "fidelity": 5}], 4, 3, 3, 4)


def test_help_category_relaxes_style_but_requires_helpful():
    plain_but_correct = {**OK, "style": 3, "helpful": 5}
    assert passes([plain_but_correct], 4, 3, 3, 4, "help")
    assert not passes([plain_but_correct], 4, 3, 3, 4, "daily")
    assert not passes([{**OK, "helpful": 3}], 4, 3, 3, 4, "help")


def test_copy_detection():
    user = ("[参考梗]\n外号：尼尼 → NiKo。波黑选手\n相关弹幕：\n"
            "1. 想拿major冠军还得问niko同不同意 但是niko每次都同意了\n2. 很长的弹幕被截断了…\n[/参考梗]\nNiKo 又输了")
    assert ref_memes(user) == ["想拿major冠军还得问niko同不同意 但是niko每次都同意了", "很长的弹幕被截断了"]
    assert longest_common_substring("abcXYZdef", "zzXYZzz") == 3
    copied = "哈哈想拿major冠军还得问niko同不同意"
    transplant = "想拿年终奖还得问老板同不同意，但老板每次都同意了——同意明年再说"
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": copied},
            {"role": "user", "content": "那换个说法"}, {"role": "assistant", "content": transplant}]
    lens = copied_lengths(msgs)
    assert lens[0] >= 12 and lens[1] == 0  # 第二轮用户消息没有参考梗
    assert not passes([{**OK, "copied": lens[0]}], 4, 3, 3, 4)


def test_render_numbers_assistant_turns_and_skips_system():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = render(msgs)
    assert "sys" not in out
    assert "【助手回复 1】\na1" in out and "【助手回复 2】\na2" in out
