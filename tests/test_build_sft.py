from scripts.build_sft import cap_repeated_phrases, training_rows


def row(text):
    return {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": text}]}


def test_cap_repeated_phrases_limits_template_replies():
    template = "你截一下弹幕或者说说是哪局我帮你对对"
    rows = [row(f"第{i}条：{template}") for i in range(8)] + [row("完全不一样的回复内容，没有套话")]
    kept, top = cap_repeated_phrases(rows, n=10, cap=3)
    assert sum(template in r["messages"][1]["content"] for r in kept) == 3
    assert any("没有套话" in r["messages"][1]["content"] for r in kept)
    assert top[0][1] == 8


def test_cap_ignores_user_text():
    rows = [{"messages": [{"role": "user", "content": "同样的用户问题同样的用户问题"},
                          {"role": "assistant", "content": f"不同回复{i}"}]} for i in range(6)]
    kept, _ = cap_repeated_phrases(rows, n=10, cap=2)
    assert len(kept) == 6


def test_training_rows_keeps_eval_cards_out():
    gold = [{"id": f"{kind}-{verdict}", "kind": kind, "verdict": verdict}
            for kind in ("meme", "template", "normal", "eval") for verdict in ("good", "wave", "bad")]
    kept = {r["id"] for r in training_rows(gold)}
    assert kept == {f"{kind}-{verdict}" for kind in ("meme", "template", "normal") for verdict in ("good", "wave")}
