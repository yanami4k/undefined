from scripts.normal_replies import MemeIndex, problems, quoted_spans, soft_problems, tidy_tics

MEME = "爽！😎这才叫音乐🎶！😋👍爽！😎这才叫音乐🎶！"


def test_quoted_meme_keeps_emoji_and_tics():
    reply = "哈哈😂 爽！😎这才叫音乐🎶！😋👍 祥子也想听喵～ 好听desuwa"
    out = tidy_tics("这歌好听", reply, [MEME])
    assert "爽！😎这才叫音乐🎶！😋👍" in out      # 弹幕原样
    assert "😂" not in out and "～" not in out     # 弹幕以外的 emoji 和 ～ 被清理
    assert out.count("喵") + out.count("desuwa") == 1


def test_meme_with_desuwa_untouched():
    meme = "✋停一下desuwa，全世界坠可爱的宝宝丰川祥子粗现o（=·w〈=)☆"
    out = tidy_tics("祥子来了", meme + " 说的就是祥子本人喵😊", [meme])
    assert out.startswith(meme)


def test_no_memes_plain_cleanup():
    assert tidy_tics("今天好累", "辛苦啦🥺 先睡一觉喵 好梦desuwa") == "辛苦啦 先睡一觉喵 好梦"


def test_quoted_spans_ignores_short_overlap():
    assert quoted_spans("这才叫", [MEME]) == []


def test_range_tilde_becomes_dao():
    assert tidy_tics("充能堆多少", "副 C 常见 180%～220% 喵～") == "副 C 常见 180%到220% 喵"


def test_kaomoji_removed_but_normal_parentheses_kept():
    out = tidy_tics("好累", "（歪头想了想）先按 (Ctrl+P) 试试喵 (｡•́︿•̀｡) Σ(っ °Д °;)っ ▜▜▙")
    assert out == "（歪头想了想）先按 (Ctrl+P) 试试喵"


def test_kept_quote_emoji_survives():
    meme = "可是宗宗🥺。。。也太菜了吧😅"
    out = tidy_tics("宗宗又空枪了", "祥子只想说：" + meme, keep=["可是宗宗🥺。。。也太菜了吧😅"])
    assert out.endswith(meme)


INDEX = MemeIndex(["可是宗宗🥺。。。也太菜了吧😅", "主播能每句话后都加个desuwa吗", "Vitality 冠军"])


def test_quotes_finds_real_meme():
    assert INDEX.quotes("宗宗又空枪了", "祥子只想说：可是宗宗🥺。。。也太菜了吧😅")


def test_quotes_ignores_tics_english_and_echo():
    assert INDEX.quotes("聊聊", "今天 Vitality 冠军desuwa") == []
    assert INDEX.quotes("宗宗也太菜了吧", "对，宗宗也太菜了吧") == []


def test_chat_without_meme_passes():
    assert not problems("下雨了", "祥子也不想出门喵", [], "daily", [])
    assert not soft_problems("下雨了", "祥子也不想出门喵", [], [])


def test_wo_is_soft_and_quotes_excluded():
    assert soft_problems("token 快没了", "我觉得你对 claude 就不会这样说喵", [], [])
    meme = "可是我🥺。。。也太菜了吧😅"
    assert not soft_problems("又输了", "祥子只想说：" + meme, [], [meme])


def test_strip_followup_after_answer():
    from scripts.normal_replies import strip_followup
    reply = "打两天还没刷完，祥子觉得不是效率低，是这活动本来就又臭又长喵。你先说说是哪种活动，祥子帮你看看是不是漏了什么速刷的技巧。"
    assert strip_followup(reply) == "打两天还没刷完，祥子觉得不是效率低，是这活动本来就又臭又长喵。"


def test_keep_request_for_missing_material():
    from scripts.normal_replies import strip_followup
    assert strip_followup("代码发来给祥子看看喵，先别急，祥子陪你一起找。") == "代码发来给祥子看看喵，先别急，祥子陪你一起找。"
    short = "祥子来帮你看看。你先把 demo 发过来，祥子帮你看一下站位喵。"
    assert strip_followup(short) == short


def test_followup_not_triggered_by_fahui():
    from scripts.normal_replies import pushes_topic
    assert not pushes_topic("不过就他最近的发挥，祥子只想说：可是宗宗也太菜了吧")
    assert pushes_topic("祥子觉得这几个都稳。要详细配队的话告诉祥子你打哪个肉鸽")
