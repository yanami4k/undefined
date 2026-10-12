from meme6657.retrieval import AliasEntry, Meme, Retriever, bigrams, normalize

ALIASES = [
    AliasEntry("NiKo", ["尼尼", "大虾", "🦐"], kind="player", note="波黑选手", tags=["NiKo"]),
    AliasEntry("玩机器", ["机器", "玩神"], kind="streamer", note="主播", exclude=["机器学习", "机器人"]),
    AliasEntry("爆了", [], kind="term", note="直播间复读梗"),
]
MEMES = [
    Meme(1, "Niko：看我用虾须跳绳 🦐", 100, ["NiKo"]),
    Meme(2, "尼尼💚孩孩💚尼尼💚孩孩", 50, ["NiKo"]),
    Meme(3, "机器今天又迟到了", 30, ["喷玩机器"]),
    Meme(4, "鱼越大鱼刺越大，鱼刺越大鱼肉越少", 80, []),
    Meme(5, "这条包含屏蔽词傻逼", 999, []),
    Meme(6, "Niko：看我用虾须跳绳 🦐🦐", 10, ["NiKo"]),  # 与 1 高度相似
]


def make(**kw):
    return Retriever(MEMES, ALIASES, blocklist=["傻逼"], **kw)


def test_normalize_and_bigrams():
    assert normalize("NiKo 今天!!") == "niko今天"
    assert bigrams("尼尼") == ["尼尼"]
    assert bigrams("") == []


def test_alias_match_longest_and_emoji():
    r = make()
    hits = r.match_aliases("大虾🦐今天又输了")
    assert [e.canonical for e in hits] == ["NiKo"]


def test_alias_exclude():
    r = make()
    assert r.match_aliases("我在学机器学习") == []
    assert r.match_aliases("机器人也会打 CS 吗") == []
    assert [e.canonical for e in r.match_aliases("机器今天开播吗")] == ["玩机器"]


def test_ambiguous_term_needs_context():
    entry = AliasEntry("登山", ["登山事件"], kind="term", note="停播",
                       ambiguous=["登山"], requires_any=["机器", "主播", "下播"])
    r = Retriever(MEMES, [*ALIASES, entry])
    assert r.match_aliases("我周末去登山") == []
    assert "登山" in [e.canonical for e in r.match_aliases("主播又登山了")]
    assert "登山" in [e.canonical for e in r.match_aliases("登山事件是怎么回事")]


def test_blocklisted_memes_not_indexed():
    r = make()
    assert all(m.id != 5 for m in r.memes)


def test_generic_query_returns_nothing():
    r = make()
    ctx = r.retrieve("帮我写一个快速排序")
    assert ctx.empty


def test_alias_query_returns_related_memes_deduplicated():
    r = make()
    ctx = r.retrieve("尼尼今天怎么了")
    ids = [m.id for m in ctx.memes]
    assert ids and set(ids) <= {1, 2, 6}
    assert not ({1, 6} <= set(ids)), "高度相似的复读变体只应保留一条"


def test_term_entry_matches():
    r = make()
    assert [e.canonical for e in r.retrieve("这把爆了吗").aliases] == ["爆了"]


def test_load_blocklist_merges_local_file(tmp_path):
    from meme6657.retrieval import load_blocklist
    (tmp_path / "blocklist.txt").write_text("# 公开\n袜子\n", encoding="utf-8")
    (tmp_path / "blocklist.local.txt").write_text("# 隐私\n某真名\n", encoding="utf-8")
    assert load_blocklist(tmp_path / "blocklist.txt") == ["袜子", "某真名"]
    (tmp_path / "blocklist.local.txt").unlink()
    assert load_blocklist(tmp_path / "blocklist.txt") == ["袜子"]


def test_expired_memes_excluded(tmp_path):
    import json
    from meme6657.retrieval import Meme, Retriever, is_expired, load_expired
    p = tmp_path / "expired.json"
    p.write_text(json.dumps([{"match": "niko梦碎", "reason": "已夺冠"}], ensure_ascii=False), encoding="utf-8")
    expired = load_expired(p)
    assert is_expired("男人一年只哭三次：Niko梦碎上半年major", expired)
    assert not is_expired("NiKo（👀12-6❓❗）🦐腿开始发抖", expired)
    r = Retriever([Meme(1, "Niko梦碎波士顿Major"), Meme(2, "NiKo腿开始发抖")], [], expired)
    assert [m.text for m in r.memes] == ["NiKo腿开始发抖"]
