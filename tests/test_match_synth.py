from meme6657.retrieval import AliasEntry, Meme, Retriever
from scripts.match_synth import Families

ALIASES = [AliasEntry(canonical="汗还是尿", aliases=["全都是尿"], kind="term"),
           AliasEntry(canonical="NiKo", aliases=["尼尼"], kind="player")]
LIB = ["可是宗宗🥺。。。也太菜了吧😅"]


def make() -> Families:
    r = Retriever([Meme(1, LIB[0])], ALIASES)
    return Families(r, LIB)


def test_term_alias_variants_share_family():
    fam = make()
    a = fam("艾斯蒂尔！艾staehr你的裤子里面是汗还是尿呢😨")
    b = fam("staehr你的👖里面是汗还是尿呢？是尿啊全都是尿啊")
    assert a == b == "黑话:汗还是尿"


def test_player_alias_does_not_merge_families():
    fam = make()
    assert fam("尼尼💚孩孩💚尼尼💚孩孩") != fam("NiKo（👀12-6❓❗）🦐腿开始发抖")


def test_near_duplicate_maps_to_library_meme():
    fam = make()
    assert fam("可是宗宗也🥺...太菜了吧😅") == LIB[0]
    assert fam("完全不相干的一句话") == "完全不相干的一句话"


def test_augment_filters():
    from scripts.augment_pairs import copies_meme, keeps_hook
    meme = "每次流量大战前，都有两支拎不清自己的队伍在这里赖着不死"
    assert not copies_meme("又到流量大战了，总有两支队伍在赖着", meme)  # 只重合几个字的话题词不算
    assert copies_meme("你说的是每次流量大战前都有两支吧", meme)
    assert keeps_hook("你是不是被降智了啊", "降智", "AI你说实话，你是不是被降智了")
    assert not keeps_hook("你最近怎么这么笨了", "降智", "AI你说实话，你是不是被降智了")
    assert keeps_hook("随便什么", "不在原句里的词", "原句")

