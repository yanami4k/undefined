from scripts.sync_labels import merge


def card(order, reply="r"):
    return {"batch": "b", "order": order, "kind": "meme", "source": "", "user": "u", "reply": reply}


def gold_row(cid, verdict):
    return {"id": cid, "batch": "b", "kind": "meme", "source": "", "user": "u", "reply": "r",
            "verdict": verdict, "reasons": []}


def test_merge_keeps_rows_missing_from_the_export():
    gold = [gold_row("old-1", "good"), gold_row("old-2", "bad")]
    cards = {"old-1": card(1), "old-2": card(2), "new-1": card(9001)}
    rows, updated, missing = merge(gold, cards, {"new-1": {"verdict": "wave"}})
    assert [r["id"] for r in rows] == ["old-1", "old-2", "new-1"]
    assert rows[:2] == gold and updated == 0 and missing == []


def test_merge_overrides_relabeled_cards_in_place():
    gold = [gold_row("a", "good"), gold_row("b", "good")]
    cards = {"a": card(1), "b": card(2)}
    rows, updated, _ = merge(gold, cards, {"a": {"verdict": "bad", "reasons": ["x"]}, "b": {"verdict": "good"}})
    assert [(r["id"], r["verdict"]) for r in rows] == [("a", "bad"), ("b", "good")]
    assert updated == 1


def test_merge_skips_unlabeled_and_unknown_cards():
    rows, _, missing = merge([], {"a": card(1)}, {"a": {"verdict": "skip"}, "ghost": {"verdict": "good"}})
    assert rows == [] and missing == ["ghost"]
