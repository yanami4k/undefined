"""本地检索：外号匹配 + 字符二元组倒排索引。

纯 Python、无依赖，两万条规模下单次检索在毫秒级。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import ALIASES_PATH, KB_DIR, MEMES_PATH

_TOKEN_RE = re.compile(r"[0-9a-z一-鿿]+")


def normalize(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def bigrams(text: str) -> list[str]:
    s = normalize(text)
    if len(s) < 2:
        return [s] if s else []
    return [s[i : i + 2] for i in range(len(s) - 1)]


@dataclass
class AliasEntry:
    canonical: str
    aliases: list[str]
    kind: str = ""
    note: str = ""
    tags: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)  # 包含这些词时不算命中，如「机器学习」
    # 有歧义的写法（如「登山」也是普通词）只有在文本里同时出现 requires_any 之一时才算命中
    ambiguous: list[str] = field(default_factory=list)
    requires_any: list[str] = field(default_factory=list)
    status: str = "verified"  # verified / candidate

    @property
    def terms(self) -> list[str]:
        return [self.canonical, *self.aliases]


@dataclass
class Meme:
    id: int
    text: str
    cnt: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class Context:
    aliases: list[AliasEntry] = field(default_factory=list)
    memes: list[Meme] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.aliases and not self.memes


def load_memes(path: Path = MEMES_PATH) -> list[Meme]:
    memes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            memes.append(Meme(int(r["id"]), r["barrage"].strip(), int(r.get("cnt") or 0), r.get("tag_names") or []))
    return memes


def load_aliases(path: Path = ALIASES_PATH, include_candidates: bool = False) -> list[AliasEntry]:
    if not path.exists():
        path = KB_DIR / "aliases.seed.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [AliasEntry(**{k: v for k, v in e.items() if k in AliasEntry.__dataclass_fields__}) for e in raw]
    return [e for e in entries if include_candidates or e.status == "verified"]


def load_blocklist(path: Path = KB_DIR / "blocklist.txt") -> list[str]:
    """屏蔽片段：blocklist.txt（提交）+ 同目录的 blocklist.local.txt（不提交：不适合公开的隐私片段）。"""
    words = []
    for p in (path, path.with_name(f"{path.stem}.local{path.suffix}")):
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            words += [w.strip().lower() for w in lines if w.strip() and not w.startswith("#")]
    return words


class Retriever:
    def __init__(
        self,
        memes: list[Meme],
        aliases: list[AliasEntry],
        blocklist: list[str] | None = None,
        min_score: float = 6.0,
        max_meme_chars: int = 200,
    ):
        blocklist = blocklist or []
        # 屏蔽词命中、过长的复制粘贴不进索引，避免被学生模型照抄
        self.memes = [
            m for m in memes
            if len(m.text) <= max_meme_chars and not any(b in m.text.lower() for b in blocklist)
        ]
        self.aliases = aliases
        self.min_score = min_score
        self._alias_terms = sorted(
            ((t.lower(), e) for e in aliases for t in e.terms if t), key=lambda x: -len(x[0])
        )
        self._grams = [set(bigrams(m.text)) for m in self.memes]
        self._index: dict[str, list[int]] = defaultdict(list)
        for i, grams in enumerate(self._grams):
            for g in grams:
                self._index[g].append(i)
        n = max(1, len(self.memes))
        self._idf = {g: math.log(1 + n / len(ids)) for g, ids in self._index.items()}
        self._tag_index: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(self.memes):
            for t in m.tags:
                self._tag_index[t].append(i)
        self._lower = [m.text.lower() for m in self.memes]

    @classmethod
    def from_files(cls, **kw) -> "Retriever":
        return cls(load_memes(), load_aliases(), load_blocklist(), **kw)

    # ------------------------------------------------------------ aliases

    def match_aliases(self, text: str) -> list[AliasEntry]:
        low = text.lower()
        taken = [False] * len(low)
        hits: list[AliasEntry] = []
        for term, entry in self._alias_terms:
            start = low.find(term)
            while start != -1:
                end = start + len(term)
                if not any(taken[start:end]) and not self._excluded(low, start, end, entry) \
                        and self._context_ok(low, term, entry):
                    for i in range(start, end):
                        taken[i] = True
                    if entry not in hits:
                        hits.append(entry)
                start = low.find(term, end)
        return hits

    @staticmethod
    def _context_ok(low: str, term: str, entry: AliasEntry) -> bool:
        if term not in (a.lower() for a in entry.ambiguous):
            return True
        return any(r.lower() in low for r in entry.requires_any)

    @staticmethod
    def _excluded(low: str, start: int, end: int, entry: AliasEntry) -> bool:
        for ex in entry.exclude:
            ex = ex.lower()
            pos = low.find(ex)
            while pos != -1:
                if pos <= start and end <= pos + len(ex):
                    return True
                pos = low.find(ex, pos + 1)
        return False

    # ------------------------------------------------------------ memes

    def search_memes(self, text: str, k: int = 3, hits: list[AliasEntry] | None = None) -> list[Meme]:
        hits = hits if hits is not None else self.match_aliases(text)
        q = Counter(bigrams(text))
        scores: dict[int, float] = defaultdict(float)
        for g in q:
            for i in self._index.get(g, ()):
                scores[i] += self._idf[g]
        for i in list(scores):
            scores[i] /= math.sqrt(len(self._grams[i]) + 4)  # 惩罚超长复读

        # 外号命中：带对应标签或包含外号的梗加分
        for i in {i for e in hits for t in e.tags for i in self._tag_index.get(t, ())}:
            scores[i] += 3.0
        for i in {i for e in hits for t in e.terms for i in self._memes_containing(t)}:
            scores[i] += 4.0

        ranked = sorted(scores.items(), key=lambda kv: -(kv[1] + 0.15 * math.log1p(self.memes[kv[0]].cnt)))
        picked: list[int] = []
        for i, s in ranked:
            if s < self.min_score and not hits:
                break
            if s < self.min_score / 2:
                break
            # 去掉高度相似的复读变体
            if any(_jaccard(self._grams[i], self._grams[j]) > 0.4 for j in picked):
                continue
            picked.append(i)
            if len(picked) >= k:
                break
        return [self.memes[i] for i in picked]

    def _memes_containing(self, term: str) -> list[int]:
        """包含某个外号原文的梗。先用二元组倒排缩小候选，再做子串确认。"""
        term = term.lower()
        grams = bigrams(term)
        if not grams or len(normalize(term)) < 2:
            # emoji、单字等无法走索引的外号，退化为全量扫描
            return [i for i, low in enumerate(self._lower) if term in low]
        candidates = set(self._index.get(grams[0], ()))
        for g in grams[1:]:
            candidates &= set(self._index.get(g, ()))
            if not candidates:
                return []
        return [i for i in candidates if term in self._lower[i]]

    def retrieve(self, text: str, k: int = 3) -> Context:
        hits = self.match_aliases(text)
        return Context(aliases=hits[:4], memes=self.search_memes(text, k=k, hits=hits))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
