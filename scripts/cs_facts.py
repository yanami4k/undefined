"""CS 事实表：从 Liquipedia 官方 API 拉主要战队的当前阵容和选手信息。

    python -m scripts.cs_facts

生成用户消息和回复时，老师模型会凭印象编选手位置、所在战队，容易出错
（例：把自由人 zont1x 说成狙击手）。事实表交给老师，要求只用表里的事实。

Liquipedia 的 role 字段比较粗（rifle / awp / entry / igl …），更细的位置
（如自由人）以 kb/cs_roles_override.json 里人工补充的为准。
请求用批量 query 接口，带标识性 User-Agent，请求间隔 3 秒以上。
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date

from meme6657.config import DATA_DIR, KB_DIR

API = "https://liquipedia.net/counterstrike/api.php"
USER_AGENT = "meme6657-dataset/0.1 (personal research; low-rate)"
OUT_JSON = KB_DIR / "cs_facts.json"
OUT_MD = KB_DIR / "cs_facts.md"
OVERRIDE = KB_DIR / "cs_roles_override.json"
CACHE = DATA_DIR / "cs"

TEAMS = [
    "Team Vitality", "Team Spirit", "Team Falcons", "MOUZ", "FaZe Clan", "Natus Vincere", "G2 Esports",
    "The MongolZ", "Aurora Gaming", "Team Liquid", "Astralis", "FURIA Esports", "PaiN Gaming", "MIBR",
    "TyLoo", "Lynn Vision Gaming", "3DMAX", "Virtus.pro", "HEROIC", "Eternal Fire", "GamerLegion",
    "PARIVISION", "B8", "Legacy",
]
ROLE_ZH = {"rifle": "步枪手", "rifler": "步枪手", "awp": "狙击手", "awper": "狙击手", "entry": "突破手",
           "entryfragger": "突破手", "entry fragger": "突破手",
           "igl": "指挥", "lurk": "自由人", "lurker": "自由人", "support": "辅助", "coach": "教练",
           "anchor": "防守位"}


def query(titles: list[str]) -> dict[str, str]:
    """批量读取页面源码（一个请求最多 50 页），返回 {页面标题: wikitext}。"""
    params = {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
              "format": "json", "redirects": "1", "titles": "|".join(titles)}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
    data = json.loads(raw.decode("utf-8"))
    redirects = {r["to"]: r["from"] for r in data["query"].get("redirects", [])}
    out = {}
    for p in data["query"]["pages"].values():
        if "revisions" in p:
            out[redirects.get(p["title"], p["title"])] = p["revisions"][0]["slots"]["main"]["*"]
    time.sleep(3)
    return out


def _templates(text: str, name: str) -> list[str]:
    """按花括号配对取出所有 {{name|...}} 的内部文本，正确处理嵌套模板。"""
    out, start = [], 0
    opener = "{{" + name + "|"
    while (i := text.find(opener, start)) != -1:
        depth, j = 0, i
        while j < len(text) - 1:
            if text.startswith("{{", j):
                depth += 1
                j += 2
            elif text.startswith("}}", j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        out.append(text[i + len(opener) : j - 2])
        start = j
    return out


def active_squad(wikitext: str) -> list[dict]:
    """解析战队页「Active」阵容里的 {{Person|...}}。"""
    m = re.search(r"===\s*Active\s*===(.*?)(?:===|\Z)", wikitext, re.S)
    if not m:
        return []
    people = []
    for person in _templates(m.group(1), "Person"):
        # 去掉嵌套模板（如 joindate={{abbr|...}}），再按 | 拆字段
        flat = re.sub(r"\{\{.*?\}\}", "", person)
        fields = dict(re.findall(r"(\w+)=([^|]*)", flat))
        if fields.get("id"):
            people.append({"id": fields["id"].strip(), "flag": fields.get("flag", "").strip(),
                           "link": fields.get("link", "").strip(),
                           "igl": fields.get("igl", "").strip().lower() in ("y", "yes", "true"),
                           "coach": fields.get("role", "").strip().lower() == "coach"})
    return people


def infobox(wikitext: str) -> dict:
    m = re.search(r"\{\{Infobox player(.*?)\n\}\}", wikitext, re.S)
    if not m:
        return {}
    return {k.strip(): v.strip() for k, v in re.findall(r"^\|(\w+)=(.*)$", m.group(1), re.M)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    CACHE.mkdir(parents=True, exist_ok=True)
    team_pages = query(TEAMS)
    teams = {}
    for name in TEAMS:
        squad = active_squad(team_pages.get(name, ""))
        if squad:
            teams[name] = squad
        else:
            print(f"  没解析到阵容：{name}")

    # 同名选手用阵容里的 link 指定页面（例：ryu (Lithuanian player)）
    def title_of(p):
        t = p["link"] or p["id"]
        return t[0].upper() + t[1:]

    titles = sorted({title_of(p) for squad in teams.values() for p in squad if not p["coach"]})
    players = {}
    for b in range(0, len(titles), 40):
        pages = query(titles[b : b + 40])
        for t, text in pages.items():
            players[t.lower()] = infobox(text)

    override = json.loads(OVERRIDE.read_text(encoding="utf-8")) if OVERRIDE.exists() else {}
    facts = {"as_of": date.today().isoformat(), "source": "Liquipedia", "teams": {}}
    for name, squad in teams.items():
        rows = []
        for p in squad:
            info = players.get(title_of(p).lower(), {}) if not p["coach"] else {}
            roles = [r.strip().lower() for r in info.get("roles", "").split(",") if r.strip()]
            zh = [ROLE_ZH.get(r, r) for r in roles]
            if p["igl"] and "指挥" not in zh:
                zh.insert(0, "指挥")
            if p["coach"]:
                zh = ["教练"]
            if p["id"].lower() in override:
                zh = override[p["id"].lower()]
            rows.append({"id": p["id"], "country": info.get("country", p["flag"]), "roles": zh})
        facts["teams"][name] = rows
    OUT_JSON.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"## CS 事实表（{facts['as_of']}，来源 Liquipedia）", "",
             "只能使用这里的阵容和位置信息；表里没有的事实（转会、比赛结果、选手位置）不要编。", ""]
    for name, rows in facts["teams"].items():
        cells = "、".join(f"{r['id']}（{'/'.join(r['roles']) or '位置未知'}）" for r in rows)
        lines.append(f"- **{name}**：{cells}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(facts['teams'])} 支战队、{sum(len(v) for v in facts['teams'].values())} 人 → {OUT_MD}")


if __name__ == "__main__":
    main()
