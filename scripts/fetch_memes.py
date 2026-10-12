"""从 sb6657 公开只读接口低速拉取烂梗样本。

    python -m scripts.fetch_memes                 # 复制最多 2000 条 + 最新 1000 条
    python -m scripts.fetch_memes --top-pages 500 # 全量（约 24k 条，建议先征得站长同意）

每 4 秒一次请求，不携带官网前端的来源统计请求头（见 sb6657 README 的接口调用说明）。
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://api.hguofichp.cn"
USER_AGENT = "sb6657-style-prototype/0.1 (personal research; low-rate)"
PAGE_SIZE = 50
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_json(path: str, params: dict | None = None) -> dict:
    url = API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8-sig"))
    if str(body.get("code")) != "200":
        raise RuntimeError(f"{path} 返回 code={body.get('code')}: {body.get('msg')}")
    return body["data"]


def fetch_pages(path: str, pages: int, delay: float, seen: dict, source: str) -> None:
    for page in range(1, pages + 1):
        data = get_json(path, {"tags": "", "pageNum": page, "pageSize": PAGE_SIZE})
        rows = data.get("list") or []
        for row in rows:
            row.setdefault("source", source)
            seen.setdefault(row["id"], row)
        print(f"[{source}] 第 {page}/{pages} 页，累计 {len(seen)} 条", flush=True)
        if not rows or data.get("lastPage"):
            break
        time.sleep(delay)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-pages", type=int, default=40, help="按复制数排序拉取的页数")
    ap.add_argument("--latest-pages", type=int, default=20, help="按时间拉取最新的页数")
    ap.add_argument("--delay", type=float, default=4.0, help="请求间隔秒数")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tags = get_json("/machine/dictList")
    tag_map = {t["dictValue"]: t["dictLabel"] for t in tags}
    (DATA_DIR / "tags.json").write_text(json.dumps(tag_map, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(args.delay)

    seen: dict[int, dict] = {}
    fetch_pages("/machine/sortAllBarrage", args.top_pages, args.delay, seen, "top")
    time.sleep(args.delay)
    fetch_pages("/machine/Page", args.latest_pages, args.delay, seen, "latest")

    with open(DATA_DIR / "memes.jsonl", "w", encoding="utf-8") as f:
        for row in seen.values():
            row["tag_names"] = [tag_map.get(t, t) for t in (row.get("tags") or "").split(",") if t]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"完成，共 {len(seen)} 条，已写入 {DATA_DIR / 'memes.jsonl'}")


if __name__ == "__main__":
    main()
