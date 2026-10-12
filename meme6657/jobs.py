"""批处理工具：jsonl 读写、可断点续跑的并发执行。"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def stable_hash(text: str) -> float:
    """把字符串映射到 [0, 1)，用于可复现的训练 / 评估切分。"""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def run_parallel(
    items: list[dict],
    fn: Callable[[dict], dict | None],
    out_path: Path,
    key: str = "id",
    workers: int = 8,
) -> None:
    """对 items 并发执行 fn，结果逐条追加到 out_path。

    已存在于 out_path 中的 key 会跳过，中断后重跑即可续上。fn 返回 None 表示丢弃。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r[key] for r in read_jsonl(out_path) if key in r}
    todo = [it for it in items if it[key] not in done]
    print(f"共 {len(items)} 条，已完成 {len(done)}，待处理 {len(todo)}", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    ok = fail = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fn, it): it for it in todo}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                fail += 1
                print(f"  失败 {futures[fut][key]}: {e}", file=sys.stderr, flush=True)
                continue
            with lock:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                ok += 1
            if (ok + fail) % 20 == 0:
                rate = (ok + fail) / max(1e-6, time.time() - t0)
                print(f"  进度 {ok + fail}/{len(todo)}，失败 {fail}，{rate:.1f} 条/秒", flush=True)
    print(f"完成：成功 {ok}，失败 {fail}，输出 {out_path}", flush=True)
    from .llm import USAGE

    if USAGE.by_model:
        print("本次 token 用量：\n" + USAGE.summary(), flush=True)
