"""配置：从环境变量和项目根目录的 .env 读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KB_DIR = ROOT / "kb"
LOG_DIR = ROOT / "logs"

MEMES_PATH = RAW_DIR / "memes.jsonl"
TAGS_PATH = RAW_DIR / "tags.json"
ALIASES_PATH = KB_DIR / "aliases.json"
STYLE_CARD_PATH = KB_DIR / "style_card.md"
TASTE_PATH = KB_DIR / "taste.md"
CS_FACTS_MD = KB_DIR / "cs_facts.md"
TEMPLATES_JSON = KB_DIR / "templates.json"
TEMPLATES_MD = KB_DIR / "templates.md"


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE，忽略注释和空行；已存在的环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class Endpoint:
    """一个 OpenAI 兼容的模型端点。"""

    name: str
    base_url: str
    api_key: str
    model: str
    extra_body: dict = field(default_factory=dict)


def _deepseek_extra(base_url: str) -> dict:
    # DeepSeek V4 系列默认开启思考模式，聊天和合成数据都不需要，显式关闭
    return {"thinking": {"type": "disabled"}} if "deepseek.com" in base_url else {}


def endpoint(role: str) -> Endpoint:
    """role: teacher / judge / usersim / student"""
    if role == "student":
        base = env("STUDENT_BASE_URL", "http://localhost:8000/v1")
        return Endpoint("student", base, env("STUDENT_API_KEY", "EMPTY"), env("STUDENT_MODEL", "meme6657"))

    base = env("TEACHER_BASE_URL", "https://api.deepseek.com")
    key = env("TEACHER_API_KEY") or env("DEEPSEEK_API_KEY")
    models = {
        "teacher": env("TEACHER_MODEL", "deepseek-v4-pro"),
        "judge": env("JUDGE_MODEL", "deepseek-flash"),
        "usersim": env("USERSIM_MODEL", "deepseek-flash"),
    }
    if role not in models:
        raise ValueError(f"未知角色: {role}")
    return Endpoint(role, base, key, models[role], _deepseek_extra(base))
