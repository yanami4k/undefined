import pytest

from meme6657.config import _deepseek_extra
from meme6657.llm import LLMError, parse_json


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('好的，结果如下：{"a": [1, 2]} 以上') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        parse_json("没有 json")


def test_deepseek_thinking_disabled_only_for_deepseek():
    assert _deepseek_extra("https://api.deepseek.com") == {"thinking": {"type": "disabled"}}
    assert _deepseek_extra("http://localhost:8000/v1") == {}
