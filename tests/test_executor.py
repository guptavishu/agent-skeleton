"""Tests for code execution and code block extraction."""

from agentos.executor import extract_code_blocks, execute_code


def test_extract_python_blocks():
    text = "Here is code:\n```python\nprint('hi')\n```\nDone."
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert "print('hi')" in blocks[0]


def test_extract_untagged_blocks():
    text = "```\nx = 1\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert "x = 1" in blocks[0]


def test_extract_multiple_blocks():
    text = "```python\na = 1\n```\ntext\n```python\nb = 2\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2


def test_extract_no_blocks():
    assert extract_code_blocks("just plain text") == []


def test_execute_code_success():
    result = execute_code("print(2 + 2)")
    assert result.stdout.strip() == "4"
    assert result.returncode == 0


def test_execute_code_error():
    result = execute_code("raise ValueError('oops')")
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_execute_code_timeout():
    result = execute_code("import time; time.sleep(10)", timeout=1)
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()
