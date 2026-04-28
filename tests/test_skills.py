"""Tests for Skill and SkillRegistry."""

import os
import tempfile

from agentos.skills import Skill, SkillRegistry
from agentos.tools import Tool


def _dummy_tool():
    def noop(x: str) -> str:
        """Does nothing."""
        return x
    return Tool.from_function(noop)


def test_skill_creation():
    s = Skill(name="test", prompt="do stuff", description="a test skill")
    assert s.name == "test"
    assert s.tools == []


def test_skill_load_from_file():
    code = '''
from agentos import Skill, Tool

def hello(name: str) -> str:
    """Say hi."""
    return f"hi {name}"

skill = Skill(
    name="greeter",
    prompt="You greet people.",
    tools=[Tool.from_function(hello)],
)
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        s = Skill.load(path)
        assert s.name == "greeter"
        assert len(s.tools) == 1
        assert s.tools[0].name == "hello"
    finally:
        os.unlink(path)


def test_skill_load_missing_variable():
    code = "x = 42\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        raised = False
        try:
            Skill.load(path)
        except ValueError:
            raised = True
        assert raised
    finally:
        os.unlink(path)


# --- SkillRegistry ---

def test_registry_register_and_list():
    reg = SkillRegistry()
    s = Skill(name="s1", prompt="p1")
    reg.register(s)
    assert reg.list() == [s]
    assert reg.get("s1") is s


def test_registry_names():
    reg = SkillRegistry()
    reg.register(Skill(name="a", prompt=""))
    reg.register(Skill(name="b", prompt=""))
    assert set(reg.names()) == {"a", "b"}


def test_registry_get_tools():
    tool = _dummy_tool()
    reg = SkillRegistry()
    reg.register(Skill(name="s", prompt="", tools=[tool]))
    assert reg.get_tools() == [tool]


def test_registry_get_tools_filtered():
    t1 = _dummy_tool()
    t2 = _dummy_tool()
    reg = SkillRegistry()
    reg.register(Skill(name="a", prompt="", tools=[t1]))
    reg.register(Skill(name="b", prompt="", tools=[t2]))
    tools = reg.get_tools(skill_names=["a"])
    assert tools == [t1]


def test_registry_get_prompts():
    reg = SkillRegistry()
    reg.register(Skill(name="s1", prompt="Do X"))
    reg.register(Skill(name="s2", prompt="Do Y"))
    combined = reg.get_prompts()
    assert "s1" in combined
    assert "Do X" in combined
    assert "s2" in combined


def test_registry_discover_from_dir():
    code = '''
from agentos import Skill
skill = Skill(name="discovered", prompt="I was found")
'''
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "my_skill.py"), "w") as f:
            f.write(code)
        reg = SkillRegistry()
        found = reg.discover(dirs=[__import__("pathlib").Path(d)])
        assert len(found) == 1
        assert found[0].name == "discovered"
        assert reg.get("discovered") is not None


def test_registry_discover_skips_underscored():
    code = '''
from agentos import Skill
skill = Skill(name="hidden", prompt="")
'''
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "_private.py"), "w") as f:
            f.write(code)
        reg = SkillRegistry()
        found = reg.discover(dirs=[__import__("pathlib").Path(d)])
        assert found == []
