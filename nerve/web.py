"""Backwards compatibility — use nerve.ux.WebUX instead."""

from .ux.web import WebUX


def create_app(agent=None, tools_only=False, exec_code=False, plan=False):
    from .agent import Agent
    from .providers import FileMemory
    agent = agent or Agent("web-agent", memory=FileMemory())
    return WebUX().create_app(agent, tools_only=tools_only, exec_code=exec_code, plan=plan)


def run_server(agent=None, host="127.0.0.1", port=8420, tools_only=False, exec_code=False, plan=False):
    from .agent import Agent
    from .providers import FileMemory
    agent = agent or Agent("web-agent", memory=FileMemory())
    WebUX(host=host, port=port).start(agent, tools_only=tools_only, exec_code=exec_code, plan=plan)
