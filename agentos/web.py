import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent
from .memory import FileMemory
from .types import StopReason


class ChatRequest(BaseModel):
    message: str
    tools_only: bool = False
    exec_code: bool = False
    plan: bool = False


class ChatResponse(BaseModel):
    content: str
    stop_reason: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int]


_agent: Agent | None = None


def create_app(agent: Agent | None = None) -> FastAPI:
    global _agent
    _agent = agent or Agent("web-agent", memory=FileMemory())

    app = FastAPI(title="AgentOS")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX_HTML

    @app.post("/chat")
    def chat(req: ChatRequest):
        try:
            response = _agent.run(
                req.message,
                tools_only=req.tools_only,
                exec_code=req.exec_code,
                plan=req.plan,
            )
            return ChatResponse(
                content=response.content,
                stop_reason=response.stop_reason,
                tool_calls=[
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        except Exception as e:
            return ChatResponse(
                content=f"Error: {type(e).__name__}: {e}",
                stop_reason="error",
                tool_calls=[],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

    @app.get("/tools")
    def tools():
        return [
            {"name": t.name, "description": t.description}
            for t in _agent.tool_registry.list()
        ]

    @app.get("/skills")
    def skills():
        return [
            {"name": s.name, "description": s.description or s.prompt[:80]}
            for s in _agent.skill_registry.list()
        ]

    @app.post("/stop")
    def stop():
        _agent.stop()
        return {"status": "stopping"}

    return app


def run_server(agent: Agent | None = None, host: str = "127.0.0.1", port: int = 8420):
    import uvicorn
    app = create_app(agent)
    print(f"AgentOS web UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentOS</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0a0a0a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  #header { padding: 12px 20px; border-bottom: 1px solid #222; display: flex;
            align-items: center; justify-content: space-between; }
  #header h1 { font-size: 16px; font-weight: 600; color: #fff; }
  #header .info { font-size: 12px; color: #666; }
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 10px; font-size: 14px;
         line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #1a3a5c; color: #e0e0e0; }
  .msg.assistant { align-self: flex-start; background: #1a1a1a; border: 1px solid #2a2a2a; }
  .msg.system { align-self: center; background: transparent; color: #555; font-size: 12px;
                font-style: italic; }
  .msg .meta { font-size: 11px; color: #555; margin-top: 6px; }
  #input-area { padding: 12px 20px; border-top: 1px solid #222; display: flex; gap: 8px; }
  #input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #333;
           background: #111; color: #e0e0e0; font-size: 14px; font-family: inherit;
           outline: none; resize: none; }
  #input:focus { border-color: #555; }
  #input:disabled { opacity: 0.5; }
  button { padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer;
           font-size: 14px; font-weight: 500; }
  #send { background: #2563eb; color: #fff; }
  #send:hover { background: #1d4ed8; }
  #send:disabled { opacity: 0.5; cursor: default; }
  #stop-btn { background: #dc2626; color: #fff; display: none; }
  #stop-btn:hover { background: #b91c1c; }
  .mode-bar { display: flex; gap: 12px; padding: 0 20px 0; }
  .mode-bar label { font-size: 12px; color: #888; display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .mode-bar input { accent-color: #2563eb; }
</style>
</head>
<body>
<div id="header">
  <h1>AgentOS</h1>
  <span class="info" id="status">ready</span>
</div>
<div class="mode-bar">
  <label><input type="radio" name="mode" value="hybrid" checked> hybrid</label>
  <label><input type="radio" name="mode" value="tools_only"> tools only</label>
  <label><input type="radio" name="mode" value="code_only"> code only</label>
  <label><input type="checkbox" id="plan-check"> planning</label>
</div>
<div id="messages"></div>
<div id="input-area">
  <textarea id="input" rows="1" placeholder="Type a message..." autofocus></textarea>
  <button id="send">Send</button>
  <button id="stop-btn">Stop</button>
</div>
<script>
const msgs = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const stopBtn = document.getElementById('stop-btn');
const status = document.getElementById('status');

function addMsg(role, text, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function getMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : 'hybrid';
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg('user', text);

  const mode = getMode();
  const plan = document.getElementById('plan-check').checked;

  sendBtn.disabled = true;
  input.disabled = true;
  stopBtn.style.display = 'inline-block';
  status.textContent = 'thinking...';

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: text,
        tools_only: mode === 'tools_only',
        exec_code: mode === 'code_only',
        plan: plan,
      }),
    });
    const data = await res.json();
    const tokens = (data.usage && data.usage.total_tokens) || 0;
    const meta = `${data.stop_reason || 'unknown'} · ${tokens} tokens`;
    addMsg('assistant', data.content || '(no content)', meta);
    if (data.tool_calls && data.tool_calls.length > 0) {
      const tools = data.tool_calls.map(t => t.name + '(' + JSON.stringify(t.arguments) + ')').join('\\n');
      addMsg('system', 'tools used: ' + tools);
    }
  } catch (e) {
    addMsg('system', 'error: ' + e.message);
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    stopBtn.style.display = 'none';
    status.textContent = 'ready';
    input.focus();
  }
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

stopBtn.addEventListener('click', async () => {
  await fetch('/stop', {method: 'POST'});
  status.textContent = 'stopping...';
});

fetch('/tools').then(r => r.json()).then(tools => {
  status.textContent = tools.length + ' tools loaded';
});
</script>
</body>
</html>
"""
