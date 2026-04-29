import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent, Session
from .memory import FileMemory
from .types import StopReason


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    tools_only: bool = False
    exec_code: bool = False
    plan: bool = False


class ChatResponse(BaseModel):
    content: str
    stop_reason: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int]


class SessionInfo(BaseModel):
    id: str
    title: str
    message_count: int
    created: float


_agent: Agent | None = None
_sessions: dict[str, dict] = {}


def _get_or_create_session(
    session_id: str | None,
    tools_only: bool = False,
    exec_code: bool = False,
    plan: bool = False,
) -> tuple[str, Session]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]["session"]

    session = _agent.session(tools_only=tools_only, exec_code=exec_code, plan=plan)
    sid = session.session_id
    _sessions[sid] = {
        "session": session,
        "title": "",
        "created": time.time(),
    }
    return sid, session


def create_app(agent: Agent | None = None) -> FastAPI:
    global _agent
    _agent = agent or Agent("web-agent", memory=FileMemory())

    app = FastAPI(title="AgentOS")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX_HTML

    @app.post("/chat")
    def chat(req: ChatRequest):
        sid, session = _get_or_create_session(
            req.session_id, req.tools_only, req.exec_code, req.plan,
        )
        try:
            response = session.send(req.message)
            if not _sessions[sid]["title"]:
                _sessions[sid]["title"] = req.message[:50]
            return {
                "session_id": sid,
                "content": response.content,
                "stop_reason": response.stop_reason,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            return {
                "session_id": sid,
                "content": f"Error: {type(e).__name__}: {e}",
                "stop_reason": "error",
                "tool_calls": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        sid, session = _get_or_create_session(
            req.session_id, req.tools_only, req.exec_code, req.plan,
        )
        if not _sessions[sid]["title"]:
            _sessions[sid]["title"] = req.message[:50]

        def generate():
            yield f"data: {json.dumps({'type': 'session_id', 'data': sid})}\n\n"
            for event in session.send_stream(req.message):
                if event.type == "text":
                    yield f"data: {json.dumps({'type': 'text', 'data': event.data})}\n\n"
                elif event.type == "tool_call":
                    yield f"data: {json.dumps({'type': 'tool_call', 'data': {'name': event.data.name, 'arguments': event.data.arguments}})}\n\n"
                elif event.type == "tool_result":
                    yield f"data: {json.dumps({'type': 'tool_result', 'data': {'output': event.data.output, 'error': event.data.error}})}\n\n"
                elif event.type == "code_exec":
                    yield f"data: {json.dumps({'type': 'code_exec', 'data': event.data[:200]})}\n\n"
                elif event.type == "code_result":
                    yield f"data: {json.dumps({'type': 'code_result', 'data': event.data[:500]})}\n\n"
                elif event.type == "done":
                    yield f"data: {json.dumps({'type': 'done', 'data': {'content': event.data.content, 'stop_reason': event.data.stop_reason}})}\n\n"
                elif event.type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': event.data})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/sessions")
    def list_sessions():
        return [
            SessionInfo(
                id=sid,
                title=info["title"] or "New chat",
                message_count=len(info["session"].messages),
                created=info["created"],
            )
            for sid, info in sorted(_sessions.items(), key=lambda x: -x[1]["created"])
        ]

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        if session_id in _sessions:
            del _sessions[session_id]
            return {"status": "deleted"}
        return {"status": "not found"}

    @app.get("/sessions/{session_id}/messages")
    def get_messages(session_id: str):
        if session_id not in _sessions:
            return []
        return [
            {"role": m.role, "content": m.content}
            for m in _sessions[session_id]["session"].messages
            if m.role in ("user", "assistant")
        ]

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
         background: #0a0a0a; color: #e0e0e0; height: 100vh; display: flex; }

  /* Sidebar */
  #sidebar { width: 260px; background: #111; border-right: 1px solid #222;
             display: flex; flex-direction: column; flex-shrink: 0; }
  #sidebar-header { padding: 16px; border-bottom: 1px solid #222; display: flex;
                     align-items: center; justify-content: space-between; }
  #sidebar-header h1 { font-size: 15px; font-weight: 600; color: #fff; }
  #new-chat { background: #2563eb; color: #fff; border: none; border-radius: 6px;
              padding: 6px 12px; font-size: 12px; cursor: pointer; font-weight: 500; }
  #new-chat:hover { background: #1d4ed8; }
  #session-list { flex: 1; overflow-y: auto; padding: 8px; }
  .session-item { padding: 10px 12px; border-radius: 8px; cursor: pointer;
                  font-size: 13px; color: #999; margin-bottom: 2px;
                  display: flex; justify-content: space-between; align-items: center; }
  .session-item:hover { background: #1a1a1a; color: #ddd; }
  .session-item.active { background: #1a2a3a; color: #fff; }
  .session-item .title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .session-item .delete { opacity: 0; color: #666; font-size: 16px; padding: 0 4px;
                           cursor: pointer; flex-shrink: 0; }
  .session-item:hover .delete { opacity: 1; }
  .session-item .delete:hover { color: #dc2626; }
  .session-item .count { font-size: 11px; color: #555; margin-left: 8px; flex-shrink: 0; }

  /* Main area */
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #header { padding: 10px 20px; border-bottom: 1px solid #222; display: flex;
            align-items: center; justify-content: space-between; }
  #header .info { font-size: 12px; color: #666; }
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 10px; font-size: 14px;
         line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #1a3a5c; color: #e0e0e0; }
  .msg.assistant { align-self: flex-start; background: #1a1a1a; border: 1px solid #2a2a2a; }
  .msg.system { align-self: center; background: transparent; color: #555; font-size: 12px;
                font-style: italic; max-width: 90%; }
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
  .mode-bar { display: flex; gap: 12px; padding: 0 20px; }
  .mode-bar label { font-size: 12px; color: #888; display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .mode-bar input { accent-color: #2563eb; }
  .empty-state { flex: 1; display: flex; align-items: center; justify-content: center;
                 color: #444; font-size: 14px; }
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>AgentOS</h1>
    <button id="new-chat">+ New</button>
  </div>
  <div id="session-list"></div>
</div>

<div id="main">
  <div id="header">
    <div class="mode-bar">
      <label><input type="radio" name="mode" value="hybrid" checked> hybrid</label>
      <label><input type="radio" name="mode" value="tools_only"> tools only</label>
      <label><input type="radio" name="mode" value="code_only"> code only</label>
      <label><input type="checkbox" id="plan-check"> planning</label>
    </div>
    <span class="info" id="status">ready</span>
  </div>
  <div id="messages"><div class="empty-state">Start a new conversation</div></div>
  <div id="input-area">
    <textarea id="input" rows="1" placeholder="Type a message..." autofocus></textarea>
    <button id="send">Send</button>
    <button id="stop-btn">Stop</button>
  </div>
</div>

<script>
const msgs = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const stopBtn = document.getElementById('stop-btn');
const status = document.getElementById('status');
const sessionList = document.getElementById('session-list');

let currentSessionId = null;

function addMsg(role, text, meta) {
  // Remove empty state
  const empty = msgs.querySelector('.empty-state');
  if (empty) empty.remove();

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

async function loadSessions() {
  const res = await fetch('/sessions');
  const sessions = await res.json();
  sessionList.innerHTML = '';
  for (const s of sessions) {
    const div = document.createElement('div');
    div.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
    div.innerHTML = `<span class="title">${escHtml(s.title)}</span>
      <span class="count">${s.message_count}</span>
      <span class="delete" data-id="${s.id}">&times;</span>`;
    div.addEventListener('click', (e) => {
      if (e.target.classList.contains('delete')) return;
      switchSession(s.id);
    });
    div.querySelector('.delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(s.id);
    });
    sessionList.appendChild(div);
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function switchSession(id) {
  currentSessionId = id;
  msgs.innerHTML = '';
  const res = await fetch(`/sessions/${id}/messages`);
  const messages = await res.json();
  if (messages.length === 0) {
    msgs.innerHTML = '<div class="empty-state">Start a new conversation</div>';
  }
  for (const m of messages) {
    addMsg(m.role, m.content);
  }
  loadSessions();
}

async function deleteSession(id) {
  await fetch(`/sessions/${id}`, {method: 'DELETE'});
  if (currentSessionId === id) {
    currentSessionId = null;
    msgs.innerHTML = '<div class="empty-state">Start a new conversation</div>';
  }
  loadSessions();
}

function newChat() {
  currentSessionId = null;
  msgs.innerHTML = '<div class="empty-state">Start a new conversation</div>';
  loadSessions();
  input.focus();
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
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId,
        tools_only: mode === 'tools_only',
        exec_code: mode === 'code_only',
        plan: plan,
      }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let assistantDiv = null;
    let fullText = '';
    let buffer = '';

    while (true) {
      const {done: streamDone, value} = await reader.read();
      if (streamDone) break;

      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;

        let evt;
        try { evt = JSON.parse(payload); } catch { continue; }

        if (evt.type === 'session_id') {
          currentSessionId = evt.data;
        } else if (evt.type === 'text') {
          if (!assistantDiv) {
            assistantDiv = document.createElement('div');
            assistantDiv.className = 'msg assistant';
            msgs.appendChild(assistantDiv);
          }
          fullText += evt.data;
          assistantDiv.textContent = fullText;
          msgs.scrollTop = msgs.scrollHeight;
        } else if (evt.type === 'tool_call') {
          addMsg('system', 'calling ' + evt.data.name + '(' + JSON.stringify(evt.data.arguments) + ')');
          status.textContent = 'tool: ' + evt.data.name;
        } else if (evt.type === 'tool_result') {
          const out = evt.data.error || evt.data.output || '';
          addMsg('system', 'result: ' + out.slice(0, 200));
          assistantDiv = null;
          fullText = '';
        } else if (evt.type === 'code_exec') {
          addMsg('system', 'executing code...');
          status.textContent = 'executing code';
        } else if (evt.type === 'code_result') {
          addMsg('system', 'output: ' + (evt.data || '').slice(0, 200));
          assistantDiv = null;
          fullText = '';
        } else if (evt.type === 'done') {
          if (!assistantDiv && evt.data.content) {
            addMsg('assistant', evt.data.content);
          }
          if (assistantDiv) {
            const meta = document.createElement('div');
            meta.className = 'meta';
            meta.textContent = evt.data.stop_reason || 'done';
            assistantDiv.appendChild(meta);
          }
        } else if (evt.type === 'error') {
          addMsg('system', 'error: ' + evt.data);
        }
      }
    }
  } catch (e) {
    addMsg('system', 'error: ' + e.message);
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    stopBtn.style.display = 'none';
    status.textContent = 'ready';
    input.focus();
    loadSessions();
  }
}

document.getElementById('new-chat').addEventListener('click', newChat);
sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

stopBtn.addEventListener('click', async () => {
  await fetch('/stop', {method: 'POST'});
  status.textContent = 'stopping...';
});

loadSessions();
</script>
</body>
</html>
"""
