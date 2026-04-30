"""Defer a background run after the first tool call, then resume it."""

from nerve import Agent

agent = Agent("bg-test")

# Defer after the first tool call completes
call_count = [0]
def defer_after_first(tc):
    call_count[0] += 1
    print(f"  [tool] {tc.name}({tc.arguments})")
    if call_count[0] >= 1:
        agent.defer()

agent.on_tool_call = defer_after_first

handle = agent.run_background(
    "List files in the current directory, then read pyproject.toml and summarize it",
    tools_only=True,
)

result = handle.result
print(f"stop_reason: {result.stop_reason}")

if result.stop_reason == "deferred":
    state = agent.last_state
    print(f"state_id: {state.state_id}")
    print(f"round: {state.round}/{state.max_rounds}")
    print(f"messages so far: {len(state.messages)}")

    print("\nResuming...")
    agent.on_tool_call = lambda tc: print(f"  [tool] {tc.name}({tc.arguments})")
    result = agent.resume(state.state_id)
    print(f"stop_reason: {result.stop_reason}")
    print(f"content: {result.content[:300]}")
else:
    print("Agent finished before defer was triggered.")
    print(f"content: {result.content[:300]}")
