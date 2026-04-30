"""Background run interrupted after the first tool call."""

from nerve import Agent

agent = Agent("bg-test")

call_count = [0]
def stop_after_first(tc):
    call_count[0] += 1
    print(f"  [tool] {tc.name}({tc.arguments})")
    if call_count[0] >= 1:
        print("  Stopping...")
        agent.stop()

agent.on_tool_call = stop_after_first

handle = agent.run_background(
    "Read every file in the nerve directory and summarize each one",
    tools_only=True,
    max_rounds=20,
)

result = handle.result
print(f"stop_reason: {result.stop_reason}")
if result.content:
    print(f"partial content: {result.content[:200]}")
