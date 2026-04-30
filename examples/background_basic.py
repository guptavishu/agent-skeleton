"""Basic background run with callback notification."""

from nerve import Agent

agent = Agent("bg-test")


def notify(response):
    print(f"\n[done] {response.content[:200]}")
    print(f"[stop_reason] {response.stop_reason}")


handle = agent.run_background(
    "List the files in the current directory",
    on_complete=notify,
)

print("Agent is running in background...")
print(f"Done yet? {handle.done}")

handle.wait()
print(f"Done now? {handle.done}")
