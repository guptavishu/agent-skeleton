"""Demo of multi-turn conversation sessions."""

from nerve import Agent

agent = Agent("chat")
session = agent.session()

print("Multi-turn session (type 'quit' to exit, 'clear' to reset)\n")

while True:
    try:
        msg = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not msg:
        continue
    if msg.lower() in ("quit", "exit", "q"):
        break
    if msg.lower() == "clear":
        session.clear()
        print("  (history cleared)\n")
        continue

    response = session.send(msg)
    print(f"\n{response.content}\n")
    print(f"  [{len(session.messages)} messages in context]\n")
