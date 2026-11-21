"""
chat_windows.py
Windows-native chat interface for CMOS.
Run with: python chat_windows.py

Key difference from chat.py:
  - Applies Windows asyncio policy fix at the very top
  - UTF-8 console encoding for diary content
  - Handles Windows keyboard interrupt cleanly
"""

# ── MUST be first — before any other imports ──
import sys
import os
import asyncio

# Windows asyncio fix — prevents "Event loop is closed" errors
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Suppress PaddlePaddle GPU detection noise
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dotenv import load_dotenv
load_dotenv()

from orchestrator import CMOSApplication, CMOSConfig
from memory_repo import (
    SimpleMemoryRepo, SimplePatternRepo,
    SimpleContradictionRepo, SimplePsychModelStore
)

config = CMOSConfig(
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
)


async def main():
    print("\n" + "=" * 60)
    print("  COGNITIVE MEMORY OS")
    print("  Your Personal Memory Agent")
    print("  Type 'quit' to exit | 'list' to see memories | 'help' for commands")
    print("=" * 60 + "\n")

    app = CMOSApplication(config)
    repo = SimpleMemoryRepo()

    await app.initialize(
        memory_repo=repo,
        pattern_repo=SimplePatternRepo(),
        contradiction_repo=SimpleContradictionRepo(),
        psych_model_store=SimplePsychModelStore(),
    )

    all_mems = await repo.get_all()
    print(f"Memory store: {len(all_mems)} memories loaded\n")

    if len(all_mems) == 0:
        print("No memories found yet.")
        print("Run: python quickstart_windows.py  to add your first entry\n")
        return

    commands = {
        "list":    "List your 10 most recent memories",
        "count":   "Count memories by type",
        "help":    "Show this help",
        "quit":    "Exit",
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        elif user_input.lower() == "help":
            print("\nCommands:")
            for cmd, desc in commands.items():
                print(f"  {cmd:<10} {desc}")
            print("\nOr just ask anything about your life.\n")
            continue

        elif user_input.lower() == "list":
            mems = sorted(
                all_mems,
                key=lambda m: str(m.date_info.occurred or "0000"),
                reverse=True
            )[:10]
            print(f"\nMost recent {len(mems)} memories:")
            for m in mems:
                date = m.date_info.occurred or "unknown date"
                print(f"  [{m.memory_type}] {date} — {m.title}")
            print()
            continue

        elif user_input.lower() == "count":
            type_counts = {}
            for m in all_mems:
                t = str(m.memory_type)
                type_counts[t] = type_counts.get(t, 0) + 1
            print("\nMemories by type:")
            for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {t:<15} {count}")
            print(f"  {'TOTAL':<15} {len(all_mems)}\n")
            continue

        print("\nAgent: ", end="", flush=True)
        try:
            response = await app.ask(user_input)
            print(response)
        except Exception as e:
            print(f"\n[Error: {e}]")
            print("This might be an API issue. Check your .env file.")
        print()


if __name__ == "__main__":
    # Windows requires this guard for multiprocessing compatibility
    asyncio.run(main())
