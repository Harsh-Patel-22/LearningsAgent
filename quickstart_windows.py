"""
quickstart_windows.py
Windows-native first-run ingestion script.

Run with: python quickstart_windows.py

This is your starting point — paste a real diary entry below and run it.
The system will extract memory objects and index them.
"""

# ── Windows fixes — MUST be first ──
import sys
import os
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"
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

# ════════════════════════════════════════════════════════════
# PASTE YOUR DIARY ENTRY BELOW
# Replace this with a REAL entry from your life.
# The more honest and specific, the better the system works.
# Include: emotions, decisions, people (use initials), events.
# ════════════════════════════════════════════════════════════

MY_DIARY_ENTRY = """
March 14, 2021

I finally told R I couldn't continue at the company.
The board meeting yesterday changed everything. I felt an overwhelming sense of burnout and sadness, but also relief after making the decision. I realized that my identity was too tied to achievement at the expense of my health. This overcommitment-spiral had been going on for two years, and staying longer would have been a delayed-exit loop.
"""

# The date of this entry (YYYY-MM-DD format)
ENTRY_DATE = "2024-01-01"   # <-- change this to the actual date

# Optional: what phase of life was this?
# Examples: startup_phase, burnout_recovery, post_exit,
#           relationship_start, transition_period, creative_phase
LIFE_PHASE = "current"

# ════════════════════════════════════════════════════════════


async def main():
    print("=" * 55)
    print("CMOS — First Entry Ingestion")
    print("=" * 55)

    if "Write your actual diary entry here" in MY_DIARY_ENTRY:
        print("\n⚠  You haven't added a real diary entry yet.")
        print("   Open quickstart_windows.py and paste your")
        print("   actual diary text into MY_DIARY_ENTRY.\n")
        return

    print(f"\nEntry date: {ENTRY_DATE}")
    print(f"Life phase: {LIFE_PHASE}")
    print(f"Text length: {len(MY_DIARY_ENTRY.split())} words\n")

    app = CMOSApplication(config)
    await app.initialize(
        memory_repo=SimpleMemoryRepo(),
        pattern_repo=SimplePatternRepo(),
        contradiction_repo=SimpleContradictionRepo(),
        psych_model_store=SimplePsychModelStore(),
    )

    print("Processing... (this takes 20-60 seconds, making API calls)\n")

    result = await app.ingest_note(
        text=MY_DIARY_ENTRY,
        date_str=ENTRY_DATE,
        hint_life_phase=LIFE_PHASE,
    )

    if not result.success:
        print(f"✗ Ingestion failed: {result.error}")
        return

    print(f"✓ Ingested successfully")
    print(f"  Memories extracted: {result.memories_created}")
    print(f"  Life phase detected: {result.life_phase}")
    print()

    # Show what was extracted
    import json
    from pathlib import Path

    memories_dir = Path("./data/memories")
    if memories_dir.exists():
        files = sorted(memories_dir.glob("*.json"))
        print(f"Extracted memory objects ({len(files)} total in store):")
        for f in files[-result.memories_created:]:   # show only newly created
            m = json.loads(f.read_text(encoding="utf-8"))
            print(f"\n  ─── [{m['memory_type'].upper()}] ───────────────")
            print(f"  Title:   {m['title']}")
            print(f"  Emotion: {m.get('emotion_primary','?')} "
                  f"(intensity {m.get('emotion_intensity', 0):.2f})")
            domains = m.get('domains', [])
            if domains:
                print(f"  Domains: {', '.join(str(d) for d in domains)}")
            if m.get('lesson_content'):
                print(f"  Lesson:  {m['lesson_content'][:100]}")

    print("\n" + "=" * 55)
    print("Next step: run  python chat_windows.py")
    print("Then ask:  'What does this entry reveal about me?'")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
