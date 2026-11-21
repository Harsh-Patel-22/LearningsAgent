"""
cmos_interactive.py
Interactive CLI for Cognitive Memory OS.

Provides options:
1) Add Experience/Knowledge Base (paste text, optional date)
2) Seek Advice/Remember incident/Guidance (ask anything)
3) Clear Knowledge Base / Reset CMOS
4) Exit
"""

import sys
import os
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Apply Windows event loop fixes
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

load_dotenv()

from orchestrator import CMOSApplication, CMOSConfig
from memory_repo import (
    SimpleMemoryRepo, SimplePatternRepo,
    SimpleContradictionRepo, SimplePsychModelStore
)

async def main():
    config = CMOSConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    app = CMOSApplication(config)
    
    print("Initializing CMOS Application repositories...")
    await app.initialize(
        memory_repo=SimpleMemoryRepo(),
        pattern_repo=SimplePatternRepo(),
        contradiction_repo=SimpleContradictionRepo(),
        psych_model_store=SimplePsychModelStore(),
    )
    print("CMOS Ready.")

    while True:
        print("\n" + "=" * 60)
        print("  COGNITIVE MEMORY OS - MAIN MENU")
        print("=" * 60)
        print("  1) Add Experience/Knowledge Base")
        print("  2) Seek Advice / Remember Incident / Guidance")
        print("  3) Clear Knowledge Base / Reset CMOS")
        print("  4) Exit")
        print("-" * 60)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == "4" or choice.lower() == "exit":
            print("Exiting CMOS. Goodbye!")
            break
            
        elif choice == "1":
            print("\n" + "-" * 50)
            print("OPTION 1: ADD EXPERIENCE / KNOWLEDGE BASE")
            print("-" * 50)
            
            # Get date
            date_str = input("Enter the date of the experience [YYYY-MM-DD] (Press Enter to extract automatically): ").strip()
            if not date_str:
                date_str = None
                
            print("\nEnter your experience text below.")
            print("Type 'END' on a new line and press Enter when finished pasting:\n")
            
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            
            text = "\n".join(lines).strip()
            
            if not text:
                print("No text entered. Ingestion aborted.")
                continue
                
            print("\nProcessing experience... (generating embeddings & extracting concepts)...")
            result = await app.ingest_note(
                text=text,
                date_str=date_str,
                hint_life_phase="current"
            )
            
            if result.success:
                print(f"✓ Ingested successfully!")
                print(f"  Memories extracted: {result.memories_created}")
                print("\nRunning background reflection/pattern analysis...")
                await app.run_background_analysis()
                print("✓ Background analysis update completed.")
            else:
                print(f"✗ Ingestion failed: {result.error}")
                
        elif choice == "2":
            print("\n" + "-" * 50)
            print("OPTION 2: SEEK ADVICE / REMEMBER INCIDENT / GUIDANCE")
            print("-" * 50)
            
            query = input("Ask anything (e.g. 'Why did I quit my job?', 'Give advice on dealing with work burnout'): ").strip()
            
            if not query:
                print("No query entered.")
                continue
                
            print("\nAgent (thinking)...")
            try:
                response = await app.ask(query)
                print("\n" + "=" * 55)
                print(response)
                print("=" * 55)
            except Exception as e:
                print(f"\n[Error querying agent: {e}]")
                
        elif choice == "3":
            print("\n" + "-" * 50)
            print("OPTION 3: CLEAR KNOWLEDGE BASE / RESET CMOS")
            print("-" * 50)
            confirm = input("Are you sure you want to delete all local files and Qdrant memory vectors? (y/n): ").strip().lower()
            if confirm == "y":
                print("\nResetting CMOS...")
                # 1. Clear local files
                data_path = Path("./data")
                for sub in ["source_docs", "memories", "patterns", "contradictions"]:
                    p = data_path / sub
                    if p.exists():
                        for f in p.glob("*.json"):
                            try:
                                f.unlink()
                            except Exception as e:
                                print(f"Error deleting {f}: {e}")
                
                psych_model_file = data_path / "psych_model.json"
                if psych_model_file.exists():
                    try:
                        psych_model_file.unlink()
                    except Exception as e:
                        print(f"Error deleting {psych_model_file}: {e}")

                # 2. Reset Qdrant
                try:
                    from qdrant_client import QdrantClient
                    qclient = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
                    for col in ["semantic", "emotional"]:
                        try:
                            qclient.delete_collection(col)
                            print(f"Deleted Qdrant collection: {col}")
                        except Exception as e:
                            pass
                    # Reinitialize collections
                    from memory_repo import QdrantMemoryStore
                    qstore = QdrantMemoryStore(host=config.qdrant_host, port=config.qdrant_port)
                    qstore.initialize_collections()
                    print("✓ Qdrant collections reset successfully.")
                except Exception as e:
                    print(f"Warning: Could not reset Qdrant collections: {e}")

                # 3. Reload/Reinitialize Application
                await app.initialize(
                    memory_repo=SimpleMemoryRepo(),
                    pattern_repo=SimplePatternRepo(),
                    contradiction_repo=SimpleContradictionRepo(),
                    psych_model_store=SimplePsychModelStore(),
                )
                print("\n✓ CMOS database and vector index reset successfully!")
            else:
                print("Reset cancelled.")
                
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting CMOS. Goodbye!")
