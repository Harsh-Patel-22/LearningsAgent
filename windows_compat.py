"""
WINDOWS COMPATIBILITY LAYER
Cognitive Memory OS

Drop-in replacements and patches for Windows-specific issues.
Import this at the top of any script that has Windows problems.

Issues addressed:
  1. asyncio event loop policy (ProactorEventLoop on Windows)
  2. PaddleOCR Windows installation path
  3. Tesseract binary location
  4. ffmpeg for Whisper
  5. Path separators (already handled by pathlib, but double-checked)
  6. Console encoding for Unicode diary content
"""

import sys
import os
import platform

IS_WINDOWS = platform.system() == "Windows"


# ─────────────────────────────────────────────
# FIX 1: asyncio event loop (CRITICAL on Windows)
# Without this, httpx async calls hang or crash
# ─────────────────────────────────────────────
if IS_WINDOWS:
    import asyncio
    # Windows default is ProactorEventLoop which breaks some async libs
    # Switch to SelectorEventLoop for compatibility with httpx + qdrant
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ─────────────────────────────────────────────
# FIX 2: Console encoding for Unicode (diary entries have special chars)
# ─────────────────────────────────────────────
if IS_WINDOWS:
    import io
    # Force UTF-8 output so diary text with special characters prints correctly
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Also set environment variable for subprocess calls
    os.environ["PYTHONIOENCODING"] = "utf-8"


# ─────────────────────────────────────────────
# FIX 3: Tesseract binary path
# Default Tesseract install location on Windows
# ─────────────────────────────────────────────
def configure_tesseract():
    """Set Tesseract path for pytesseract on Windows."""
    if not IS_WINDOWS:
        return

    try:
        import pytesseract
        # Default Tesseract install location from official installer
        default_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(
                os.environ.get("USERNAME", "")
            ),
        ]
        for path in default_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"[Windows] Tesseract found at: {path}")
                return

        print("[Windows] Tesseract not found at default locations.")
        print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Then add to PATH or set pytesseract.pytesseract.tesseract_cmd")
    except ImportError:
        pass


# ─────────────────────────────────────────────
# FIX 4: PaddleOCR import fix
# PaddleOCR on Windows sometimes has import order issues
# ─────────────────────────────────────────────
def safe_import_paddle():
    """Safely import PaddleOCR on Windows with fallback."""
    try:
        # Set this env var before importing paddle to avoid GPU detection crash
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FLAGS_use_cuda"] = "0"

        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            # On Windows, specify CPU device explicitly
            det_db_thresh=0.3,
        )
        return ocr, "paddleocr"
    except ImportError:
        print("[Windows] PaddleOCR not installed. Falling back to Tesseract.")
        return None, "tesseract"
    except Exception as e:
        print(f"[Windows] PaddleOCR init error: {e}. Falling back to Tesseract.")
        return None, "tesseract"


# ─────────────────────────────────────────────
# FIX 5: Multiprocessing start method
# Windows uses 'spawn' not 'fork' — affects Celery workers
# ─────────────────────────────────────────────
if IS_WINDOWS:
    import multiprocessing
    # Must be called in __main__ guard, but set the default here
    # Your worker scripts need: if __name__ == '__main__': multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set


# ─────────────────────────────────────────────
# UTILITY: Check all Windows dependencies
# Run this script directly to diagnose your setup
# ─────────────────────────────────────────────
def run_windows_diagnostics():
    """Check all Windows-specific dependencies and print status."""
    print("=" * 55)
    print("CMOS Windows Diagnostics")
    print("=" * 55)
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.system()} {platform.release()}")
    print()

    checks = [
        ("anthropic",          "import anthropic",                      True),
        ("openai",             "import openai",                         True),
        ("httpx",              "import httpx",                          True),
        ("pydantic",           "import pydantic; pydantic.VERSION",     True),
        ("qdrant-client",      "import qdrant_client",                  True),
        ("Pillow",             "from PIL import Image",                  True),
        ("opencv-python",      "import cv2",                            True),
        ("numpy",              "import numpy",                          True),
        ("sentence-transformers","from sentence_transformers import SentenceTransformer", True),
        ("spacy",              "import spacy",                          True),
        ("paddlepaddle",       "import paddle",                         False),
        ("paddleocr",          "from paddleocr import PaddleOCR",       False),
        ("pytesseract",        "import pytesseract",                    False),
        ("whisper",            "import whisper",                        False),
        ("redis",              "import redis",                          True),
        ("celery",             "import celery",                         False),
        ("fastapi",            "import fastapi",                        True),
        ("sqlalchemy",         "import sqlalchemy",                     True),
        ("psycopg2",           "import psycopg2",                       True),
    ]

    required_ok = 0
    optional_ok = 0
    required_fail = 0
    optional_fail = 0

    for name, import_stmt, required in checks:
        try:
            exec(import_stmt)
            status = "✓"
            label = "required" if required else "optional"
            print(f"  {status} {name:<28} [{label}]")
            if required:
                required_ok += 1
            else:
                optional_ok += 1
        except ImportError:
            status = "✗" if required else "○"
            label = "REQUIRED - install this" if required else "optional"
            print(f"  {status} {name:<28} [{label}]")
            if required:
                required_fail += 1
            else:
                optional_fail += 1

    print()
    print(f"Required:  {required_ok} ok, {required_fail} missing")
    print(f"Optional:  {optional_ok} ok, {optional_fail} missing")

    # Check Docker
    print()
    print("Infrastructure:")
    import subprocess
    for cmd, label in [
        (["docker", "ps"], "Docker"),
        (["docker", "inspect", "--format={{.State.Health.Status}}", "cmos_postgres"], "PostgreSQL"),
        (["docker", "inspect", "--format={{.State.Health.Status}}", "cmos_qdrant"], "Qdrant"),
    ]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
            )
            if result.returncode == 0:
                out = result.stdout.strip() or "running"
                print(f"  ✓ {label}: {out}")
            else:
                print(f"  ✗ {label}: not running")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ✗ {label}: not found")

    # Check Tesseract
    configure_tesseract()

    print()
    if required_fail == 0:
        print("✓ All required packages installed. Ready to run.")
    else:
        print(f"✗ {required_fail} required packages missing. Run:")
        print("  pip install -r requirements_windows.txt")


if __name__ == "__main__":
    run_windows_diagnostics()


async def call_gemini_openai(api_key: str, model: str, messages: list, system_prompt: str = None, max_tokens: int = 2048) -> str:
    import httpx
    from pii_masker import PIIMasker
    
    masker = PIIMasker()
    
    # Mask system prompt if present
    masked_system_prompt = None
    sys_mapping = {}
    if system_prompt:
        masked_system_prompt, sys_mapping = masker.mask_text(system_prompt)
        
    # Mask messages list
    masked_messages, msg_mapping = masker.mask_messages(messages)
    
    # Combine mappings
    combined_mapping = {**sys_mapping, **msg_mapping}
    
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use gemini-3.1-flash-lite as the fallback/default model since gemini-3.1-flash is not available in public API
    model_name = "gemini-3.1-flash-lite"
    if model and ("gemini" in model.lower()):
        model_name = model
    if model_name == "gemini-3.1-flash":
        model_name = "gemini-3.1-flash-lite"
        
    payload = {
        "model": model_name,
        "messages": masked_messages,
        "max_tokens": max_tokens
    }
    if masked_system_prompt:
        payload["messages"] = [{"role": "system", "content": masked_system_prompt}] + masked_messages

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        raw_response = result["choices"][0]["message"]["content"]
        
        # Unmask the response
        unmasked_response = masker.unmask_text(raw_response, combined_mapping)
        return unmasked_response
