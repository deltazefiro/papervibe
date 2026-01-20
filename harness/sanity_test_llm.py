#!/usr/bin/env python3
"""
Basic LLM connectivity sanity test.

Quick test to verify LLM is reachable and responding.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from papervibe.llm import LLMClient, LLMSettings


async def test_llm_connectivity():
    """Test basic LLM connectivity with a simple request."""
    print("Testing LLM connectivity...")

    try:
        # Initialize client
        settings = LLMSettings()
        llm_client = LLMClient(settings=settings, concurrency=1)
        print(f"  Model: {settings.light_model}")
        print(f"  Timeout: {settings.request_timeout_seconds}s")

        # Make a simple request using the new interface
        print("\n  Sending test request...")
        result = await llm_client.complete(
            model_type="light",
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello, LLM connection test successful!' and nothing else.",
            temperature=0.3,
        )

        if result:
            print(f"  ✓ LLM responded successfully")
            print(f"  Response: {result[:100]}")
            return True
        else:
            print("  ✗ LLM returned empty response")
            return False

    except Exception as e:
        print(f"  ✗ LLM connection failed:")
        print(f"     {type(e).__name__}: {e}")
        return False


async def main():
    print("=" * 50)
    print("LLM Connectivity Sanity Test")
    print("=" * 50 + "\n")

    success = await test_llm_connectivity()

    print("\n" + "=" * 50)
    if success:
        print("✓ LLM is working")
        return 0
    else:
        print("✗ LLM is not working")
        print("\nCheck:")
        print("  - .env file exists with valid API credentials")
        print("  - Network connectivity")
        print("  - API rate limits")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
