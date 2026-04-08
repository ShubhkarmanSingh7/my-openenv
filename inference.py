#!/usr/bin/env python3
"""
inference.py — Mandatory autonomous inference script for the
Warehouse Logistics OpenEnv Environment.

Runs the LLM agent through all 3 task difficulties (easy → medium → hard),
logging each episode in the strict [START]/[STEP]/[END] format required by
the hackathon validator.

Environment variables:
    API_BASE_URL   – base URL for the OpenAI-compatible LLM endpoint
    MODEL_NAME     – model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       – HuggingFace token (used as Bearer for vLLM endpoints)
    ENV_URL        – (optional) URL of the running env server
                     defaults to http://localhost:8000

Must complete in under 20 minutes total.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ────────────────────────  Configuration  ────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

MAX_RETRIES = 3
TIMEOUT = 60  # seconds per LLM call

# ────────────────────────  System Prompt  ────────────────────────

SYSTEM_PROMPT = """\
You are an autonomous warehouse fleet-management AI. You control a mobile robot
navigating a 2-D grid warehouse to collect packages and return.

## Observation format (JSON you will receive each turn)
- robot_position: [row, col]  — your current location
- grid_size: [rows, cols]     — warehouse dimensions
- packages: list of {position: [r,c], value: int, collected: bool}
- adjacent_cells: {north/south/east/west: "empty"|"wall"|"obstacle"|"package"|"dock"}
- visible_obstacles: [[r,c], ...]  — obstacles within 3-cell radius
- battery_level: 0-100  — current battery percentage
- nearest_dock: [r,c] or null  — closest charging station
- step_number / max_steps      — progress tracking
- task_difficulty: easy|medium|hard
- last_action_error: error string from previous action or null

## Valid actions (reply with EXACTLY one)
  move_north   — move up    (row - 1)
  move_south   — move down  (row + 1)
  move_east    — move right (col + 1)
  move_west    — move left  (col - 1)
  pickup       — pick up package at current cell
  charge       — recharge at a charging dock
  wait         — stay in place

## Strategy guidelines
1. Navigate toward the nearest uncollected package.
2. Avoid obstacles — check adjacent_cells before moving.
3. If battery < 25% and battery management is enabled, navigate to nearest_dock
   and use "charge" before continuing.
4. On hard difficulty, prioritize high-value packages first.
5. Minimise total steps to maximise your efficiency score.

## Response format
Reply with ONLY the action name (e.g. "move_north"). No JSON, no explanation,
no markdown — just the bare action string on a single line.
"""

# ────────────────────────  Helpers  ────────────────────────


def env_reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    """POST /reset and return the observation dict."""
    resp = httpx.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: str) -> Dict[str, Any]:
    """POST /step and return the full response dict."""
    resp = httpx.post(
        f"{ENV_URL}/step",
        json={"action": {"action": action}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> Dict[str, Any]:
    """GET /state."""
    resp = httpx.get(f"{ENV_URL}/state", timeout=15)
    resp.raise_for_status()
    return resp.json()


def ask_llm(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    """Call the LLM and extract a single action string."""
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=20,
                temperature=0.1,
            )
            raw = completion.choices[0].message.content.strip().lower()
            # Sanitise: take the first token that looks like an action
            for token in raw.replace(",", " ").split():
                cleaned = token.strip("\"'`.,!;:")
                if cleaned in {
                    "move_north", "move_south", "move_east", "move_west",
                    "pickup", "charge", "wait",
                }:
                    return cleaned
            # If parsing failed, return the raw (will produce an error in env)
            return raw.split()[0].strip("\"'`.,!;:") if raw else "wait"
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                return "wait"
            time.sleep(2 ** attempt)
    return "wait"


def fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def fmt_error(e: Optional[str]) -> str:
    return "null" if e is None else e


# ────────────────────────  Run Episode  ────────────────────────


def run_episode(client: OpenAI, task_id: str):
    """Run a single episode and emit structured logs."""
    print(f"[START] task=warehouse-logistics env=warehouse_env_v1 model={MODEL_NAME}")

    success = False
    steps = 0
    score = 0.0
    all_rewards: List[float] = []

    try:
        # ── Reset ──
        reset_resp = env_reset(task_id=task_id, seed=42)
        obs = reset_resp["observation"]
        done = reset_resp.get("done", False)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        while not done:
            # Build user message with current observation
            obs_text = json.dumps(obs, indent=2)
            messages.append({
                "role": "user",
                "content": f"Current observation:\n{obs_text}\n\nChoose your next action:",
            })

            # Ask LLM
            action = ask_llm(client, messages)

            # Step environment
            step_resp = env_step(action)
            obs = step_resp["observation"]
            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", False)
            error = obs.get("last_action_error", None)

            steps += 1
            all_rewards.append(reward)

            print(
                f"[STEP] step={steps} action={action} "
                f"reward={reward:.2f} done={fmt_bool(done)} "
                f"error={fmt_error(error)}"
            )

            # Append assistant action for context continuity
            messages.append({"role": "assistant", "content": action})

            # Keep context window manageable: sliding window
            if len(messages) > 20:
                messages = [messages[0]] + messages[-18:]

        # Final state
        st = env_state()
        score = st.get("score", 0.0)
        collected = st.get("packages_collected", 0)
        total = st.get("packages_total", 1)
        success = collected == total

    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards) if all_rewards else ""
        print(
            f"[END] success={fmt_bool(success)} steps={steps} "
            f"score={score:.2f} rewards={rewards_str}"
        )


# ────────────────────────  Main  ────────────────────────


def main():
    # Build OpenAI client pointing at the provided endpoint
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "no-key",
    )

    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"  Running task: {task}")
        print(f"{'='*60}\n")
        run_episode(client, task_id=task)
        print()


if __name__ == "__main__":
    main()
