# 🏭 Autonomous Warehouse Logistics & Path Planning Environment

> **OpenEnv-compatible** reinforcement learning environment for the **Meta × Scaler Hackathon**.  
> An LLM agent acts as a fleet manager, guiding a mobile robot through a warehouse grid to retrieve packages while avoiding obstacles and managing battery life.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Tasks & Grading](#tasks--grading)
- [Setup & Installation](#setup--installation)
- [Running Locally](#running-locally)
- [Docker Deployment](#docker-deployment)
- [Inference Script](#inference-script)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)

---

## Overview

This environment simulates an **autonomous warehouse** where an AI agent must:

1. Navigate a 2-D grid warehouse
2. Locate and collect packages (single or multi-package)
3. Avoid static and dynamic obstacles (forklift traffic)
4. Manage battery life (recharge at docking stations)
5. Minimise total steps for maximum efficiency

All observations are returned as **structured JSON/dictionaries** (no raw continuous arrays), making them directly consumable by LLM-based agents.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  LLM Agent (Client)                 │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ inference  │  │   OpenAI     │  │  Structured  │ │
│  │   .py      │──│   Client     │──│    Logs      │ │
│  └─────┬──────┘  └──────────────┘  └──────────────┘ │
└────────┼────────────────────────────────────────────┘
         │  HTTP POST /reset, /step   GET /state
         ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Server (Docker)                 │
│  ┌──────────────────────────────────────────────┐   │
│  │         WarehouseEnvironment                  │   │
│  │  ┌────────┐ ┌────────┐ ┌──────────────────┐  │   │
│  │  │ Grid   │ │Battery │ │ Dynamic Obstacle │  │   │
│  │  │ Engine │ │Manager │ │    Ticker        │  │   │
│  │  └────────┘ └────────┘ └──────────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Action Space

| Action        | Description                                     |
|---------------|-------------------------------------------------|
| `move_north`  | Move robot up one cell (row - 1)                |
| `move_south`  | Move robot down one cell (row + 1)              |
| `move_east`   | Move robot right one cell (col + 1)             |
| `move_west`   | Move robot left one cell (col - 1)              |
| `pickup`      | Pick up a package at the robot's current cell   |
| `charge`      | Recharge battery at a charging dock             |
| `wait`        | Stay in place for one step                      |

**Action model** (Pydantic):
```python
class WarehouseAction(BaseModel):
    action: str   # one of the above
    metadata: Dict[str, Any] = {}
```

## Observation Space

Every observation is a flat JSON dictionary:

| Field                | Type              | Description                               |
|----------------------|-------------------|-------------------------------------------|
| `robot_position`     | `[row, col]`      | Current robot location                    |
| `grid_size`          | `[rows, cols]`    | Warehouse dimensions                      |
| `packages`           | `list[dict]`      | `{position, value, collected}` per package|
| `adjacent_cells`     | `dict`            | N/S/E/W → `empty\|wall\|obstacle\|package\|dock` |
| `visible_obstacles`  | `list[[r,c]]`     | Obstacles within 3-cell radius            |
| `battery_level`      | `float`           | Battery % (0.0 – 100.0)                  |
| `nearest_dock`       | `[r,c] \| null`   | Closest charging station                  |
| `step_number`        | `int`             | Current step in the episode               |
| `max_steps`          | `int`             | Step budget for this task                 |
| `task_difficulty`     | `str`            | `easy \| medium \| hard`                  |
| `message`            | `str`             | Human-readable status / hints             |
| `last_action_error`  | `str \| null`     | Error from previous action                |

## Tasks & Grading

### Easy – Point-to-Point Navigation
- **Grid:** 8×8, no obstacles, no battery drain
- **Goal:** Navigate to and pick up 1 package
- **Score:** Up to **1.0** based on step efficiency
- **Budget:** 50 steps

### Medium – Obstacle Avoidance + Battery Management  
- **Grid:** 10×10, 12 static obstacles, 2 charging docks
- **Goal:** Collect 1 package while managing battery (2.5% drain/step)
- **Score:** Package retrieval (0.6) + efficiency (0.3) - battery penalty (0.1)
- **Budget:** 80 steps

### Hard – Multi-Package Retrieval with Dynamic Obstacles
- **Grid:** 12×12, 10 static + 4 dynamic obstacles, 3 docks
- **Goal:** Collect 4 packages with varying values under time pressure
- **Score:** Value-weighted progress (0.6) + efficiency (0.3) - battery penalty (0.1)
- **Budget:** 120 steps

**All scores are normalised to [0.0, 1.0].**

## Setup & Installation

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)

### Local Setup
```bash
# Clone and enter the directory
cd scaler_ws

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Locally

### Start the environment server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Test endpoints
```bash
# Health check
curl http://localhost:8000/health

# Reset (easy task)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "move_south"}}'

# State
curl http://localhost:8000/state
```

## Docker Deployment

```bash
# Build the image
docker build -t warehouse-env:latest .

# Run the container
docker run -d -p 8000:8000 --name warehouse-env warehouse-env:latest

# Verify
curl http://localhost:8000/health
```

## Inference Script

The `inference.py` script in the project root runs the LLM agent through all 3 tasks:

```bash
# Set environment variables
export API_BASE_URL="https://your-vllm-endpoint/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:8000"

# Run inference (must complete in < 20 minutes)
python inference.py
```

### Log Format
```
[START] task=warehouse-logistics env=warehouse_env_v1 model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move_south reward=0.10 done=false error=null
[STEP] step=2 action=pickup reward=0.50 done=false error=null
[END] success=true steps=2 score=1.00 rewards=0.10,0.50
```

## Project Structure

```
scaler_ws/
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                 # Container build
├── requirements.txt           # Python dependencies
├── models.py                  # Pydantic Action/Observation/State
├── inference.py               # LLM inference script (root)
├── README.md                  # This file
├── __init__.py                # Package exports
└── server/
    ├── __init__.py
    ├── warehouse_environment.py  # Core environment logic
    └── app.py                    # FastAPI server
```

## Environment Variables

| Variable       | Description                                    | Default                           |
|----------------|------------------------------------------------|-----------------------------------|
| `API_BASE_URL` | Base URL for OpenAI-compatible LLM endpoint    | `https://api.openai.com/v1`       |
| `MODEL_NAME`   | Model identifier for inference                 | `Qwen/Qwen2.5-72B-Instruct`      |
| `HF_TOKEN`     | HuggingFace / API authentication token         | *(empty)*                         |
| `ENV_URL`      | URL of the running environment server          | `http://localhost:8000`           |

---

**Built for the Meta × Scaler Hackathon** | OpenEnv Spec v1
