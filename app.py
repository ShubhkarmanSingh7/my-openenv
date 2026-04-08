"""
FastAPI application for the Warehouse Logistics Environment.

Exposes the WarehouseEnvironment over HTTP REST + WebSocket endpoints,
compatible with the OpenEnv specification (step / reset / state).

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import json
import traceback
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import WarehouseAction, WarehouseObservation, WarehouseState
from server.warehouse_environment import WarehouseEnvironment

# ──────────────────────── App ────────────────────────

app = FastAPI(
    title="Warehouse Logistics Environment",
    description="OpenEnv-compatible autonomous warehouse path-planning environment",
    version="1.0.0",
)

# Global environment instance (single-session for hackathon simplicity)
_env: Optional[WarehouseEnvironment] = None


def _get_env() -> WarehouseEnvironment:
    global _env
    if _env is None:
        _env = WarehouseEnvironment(task_id="easy")
    return _env


# ──────────────────────── REST Endpoints ────────────────────────


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    global _env
    task = req.task_id or "easy"
    _env = WarehouseEnvironment(task_id=task)
    obs = _env.reset(seed=req.seed, episode_id=req.episode_id, task_id=req.task_id)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


@app.post("/step")
async def step(req: StepRequest):
    env = _get_env()
    obs = env.step(action=req.action)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


@app.get("/state")
async def state():
    env = _get_env()
    return JSONResponse(content=env.state.model_dump())


@app.get("/schema")
async def schema():
    return JSONResponse(content={
        "action": WarehouseAction.model_json_schema(),
        "observation": WarehouseObservation.model_json_schema(),
        "state": WarehouseState.model_json_schema(),
    })


# ──────────────────────── WebSocket ────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    env = WarehouseEnvironment(task_id="easy")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                data = msg.get("data", {})
                obs = env.reset(**data)
                await ws.send_json({
                    "type": "observation",
                    "data": obs.model_dump(),
                })

            elif msg_type == "step":
                data = msg.get("data", {})
                obs = env.step(action=data)
                await ws.send_json({
                    "type": "observation",
                    "data": obs.model_dump(),
                })

            elif msg_type == "state":
                await ws.send_json({
                    "type": "state",
                    "data": env.state.model_dump(),
                })

            elif msg_type == "close":
                await ws.close()
                break

            else:
                await ws.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({
                "type": "error",
                "data": {"message": str(exc), "traceback": traceback.format_exc()},
            })
        except Exception:
            pass


# ──────────────────────── Main ────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
