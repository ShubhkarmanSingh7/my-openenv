"""
Warehouse Logistics Environment — Core Implementation.

Implements the full grid-world warehouse simulation with:
  • 3 difficulty levels (easy / medium / hard)
  • Battery drain & recharge mechanics
  • Static and dynamic obstacles
  • Multi-package retrieval with value priorities
  • Normalised grader returning 0.0 – 1.0 rewards

All state is kept in lightweight Python dicts/lists (no heavy physics).
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone fallback — define minimal base classes
    class Environment:
        """Minimal Environment base when openenv-core is not installed."""
        pass

    from pydantic import BaseModel

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0

    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import WarehouseAction, WarehouseObservation, WarehouseState


# ──────────────────────────  Constants  ──────────────────────────

VALID_ACTIONS = {
    "move_north", "move_south", "move_east", "move_west",
    "pickup", "charge", "wait",
}

DIRECTION_DELTAS = {
    "move_north": (-1, 0),
    "move_south": (1, 0),
    "move_east": (0, 1),
    "move_west": (0, -1),
}

# Task configurations
TASK_CONFIG = {
    "easy": {
        "grid_rows": 8,
        "grid_cols": 8,
        "num_packages": 1,
        "num_obstacles": 0,
        "dynamic_obstacles": 0,
        "battery_enabled": False,
        "initial_battery": 100.0,
        "battery_drain_per_step": 0.0,
        "num_docks": 0,
        "max_steps": 50,
        "optimal_steps": 10,  # approximate
    },
    "medium": {
        "grid_rows": 10,
        "grid_cols": 10,
        "num_packages": 1,
        "num_obstacles": 12,
        "dynamic_obstacles": 0,
        "battery_enabled": True,
        "initial_battery": 100.0,
        "battery_drain_per_step": 2.5,
        "num_docks": 2,
        "max_steps": 80,
        "optimal_steps": 18,
    },
    "hard": {
        "grid_rows": 12,
        "grid_cols": 12,
        "num_packages": 4,
        "num_obstacles": 10,
        "dynamic_obstacles": 4,
        "battery_enabled": True,
        "initial_battery": 100.0,
        "battery_drain_per_step": 1.8,
        "num_docks": 3,
        "max_steps": 120,
        "optimal_steps": 30,
    },
}

# Cell types for internal grid
EMPTY = 0
WALL = -1
OBSTACLE_STATIC = 1
OBSTACLE_DYNAMIC = 2
PACKAGE = 3
DOCK = 4
ROBOT = 5

CELL_NAMES = {
    EMPTY: "empty",
    WALL: "wall",
    OBSTACLE_STATIC: "obstacle",
    OBSTACLE_DYNAMIC: "obstacle",
    PACKAGE: "package",
    DOCK: "dock",
    ROBOT: "robot",
}


# ──────────────────────────  Environment  ──────────────────────────

class WarehouseEnvironment(Environment):
    """Autonomous Warehouse Logistics & Path Planning Environment."""

    def __init__(self, task_id: str = "easy"):
        super().__init__()
        self._task_id = task_id
        self._cfg = TASK_CONFIG[task_id]
        self._state = WarehouseState(episode_id=str(uuid4()), task_id=task_id)
        self._grid: List[List[int]] = []
        self._robot_pos: List[int] = [0, 0]
        self._packages: List[Dict[str, Any]] = []
        self._docks: List[List[int]] = []
        self._dynamic_obs: List[Dict[str, Any]] = []
        self._battery: float = 100.0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._last_action_error: Optional[str] = None
        self._rewards_history: List[float] = []
        self._rng = random.Random(42)

    # ─────────────  reset  ─────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WarehouseObservation:
        if task_id and task_id in TASK_CONFIG:
            self._task_id = task_id
            self._cfg = TASK_CONFIG[task_id]

        self._rng = random.Random(seed if seed is not None else random.randint(0, 2**31))
        self._state = WarehouseState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task_id,
        )

        cfg = self._cfg
        rows, cols = cfg["grid_rows"], cfg["grid_cols"]

        # Build empty grid
        self._grid = [[EMPTY] * cols for _ in range(rows)]

        # Place robot at a random clear cell (top-left quadrant preferred)
        self._robot_pos = self._random_empty_cell(
            row_range=(0, rows // 3), col_range=(0, cols // 3)
        )

        # Place packages
        self._packages = []
        for i in range(cfg["num_packages"]):
            pos = self._random_empty_cell(
                row_range=(rows // 2, rows - 1), col_range=(cols // 2, cols - 1)
            )
            value = 1 if self._task_id != "hard" else self._rng.choice([1, 2, 3, 5])
            self._packages.append({
                "position": pos,
                "value": value,
                "collected": False,
            })
            self._grid[pos[0]][pos[1]] = PACKAGE

        # Place charging docks
        self._docks = []
        for _ in range(cfg["num_docks"]):
            pos = self._random_empty_cell()
            self._docks.append(pos)
            self._grid[pos[0]][pos[1]] = DOCK

        # Place static obstacles
        for _ in range(cfg["num_obstacles"]):
            pos = self._random_empty_cell()
            self._grid[pos[0]][pos[1]] = OBSTACLE_STATIC

        # Place dynamic obstacles (hard mode)
        self._dynamic_obs = []
        for _ in range(cfg["dynamic_obstacles"]):
            pos = self._random_empty_cell()
            direction = self._rng.choice(list(DIRECTION_DELTAS.keys()))
            self._dynamic_obs.append({"position": pos, "direction": direction})
            self._grid[pos[0]][pos[1]] = OBSTACLE_DYNAMIC

        # Battery
        self._battery = cfg["initial_battery"]

        # Bookkeeping
        self._done = False
        self._cumulative_reward = 0.0
        self._last_action_error = None
        self._rewards_history = []

        self._state.packages_total = cfg["num_packages"]
        self._state.packages_collected = 0
        self._state.battery_level = self._battery
        self._state.score = 0.0
        self._state.done = False

        return self._build_observation(reward=0.0)

    # ─────────────  step  ─────────────

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> WarehouseObservation:
        if self._done:
            return self._build_observation(
                reward=0.0,
                error="Episode already finished. Call reset().",
            )

        # Parse action string
        if isinstance(action, dict):
            act_str = action.get("action", "")
        elif isinstance(action, WarehouseAction):
            act_str = action.action
        elif hasattr(action, "action"):
            act_str = action.action
        else:
            act_str = str(action)

        act_str = act_str.strip().lower()

        if act_str not in VALID_ACTIONS:
            return self._build_observation(
                reward=0.0,
                error=f"Invalid action '{act_str}'. Valid: {sorted(VALID_ACTIONS)}",
            )

        self._last_action_error = None
        reward = 0.0
        cfg = self._cfg

        # ── Movement actions ──
        if act_str in DIRECTION_DELTAS:
            dr, dc = DIRECTION_DELTAS[act_str]
            nr, nc = self._robot_pos[0] + dr, self._robot_pos[1] + dc

            if not self._in_bounds(nr, nc):
                self._last_action_error = "Cannot move: wall boundary."
                reward = -0.02
            elif self._grid[nr][nc] in (OBSTACLE_STATIC, OBSTACLE_DYNAMIC):
                self._last_action_error = "Cannot move: obstacle in the way."
                reward = -0.02
            else:
                self._robot_pos = [nr, nc]
                reward = -0.01  # small step penalty for efficiency

        # ── Pickup ──
        elif act_str == "pickup":
            picked = False
            for pkg in self._packages:
                if not pkg["collected"] and pkg["position"] == self._robot_pos:
                    pkg["collected"] = True
                    self._state.packages_collected += 1
                    value_bonus = pkg["value"] / max(
                        sum(p["value"] for p in self._packages), 1
                    )
                    reward = 0.3 + 0.2 * value_bonus  # 0.3 – 0.5 per pickup
                    picked = True
                    break
            if not picked:
                self._last_action_error = "No package at current position."
                reward = -0.02

        # ── Charge ──
        elif act_str == "charge":
            if self._robot_pos in self._docks:
                old = self._battery
                self._battery = min(100.0, self._battery + 40.0)
                recharged = self._battery - old
                reward = 0.05 * (recharged / 40.0)  # small reward proportional
            else:
                self._last_action_error = "Not at a charging dock."
                reward = -0.02

        # ── Wait ──
        elif act_str == "wait":
            reward = -0.01

        # ── Battery drain ──
        if cfg["battery_enabled"]:
            self._battery -= cfg["battery_drain_per_step"]
            if self._battery <= 0:
                self._battery = 0.0
                self._done = True
                reward = -0.3  # heavy penalty for running out

        # ── Move dynamic obstacles ──
        self._tick_dynamic_obstacles()

        # ── Step bookkeeping ──
        self._state.step_count += 1
        self._state.battery_level = self._battery

        # ── Check terminal conditions ──
        all_collected = all(p["collected"] for p in self._packages)
        if all_collected:
            self._done = True
            # Efficiency bonus: closer to optimal → higher bonus
            efficiency = max(0.0, 1.0 - (
                (self._state.step_count - cfg["optimal_steps"])
                / max(cfg["max_steps"] - cfg["optimal_steps"], 1)
            ))
            reward += 0.3 * efficiency  # up to 0.3 bonus

        if self._state.step_count >= cfg["max_steps"]:
            self._done = True

        # ── Normalise and clamp reward to [0, 1] (or allow small negatives for feedback)
        reward = max(-0.5, min(1.0, reward))
        self._cumulative_reward += reward
        self._rewards_history.append(round(reward, 4))

        # ── Compute normalised score ──
        self._state.score = self._compute_score()
        self._state.done = self._done

        return self._build_observation(reward=round(reward, 4))

    # ─────────────  state property  ─────────────

    @property
    def state(self) -> WarehouseState:
        return self._state

    # ─────────────  helpers  ─────────────

    def _build_observation(
        self,
        reward: float = 0.0,
        error: Optional[str] = None,
    ) -> WarehouseObservation:
        if error:
            self._last_action_error = error

        cfg = self._cfg
        rows, cols = cfg["grid_rows"], cfg["grid_cols"]

        return WarehouseObservation(
            done=self._done,
            reward=reward,
            grid_size=[rows, cols],
            robot_position=list(self._robot_pos),
            packages=[
                {
                    "position": list(p["position"]),
                    "value": p["value"],
                    "collected": p["collected"],
                }
                for p in self._packages
            ],
            adjacent_cells=self._get_adjacent_cells(),
            visible_obstacles=self._get_visible_obstacles(radius=3),
            battery_level=round(self._battery, 1),
            nearest_dock=self._nearest_dock(),
            step_number=self._state.step_count,
            max_steps=cfg["max_steps"],
            task_difficulty=self._task_id,
            message=self._status_message(),
            last_action_error=self._last_action_error,
        )

    def _get_adjacent_cells(self) -> Dict[str, str]:
        result = {}
        for name, (dr, dc) in [
            ("north", (-1, 0)), ("south", (1, 0)),
            ("east", (0, 1)), ("west", (0, -1)),
        ]:
            nr = self._robot_pos[0] + dr
            nc = self._robot_pos[1] + dc
            if not self._in_bounds(nr, nc):
                result[name] = "wall"
            else:
                result[name] = CELL_NAMES.get(self._grid[nr][nc], "empty")
                # check if a package is on that cell
                for p in self._packages:
                    if not p["collected"] and p["position"] == [nr, nc]:
                        result[name] = "package"
                        break
                # check dock
                if [nr, nc] in self._docks:
                    result[name] = "dock"
        return result

    def _get_visible_obstacles(self, radius: int = 3) -> List[List[int]]:
        obstacles = []
        r0, c0 = self._robot_pos
        rows, cols = self._cfg["grid_rows"], self._cfg["grid_cols"]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if self._grid[nr][nc] in (OBSTACLE_STATIC, OBSTACLE_DYNAMIC):
                        obstacles.append([nr, nc])
        return obstacles

    def _nearest_dock(self) -> Optional[List[int]]:
        if not self._docks:
            return None
        best = None
        best_dist = float("inf")
        for d in self._docks:
            dist = abs(d[0] - self._robot_pos[0]) + abs(d[1] - self._robot_pos[1])
            if dist < best_dist:
                best_dist = dist
                best = d
        return list(best) if best else None

    def _status_message(self) -> str:
        collected = sum(1 for p in self._packages if p["collected"])
        total = len(self._packages)
        if self._done:
            if collected == total:
                return f"All {total} package(s) collected! Episode complete."
            elif self._battery <= 0:
                return "Battery depleted! Episode over."
            else:
                return f"Step limit reached. Collected {collected}/{total} packages."
        parts = [f"Packages: {collected}/{total}"]
        if self._cfg["battery_enabled"]:
            parts.append(f"Battery: {self._battery:.0f}%")
            if self._battery < 25:
                parts.append("⚠ LOW BATTERY — find a charging dock!")
        return " | ".join(parts)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._cfg["grid_rows"] and 0 <= c < self._cfg["grid_cols"]

    def _random_empty_cell(
        self,
        row_range: Optional[Tuple[int, int]] = None,
        col_range: Optional[Tuple[int, int]] = None,
    ) -> List[int]:
        rows, cols = self._cfg["grid_rows"], self._cfg["grid_cols"]
        r_lo, r_hi = row_range or (0, rows - 1)
        c_lo, c_hi = col_range or (0, cols - 1)
        r_lo = max(0, r_lo)
        r_hi = min(rows - 1, r_hi)
        c_lo = max(0, c_lo)
        c_hi = min(cols - 1, c_hi)

        for _ in range(500):
            r = self._rng.randint(r_lo, r_hi)
            c = self._rng.randint(c_lo, c_hi)
            if self._grid[r][c] == EMPTY and [r, c] != self._robot_pos:
                return [r, c]
        # fallback: scan
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if self._grid[r][c] == EMPTY and [r, c] != self._robot_pos:
                    return [r, c]
        return [r_lo, c_lo]

    def _tick_dynamic_obstacles(self) -> None:
        """Move each dynamic obstacle one step in its current direction,
        bouncing off walls / static objects (simulates forklift traffic)."""
        rows, cols = self._cfg["grid_rows"], self._cfg["grid_cols"]
        for obs in self._dynamic_obs:
            pos = obs["position"]
            d = obs["direction"]
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = pos[0] + dr, pos[1] + dc

            # clear old cell
            if self._grid[pos[0]][pos[1]] == OBSTACLE_DYNAMIC:
                self._grid[pos[0]][pos[1]] = EMPTY

            if (
                self._in_bounds(nr, nc)
                and self._grid[nr][nc] == EMPTY
                and [nr, nc] != self._robot_pos
            ):
                obs["position"] = [nr, nc]
            else:
                # bounce: reverse direction
                reverse = {
                    "move_north": "move_south",
                    "move_south": "move_north",
                    "move_east": "move_west",
                    "move_west": "move_east",
                }
                obs["direction"] = reverse[d]

            self._grid[obs["position"][0]][obs["position"][1]] = OBSTACLE_DYNAMIC

    def _compute_score(self) -> float:
        """Return a normalised score in [0.0, 1.0] based on task grader logic."""
        cfg = self._cfg
        collected = self._state.packages_collected
        total = self._state.packages_total

        if total == 0:
            return 1.0

        # Partial progress component  (0 – 0.6)
        if self._task_id == "hard":
            # weight by package value
            total_value = sum(p["value"] for p in self._packages)
            collected_value = sum(
                p["value"] for p in self._packages if p["collected"]
            )
            progress = (collected_value / max(total_value, 1)) * 0.6
        else:
            progress = (collected / total) * 0.6

        # Efficiency component  (0 – 0.3)
        if collected == total:
            eff = max(0.0, 1.0 - (
                (self._state.step_count - cfg["optimal_steps"])
                / max(cfg["max_steps"] - cfg["optimal_steps"], 1)
            ))
        else:
            eff = 0.0
        efficiency = eff * 0.3

        # Battery penalty (medium/hard only)  (0 – -0.1)
        battery_penalty = 0.0
        if cfg["battery_enabled"] and self._battery <= 0:
            battery_penalty = -0.1

        score = max(0.0, min(1.0, progress + efficiency + battery_penalty))
        return round(score, 4)

    # ─────────────  Convenience  ─────────────

    @property
    def rewards_history(self) -> List[float]:
        return self._rewards_history
