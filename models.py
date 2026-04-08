"""
Warehouse Logistics Environment — Typed Models.

Defines strictly-typed Action, Observation, and State Pydantic models for the
Autonomous Warehouse Logistics & Path Planning environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class WarehouseAction(BaseModel):
    """An action the fleet-manager agent can take each step.

    Valid ``action`` values:
        ``move_north``   – move 1 cell up   (row - 1)
        ``move_south``   – move 1 cell down  (row + 1)
        ``move_east``    – move 1 cell right (col + 1)
        ``move_west``    – move 1 cell left  (col - 1)
        ``pickup``       – pick up the package at the current cell
        ``charge``       – recharge battery at a charging dock
        ``wait``         – stay in place for one step
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    action: str = Field(
        ...,
        description=(
            "One of: move_north, move_south, move_east, move_west, "
            "pickup, charge, wait"
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata attached to the action.",
    )


# ---------------------------------------------------------------------------
# Observation  (text-based, LLM-friendly)
# ---------------------------------------------------------------------------
class WarehouseObservation(BaseModel):
    """Observation returned to the LLM agent after each step.

    All fields are human-readable text / structured dicts so that an LLM
    can reason about them without parsing continuous arrays.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    done: bool = Field(default=False, description="Whether the episode has ended.")
    reward: float = Field(default=0.0, description="Reward for the last action.")

    # ---- grid & position ----
    grid_size: List[int] = Field(
        description="[rows, cols] dimensions of the warehouse grid."
    )
    robot_position: List[int] = Field(
        description="[row, col] current position of the robot."
    )

    # ---- task targets ----
    packages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of packages still to collect. Each entry: "
            "{position: [r,c], value: int, collected: bool}"
        ),
    )

    # ---- surroundings ----
    adjacent_cells: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "What is in each adjacent direction: "
            "{'north': 'empty|wall|obstacle|package|dock', ...}"
        ),
    )
    visible_obstacles: List[List[int]] = Field(
        default_factory=list,
        description="Positions of obstacles visible within a 3-cell radius.",
    )

    # ---- battery ----
    battery_level: float = Field(
        description="Battery % remaining (0.0-100.0)."
    )
    nearest_dock: Optional[List[int]] = Field(
        default=None,
        description="[row, col] of the closest charging dock, or null.",
    )

    # ---- meta ----
    step_number: int = Field(
        default=0, description="Current step in the episode."
    )
    max_steps: int = Field(
        default=100, description="Max steps allowed for this task."
    )
    task_difficulty: str = Field(
        default="easy", description="easy | medium | hard"
    )
    message: str = Field(
        default="", description="Human-readable status / hint."
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error string if the last action was invalid, else null.",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )


# ---------------------------------------------------------------------------
# State  (internal episode bookkeeping)
# ---------------------------------------------------------------------------
class WarehouseState(BaseModel):
    """Internal episode state exposed via the ``/state`` endpoint."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )

    episode_id: Optional[str] = Field(
        default=None, description="Unique episode identifier."
    )
    step_count: int = Field(
        default=0, ge=0, description="Steps taken so far."
    )
    task_id: str = Field(
        default="easy",
        description="Current task difficulty: easy | medium | hard",
    )
    score: float = Field(
        default=0.0,
        description="Accumulated normalised score (0.0-1.0).",
    )
    packages_collected: int = Field(default=0)
    packages_total: int = Field(default=1)
    battery_level: float = Field(default=100.0)
    done: bool = Field(default=False)
