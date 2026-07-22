"""Alcove — Persona Manager.

Manages the dual-persona system (Persona A and Persona B) with single-session
architecture. Handles persona configs, profile loading, signature-move
handoff cues, and handoff sequencing.

Persona differentiation lives in voice + a signature dance move per handoff.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

from alcove.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


@dataclass
class PersonaConfig:
    """Configuration for a single Alcove persona."""

    name: str
    voice: str  # OpenAI Realtime voice ID
    profile_dir: str  # Folder name under profiles/
    signature_move: str  # Name of a move in reachy_mini_dances_library.AVAILABLE_MOVES
    idle_style: str = "contemplative"  # Key into alcove.moves.IDLE_STYLES
    instructions: str = ""
    tool_names: list = field(default_factory=list)


PERSONAS: Dict[str, PersonaConfig] = {
    "Persona A": PersonaConfig(
        name="Persona A",
        voice="marin",
        profile_dir="persona_a",
        signature_move="polyrhythm_combo",
        # Slower, mindful idle rhythm — visually distinct from Persona B.
        idle_style="contemplative",
    ),
    "Persona B": PersonaConfig(
        name="Persona B",
        voice="coral",
        profile_dir="persona_b",
        signature_move="groovy_sway_and_roll",
        # Quicker, more active idle rhythm — reads as still-tuned-in energy.
        idle_style="energetic",
    ),
}


def _load_persona_instructions(persona: PersonaConfig) -> str:
    """Load instructions.txt for a persona profile."""
    from alcove.prompts import _expand_prompt_includes

    profiles_dir = Path(__file__).parent.parent / "profiles" / persona.profile_dir
    instructions_path = profiles_dir / "instructions.txt"
    if instructions_path.exists():
        raw = instructions_path.read_text(encoding="utf-8").strip()
        instructions = _expand_prompt_includes(raw)
    else:
        logger.warning("No instructions.txt for %s at %s", persona.name, instructions_path)
        instructions = f"You are {persona.name}."

    return instructions


def _load_persona_tools(persona: PersonaConfig) -> list:
    """Load tools.txt for a persona profile."""
    profiles_dir = Path(__file__).parent.parent / "profiles" / persona.profile_dir
    tools_path = profiles_dir / "tools.txt"
    if tools_path.exists():
        return [
            line.strip()
            for line in tools_path.read_text(encoding="utf-8").strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    logger.warning("No tools.txt for %s", persona.name)
    return []


class PersonaManager:
    """Manages dual persona configs and handoff logic.

    Works with a single active OpenAI Realtime session. Provides config
    for session setup and handles the sequencing of persona switches
    (signature move, context, active-persona swap).
    """

    def __init__(self, deps: ToolDependencies):
        self.deps = deps
        self.active_persona: str = "Persona A"
        self._switching = False
        self._handoff_context: Optional[str] = None

        self._register_persona_tools()

        for name, persona in PERSONAS.items():
            persona.instructions = _load_persona_instructions(persona)
            persona.tool_names = _load_persona_tools(persona)
            logger.info(
                "Loaded persona %s: voice=%s, tools=%s, instructions=%d chars",
                name, persona.voice, persona.tool_names, len(persona.instructions),
            )

    @staticmethod
    def _register_persona_tools() -> None:
        """Import any tools referenced by persona tools.txt files that the
        framework didn't already load.

        core_tools._load_profile_tools() only imports tools listed in the
        active framework profile's tools.txt. Alcove personas may reference
        tools outside that profile (switch_persona, web_search) — this method
        ensures every persona's tool list is importable and registered.
        """
        import importlib
        from alcove.tools import core_tools

        known = set(core_tools.ALL_TOOLS.keys())
        added = []

        for persona in PERSONAS.values():
            for tool_name in _load_persona_tools(persona):
                if tool_name in known:
                    continue
                try:
                    importlib.import_module(f"alcove.tools.{tool_name}")
                    added.append(tool_name)
                    known.add(tool_name)
                except ModuleNotFoundError:
                    logger.warning(
                        "Persona references tool '%s' but no module found at "
                        "alcove.tools.%s — skipping",
                        tool_name, tool_name,
                    )

        if added:
            for cls in core_tools.get_concrete_subclasses(core_tools.Tool):
                if cls.name not in core_tools.ALL_TOOLS:
                    core_tools.ALL_TOOLS[cls.name] = cls()
            logger.info("Registered additional persona tools: %s", added)

    @property
    def active_config(self) -> PersonaConfig:
        return PERSONAS[self.active_persona]

    @property
    def inactive_persona(self) -> str:
        return "Persona B" if self.active_persona == "Persona A" else "Persona A"

    def get_tool_specs(self, persona_name: str) -> list:
        """Build OpenAI tool specs for a persona's tool set."""
        from alcove.tools.core_tools import ALL_TOOLS

        persona = PERSONAS[persona_name]
        specs = []
        for tool_name in persona.tool_names:
            if tool_name in ALL_TOOLS:
                specs.append(ALL_TOOLS[tool_name].spec())
            else:
                logger.warning("Tool '%s' not found for persona %s", tool_name, persona_name)
        return specs

    def get_session_config(self, persona_name: str) -> Dict[str, Any]:
        """Build the OpenAI Realtime session configuration for a persona."""
        persona = PERSONAS[persona_name]
        return {
            "type": "realtime",
            "instructions": persona.instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": persona.voice,
                },
            },
            "tools": self.get_tool_specs(persona_name),
            "tool_choice": "auto",
        }

    async def perform_switch(self, target: str, context: str = "") -> bool:
        """Execute the visual/animation portion of a persona handoff.

        Returns True if switch was performed, False if skipped.
        The caller (AlcoveHandler) is responsible for rebuilding the session.
        """
        if self._switching:
            logger.warning("Switch already in progress, ignoring")
            return False

        if target == self.active_persona:
            logger.info("Already on %s, ignoring switch", target)
            return False

        self._switching = True
        old_persona = self.active_persona

        try:
            logger.info("Persona switch: %s -> %s", old_persona, target)

            await asyncio.sleep(0.3)

            new_config = PERSONAS[target]
            try:
                from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
                from alcove.dance_emotion_moves import DanceQueueMove

                move_name = new_config.signature_move
                if move_name in AVAILABLE_MOVES:
                    self.deps.movement_manager.queue_move(DanceQueueMove(move_name))
                    logger.info("Queued signature move for %s: %s", new_config.name, move_name)
                else:
                    logger.warning(
                        "Signature move '%s' not in AVAILABLE_MOVES; skipping handoff cue",
                        move_name,
                    )
            except Exception as e:
                logger.debug("Signature move failed (expected in sim): %s", e)

            self.active_persona = target
            self._handoff_context = context

            logger.info("Now active: %s (context: %s)", target, context or "none")
            return True

        finally:
            self._switching = False

    def consume_handoff_context(self) -> Optional[str]:
        """Consume and return any pending handoff context."""
        ctx = self._handoff_context
        self._handoff_context = None
        return ctx

    def get_status(self) -> Dict[str, Any]:
        """Return current state for debugging."""
        return {
            "active_persona": self.active_persona,
            "active_voice": self.active_config.voice,
            "switching": self._switching,
            "personas": {
                name: {
                    "voice": cfg.voice,
                    "tools": cfg.tool_names,
                    "signature_move": cfg.signature_move,
                }
                for name, cfg in PERSONAS.items()
            },
        }
