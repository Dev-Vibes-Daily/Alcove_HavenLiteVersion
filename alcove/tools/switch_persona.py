"""Alcove — switch_persona tool.

Signals the AlcoveHandler to hand off between Persona A and Persona B.
Both personas carry this tool so either can initiate a switch.
"""

import logging
from typing import Any, Dict

from alcove.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class SwitchPersona(Tool):
    """Switch the active Alcove persona between Persona A and Persona B."""

    name = "switch_persona"
    description = (
        "Hand off the conversation to the other persona. "
        "Call this AFTER saying your handoff line to the user."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["Persona A", "Persona B"],
                "description": "The persona to switch to.",
            },
            "context": {
                "type": "string",
                "description": (
                    "Brief context for the other persona about what the user needs. "
                    "Example: 'User wants a recipe for pasta' or 'User asked about weather'"
                ),
            },
        },
        "required": ["target"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Signal a persona switch.

        Returns a structured result that AlcoveHandler intercepts
        to trigger the actual session rebuild.
        """
        target = kwargs.get("target", "Persona A")
        context = kwargs.get("context", "")

        logger.info("switch_persona called: target=%s, context=%s", target, context)

        return {
            "action": "switch_persona",
            "target": target,
            "context": context,
            "status": f"Switching to {target}",
        }
