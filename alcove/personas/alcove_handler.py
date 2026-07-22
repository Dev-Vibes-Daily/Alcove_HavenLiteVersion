"""Alcove v4 — Single-Session Persona Handler.

Extends OpenaiRealtimeHandler with Alcove's dual-persona system using a
single active session at a time. On persona switch, the current session
is closed and rebuilt with the new persona's voice/instructions/tools.

This eliminates the race conditions from v3's dual-concurrent-session
architecture while keeping distinct voices per persona.
"""

import os
import json
import base64
import asyncio
import logging
from typing import Any, Dict, Literal, Optional, Tuple
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python <3.9 fallback (Alcove requires 3.10+, so this is defensive)
    ZoneInfo = None  # type: ignore

import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from alcove.config import config
from alcove.openai_realtime import (
    OpenaiRealtimeHandler,
    OPEN_AI_INPUT_SAMPLE_RATE,
    OPEN_AI_OUTPUT_SAMPLE_RATE,
)
from alcove.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
)
from alcove.personas.persona_manager import PersonaManager
from alcove.personas.memory import MemoryJournal


logger = logging.getLogger(__name__)


class AlcoveHandler(OpenaiRealtimeHandler):
    """Single-session OpenAI Realtime handler for Alcove.

    Architecture:
        Audio In -> [Single OpenAI Session] -> Audio Out
                          |
               switch_persona tool call
                          |
               1. Close current session
               2. Eye color + animation
               3. Open new session with new persona
               4. Inject handoff context
    """

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        super().__init__(deps, gradio_mode, instance_path)

        # Reuse stateful objects already attached to deps (for .copy() calls).
        # Otherwise create fresh ones. This ensures persona state and
        # conversation memory are shared across all handler copies.
        self.persona_manager = getattr(deps, "_persona_manager", None) or PersonaManager(deps)
        self.memory = getattr(deps, "_memory_journal", None) or MemoryJournal()

        # Attach to deps so all copies and tools share these single instances.
        deps._persona_manager = self.persona_manager  # type: ignore[attr-defined]
        deps._memory_journal = self.memory  # type: ignore[attr-defined]

        # Note: previously this set an initial LED eye color from
        # active_config.eye_color. The reachy-mini SDK doesn't expose
        # eye-color control on this hardware, so we removed the no-op
        # call. Persona presence is now signaled via voice + signature
        # dance moves on handoff.

    def copy(self) -> "AlcoveHandler":
        return AlcoveHandler(self.deps, self.gradio_mode, self.instance_path)

    def _get_time_context(self) -> str:
        """Build a time-of-day context block for the session prompt.

        Categorizes the current local hour into an energy bucket so the
        active persona can attune tone and pacing to the moment.

        Timezone is read from the ALCOVE_TIMEZONE environment variable
        (e.g. "America/Denver", "Europe/Paris", "Asia/Tokyo"). If unset
        or invalid, falls back to the system's local time. Language is
        deliberately neutral (they/them, no assumptions about the user)
        so any persona and any user can inherit it.
        """
        tz_name = os.environ.get("ALCOVE_TIMEZONE", "").strip()
        tz = None
        if tz_name and ZoneInfo is not None:
            try:
                tz = ZoneInfo(tz_name)
            except Exception:
                logger.debug("ALCOVE_TIMEZONE=%r invalid; falling back to system time", tz_name)
        now = datetime.now(tz) if tz else datetime.now()
        hour = now.hour
        time_str = now.strftime("%-I:%M %p") if os.name != "nt" else now.strftime("%#I:%M %p")
        day_of_week = now.strftime("%A")
        tz_label = f" ({tz_name})" if tz_name and tz else ""

        if 5 <= hour < 8:
            bucket = "early morning"
            energy = (
                "The user may be just waking up. Speak softly, don't overwhelm. "
                "Match a quiet, gentle presence."
            )
        elif 8 <= hour < 11:
            bucket = "morning"
            energy = "Warm, encouraging, curious. Ready to help but not pushy."
        elif 11 <= hour < 14:
            bucket = "midday"
            energy = "Engaged, present, ready for whatever."
        elif 14 <= hour < 17:
            bucket = "afternoon"
            energy = (
                "Energy may be dipping — offer a gentle presence, steady but "
                "not sluggish."
            )
        elif 17 <= hour < 20:
            bucket = "evening"
            energy = "Winding-down time. Warmer, more reflective, softer voice."
        elif 20 <= hour < 24:
            bucket = "night"
            energy = (
                "Quiet, reflective, warm. Not a time for high energy or complex "
                "tasks unless asked."
            )
        else:  # 0 <= hour < 5
            bucket = "late night"
            energy = (
                "Late night. If the user is up, meet them softly — don't project "
                "cheer they may not feel."
            )

        return (
            f"## CURRENT CONTEXT\n"
            f"Local time: {time_str} on {day_of_week}{tz_label}.\n"
            f"Time-of-day energy: {bucket}. {energy}"
        )

    async def _run_realtime_session(self) -> None:
        """Override: build session config from active persona instead of global profile."""
        persona_name = self.persona_manager.active_persona
        session_config = self.persona_manager.get_session_config(persona_name)

        # Sync the per-persona idle body language to MovementManager. Applied on
        # the next idle trigger; any breathing move already in flight finishes
        # with its previous params. Wrapped for simulator environments where
        # movement_manager may not be present.
        try:
            idle_style = self.persona_manager.active_config.idle_style
            self.deps.movement_manager.set_idle_style(idle_style)
        except Exception as e:
            logger.debug("Could not sync idle style (likely simulator): %s", e)

        # Inject time-of-day so the persona attunes tone/pacing to the hour.
        # Set ALCOVE_TIMEZONE env var (e.g. "America/Denver") to control this;
        # otherwise the system's local time is used.
        session_config["instructions"] += "\n\n" + self._get_time_context()

        # Inject recent conversation memory into instructions
        memory_context = self.memory.get_context_string(limit=10)
        if memory_context:
            session_config["instructions"] += (
                "\n\n## RECENT CONVERSATION HISTORY\n"
                "Here's what was said recently (use this for continuity):\n"
                + memory_context
            )

        async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
            try:
                await conn.session.update(session=session_config)
                logger.info(
                    "Alcove session initialized: persona=%s, voice=%s",
                    persona_name,
                    self.persona_manager.active_config.voice,
                )
                self._persist_api_key_if_needed()
            except Exception:
                logger.exception("Alcove session.update failed; aborting")
                return

            # Inject handoff context if this is a persona switch
            handoff_ctx = self.persona_manager.consume_handoff_context()
            if handoff_ctx:
                try:
                    # Cancel any in-flight response from the outgoing persona.
                    # Without this, OpenAI rejects our response.create() below
                    # with [conversation_already_has_active_response] — the
                    # incoming persona then handles tool calls silently but
                    # never speaks the greeting. Cancel is best-effort: if
                    # nothing is in flight OpenAI ignores it.
                    try:
                        await conn.response.cancel()
                    except Exception:
                        pass

                    await conn.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        f"[System: You just took over from your teammate. "
                                        f"Context: {handoff_ctx}. Greet the user briefly and help them.]"
                                    ),
                                },
                            ],
                        },
                    )
                    await conn.response.create()
                    logger.info("Injected handoff context for %s: %s", persona_name, handoff_ctx)
                except Exception as e:
                    logger.warning("Failed to inject handoff context: %s", e)

            # Main event loop — identical to parent except for switch_persona interception
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass

            async for event in self.connection:
                logger.debug("OpenAI event: %s", event.type)

                if event.type == "input_audio_buffer.speech_started":
                    if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                        self._clear_queue()
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()
                    self.deps.movement_manager.set_listening(True)

                if event.type == "input_audio_buffer.speech_stopped":
                    self.deps.movement_manager.set_listening(False)

                if event.type in (
                    "response.audio.done",
                    "response.output_audio.done",
                    "response.audio.completed",
                    "response.completed",
                ):
                    logger.debug("response completed")

                # Partial transcription (debounced)
                if event.type == "conversation.item.input_audio_transcription.partial":
                    self.partial_transcript_sequence += 1
                    current_sequence = self.partial_transcript_sequence
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass
                    self.partial_transcript_task = asyncio.create_task(
                        self._emit_debounced_partial(event.transcript, current_sequence)
                    )

                # Completed user transcription
                if event.type == "conversation.item.input_audio_transcription.completed":
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": event.transcript})
                    )
                    # Save to memory
                    self.memory.save_interaction("user", event.transcript, persona=persona_name)

                # Assistant transcription
                if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": event.transcript})
                    )
                    # Save to memory
                    self.memory.save_interaction("assistant", event.transcript, persona=persona_name)

                # Audio delta
                if event.type in ("response.audio.delta", "response.output_audio.delta"):
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.feed(event.delta)
                    self.last_activity_time = asyncio.get_event_loop().time()
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1),
                        ),
                    )

                # Tool calling
                if event.type == "response.function_call_arguments.done":
                    tool_name = getattr(event, "name", None)
                    args_json_str = getattr(event, "arguments", None)
                    call_id = getattr(event, "call_id", None)

                    if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                        logger.error("Invalid tool call: tool_name=%s, args=%s", tool_name, args_json_str)
                        continue

                    try:
                        tool_result = await asyncio.wait_for(
                            dispatch_tool_call(tool_name, args_json_str, self.deps),
                            timeout=15.0,
                        )
                        logger.debug("Tool '%s' result: %s", tool_name, tool_result)
                    except asyncio.TimeoutError:
                        logger.error("Tool '%s' timed out after 15s", tool_name)
                        tool_result = {"error": f"Tool '{tool_name}' took too long. Please try again."}
                    except Exception as e:
                        logger.error("Tool '%s' failed: %s", tool_name, e)
                        tool_result = {"error": str(e)}

                    # Check if this is a persona switch
                    if tool_name == "switch_persona" and tool_result.get("action") == "switch_persona":
                        target = tool_result.get("target", "Persona A")
                        context = tool_result.get("context", "")

                        # Send tool output to close the function call cleanly
                        if isinstance(call_id, str):
                            await self.connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps(tool_result),
                                },
                            )

                        await self.output_queue.put(
                            AdditionalOutputs({
                                "role": "assistant",
                                "content": f"Switching to {target}...",
                                "metadata": {"title": f"Switching to {target}", "status": "done"},
                            })
                        )

                        # Perform the switch — this closes the current session
                        switched = await self.persona_manager.perform_switch(target, context)
                        if switched:
                            # Break out of the event loop — _restart_session will open a new one
                            logger.info("Session ending for persona switch to %s", target)
                            return  # Exit _run_realtime_session; start_up retry loop will reconnect
                        continue

                    # Normal tool result handling
                    if isinstance(call_id, str):
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps(tool_result),
                            },
                        )

                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": json.dumps(tool_result),
                            "metadata": {"title": f"Used tool {tool_name}", "status": "done"},
                        })
                    )

                    # Handle camera image injection
                    if tool_name == "camera" and "b64_im" in tool_result:
                        import cv2
                        b64_im = tool_result["b64_im"]
                        if not isinstance(b64_im, str):
                            b64_im = str(b64_im)
                        await self.connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_im}",
                                    },
                                ],
                            },
                        )
                        if self.deps.camera_worker is not None:
                            np_img = self.deps.camera_worker.get_latest_frame()
                            if np_img is not None:
                                rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                            else:
                                rgb_frame = None
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": gr.Image(value=rgb_frame)})
                            )

                    # Let the robot respond to tool results
                    if self.is_idle_tool_call:
                        self.is_idle_tool_call = False
                    else:
                        # Cancel any response still streaming from before the
                        # tool ran — OpenAI rejects response.create() with
                        # [conversation_already_has_active_response] when one
                        # is in flight, which causes the lips/head to keep
                        # moving but no audio to emit.
                        try:
                            await self.connection.response.cancel()
                        except Exception:
                            pass
                        await self.connection.response.create(
                            response={
                                "instructions": "Use the tool result just returned and answer concisely in speech.",
                            },
                        )

                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()

                # Server errors
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err) if err else "unknown error")
                    code = getattr(err, "code", "")
                    logger.error("Realtime error [%s]: %s", code, msg)
                    if code not in ("input_audio_buffer_commit_empty", "conversation_already_has_active_response"):
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                        )

    async def start_up(self) -> None:
        """Override start_up to handle persona switch reconnection.

        When a persona switch triggers a return from _run_realtime_session,
        the retry loop in the parent's start_up would treat it as an error.
        We override to loop on switches explicitly.
        """
        openai_api_key = config.OPENAI_API_KEY
        if self.gradio_mode and not openai_api_key:
            await self.wait_for_args()
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args) > 3 and len(args[3]) > 0 else None
            if textbox_api_key is not None:
                openai_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                openai_api_key = config.OPENAI_API_KEY
        else:
            if not openai_api_key or not openai_api_key.strip():
                logger.warning("OPENAI_API_KEY missing. Proceeding with placeholder.")
                openai_api_key = "DUMMY"

        self.client = AsyncOpenAI(api_key=openai_api_key)

        # Basic API key format check (skip network validation — restricted keys
        # often lack models.list scope but work fine for Realtime API)
        if openai_api_key and openai_api_key not in ("DUMMY", ""):
            if openai_api_key.startswith("sk-"):
                logger.info("OpenAI API key loaded (%d chars)", len(openai_api_key))
            else:
                logger.warning("API key doesn't start with 'sk-' — may be invalid")

        # Session loop: keeps running through persona switches and reconnections
        import random
        from websockets.exceptions import ConnectionClosedError

        max_consecutive_errors = 3
        consecutive_errors = 0

        while not self._shutdown_requested:
            try:
                await self._run_realtime_session()
                # Normal return = persona switch or clean exit
                consecutive_errors = 0
                if self._shutdown_requested:
                    return
                # If we returned normally (persona switch), loop back to reconnect
                logger.info("Session ended, reconnecting with persona: %s", self.persona_manager.active_persona)
                continue
            except ConnectionClosedError as e:
                consecutive_errors += 1
                logger.warning(
                    "WebSocket closed unexpectedly (error %d/%d): %s",
                    consecutive_errors, max_consecutive_errors, e,
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Max consecutive errors reached, stopping.")
                    raise
                delay = 2 ** (consecutive_errors - 1) + random.uniform(0, 0.5)
                logger.info("Retrying in %.1f seconds...", delay)
                await asyncio.sleep(delay)
            except Exception as e:
                err_str = str(e).lower()
                if "401" in err_str or "authentication" in err_str or "unauthorized" in err_str:
                    logger.error("Authentication error — check your OpenAI API key: %s", e)
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": "[Error] OpenAI authentication failed. Please check your API key.",
                        })
                    )
                    return
                logger.error("Unexpected error in session: %s", e)
                raise
            finally:
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass
