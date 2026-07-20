"""Alcove — Camera / vision tool.

Lets both personas see what the Reachy Mini camera sees, so they can
answer questions about the physical world in front of them.

Two execution paths:

1. **Local vision_manager available** (--local-vision mode) — runs a
   local vision model on-robot and returns a text description. Cheap,
   no network, fast, but requires the model to be loaded.

2. **Fallback: no vision_manager** (default gpt-realtime mode) —
   captures a JPEG frame, base64-encodes it, returns as `b64_im`. Our
   handler in alcove_handler.py already looks for `b64_im` in the tool
   result and injects the image into the OpenAI Realtime conversation
   as an `input_image`, so the model sees it directly and answers the
   question in its next spoken turn.

Description was rewritten from the bare Pollen port to give the LLM
enough context to actually pick this tool when a "seeing" question
comes in (rather than just guessing from memory).
"""

import base64
import asyncio
import logging
from typing import Any, Dict

import cv2

from alcove.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Camera(Tool):
    """Take a picture with the robot camera and answer a question about it."""

    name = "camera"
    description = (
        "Take a picture with the robot's camera and answer a question about what's "
        "visible. Use this whenever the user asks you to look at, describe, identify, "
        "count, or assess something in front of you. Examples: 'what's on the "
        "counter?', 'is the cheese melted yet?', 'is this browned enough?', 'what "
        "ingredient is this?', 'how does the dish look?', 'who's here?', 'what am I "
        "holding?'. Always prefer this tool when the answer requires seeing rather "
        "than knowing — do not guess about the physical world."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The specific question to answer about the current view "
                    "(e.g. 'Is the salmon browned?', 'What ingredient is this?', "
                    "'How many people are in the room?')."
                ),
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Grab a frame and answer the question about it."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "Please provide a question about what to look at."}

        logger.info("Tool call: camera question=%s", question[:120])

        # Frame acquisition
        if deps.camera_worker is None:
            logger.error("camera: camera_worker is None — vision not available")
            return {"error": "Camera not available on this robot."}

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            logger.warning("camera: no frame available from camera worker yet")
            return {"error": "No camera frame available yet — try again in a moment."}

        # Path 1: local vision manager (--local-vision mode)
        if deps.vision_manager is not None:
            try:
                vision_result = await asyncio.to_thread(
                    deps.vision_manager.processor.process_image, frame, question,
                )
            except Exception as e:
                logger.exception("camera: local vision processing failed")
                return {"error": f"Local vision failed: {e}"}

            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            if isinstance(vision_result, str):
                return {"image_description": vision_result}
            return {"error": "Vision manager returned an unexpected value."}

        # Path 2: encode and return base64 for the Realtime handler to inject
        try:
            success, buffer = cv2.imencode(".jpg", frame)
        except Exception as e:
            logger.exception("camera: JPEG encoding failed")
            return {"error": f"Couldn't encode camera frame: {e}"}

        if not success:
            logger.error("camera: cv2.imencode returned False")
            return {"error": "Failed to encode camera frame as JPEG."}

        b64_encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        logger.info(
            "camera: returning %d-byte JPEG for Realtime image injection",
            len(b64_encoded),
        )
        return {"b64_im": b64_encoded}
