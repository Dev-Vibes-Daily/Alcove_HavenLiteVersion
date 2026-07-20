"""Entrypoint for the Alcove dual-persona Reachy Mini app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from alcove.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Alcove dual-persona Reachy Mini app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Alcove app."""
    from alcove.moves import MovementManager
    from alcove.console import LocalStream
    from alcove.tools.core_tools import ToolDependencies
    from alcove.audio.head_wobbler import HeadWobbler
    from alcove.personas.alcove_handler import AlcoveHandler

    logger = setup_logger(args.debug)
    logger.info("Starting Alcove")

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    _status = robot.client.get_status()
    _sim_enabled = (
        _status.get("simulation_enabled", False) if isinstance(_status, dict)
        else getattr(_status, "simulation_enabled", False)
    )
    if _sim_enabled and not args.gradio:
        logger.error(
            "Simulation mode requires Gradio interface. Please use --gradio flag when running in simulation mode."
        )
        robot.client.disconnect()
        sys.exit(1)

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = AlcoveHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)
    logger.info("Alcove handler initialized (dual-persona: Persona A + Persona B)")

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        api_key_textbox = gr.Textbox(
            label="OPENAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
        )

        from alcove.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                api_key_textbox,
                *personality_ui.additional_inputs_ordered(),
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Alcove"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )

    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class AlcoveApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for Alcove."""

    # Alcove runs headless on the robot's mic / speaker (LocalStream). It has
    # no Gradio UI, so we don't advertise a custom_app_url to Pollen Desktop:
    # if we did, Pollen would open an embedded panel pointing at our settings
    # server, expect Gradio-style heartbeats, and after ~3 minutes its
    # watchdog would send a stop signal — killing the app mid-conversation.
    # The OpenAI key ships in runtime.env (via package-data).
    custom_app_url = None
    dont_start_webserver = True

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run Alcove."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()
        args.gradio = False

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = AlcoveApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
