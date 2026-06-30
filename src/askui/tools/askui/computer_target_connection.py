from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grpc

from askui.tools.askui.askui_ui_controller_grpc.generated import (
    Controller_V1_pb2 as controller_v1_pbs,
)
from askui.tools.askui.askui_ui_controller_grpc.generated import (
    Controller_V1_pb2_grpc as controller_v1,
)
from askui.tools.askui.exceptions import AskUiControllerError

if TYPE_CHECKING:
    from askui.tools.askui.agent_os_target_computer import ComputerTarget

logger = logging.getLogger(__name__)


@dataclass
class ComputerTargetConnection:
    """
    The live gRPC connection to a `ComputerTarget`: the open channel, the
    controller stub bound to it, and the session opened on the target computer.

    Holds only the live connection handles; the `ComputerTarget` it belongs to
    is passed in when opening or closing. Encapsulates all gRPC specifics so
    that `ComputerTarget` and `ComputerTargetPool` stay free of channel / stub /
    session details.

    Args:
        channel (grpc.Channel): The open gRPC channel.
        stub (ControllerAPIStub): The controller API stub bound to `channel`.
        session_info (SessionInfo): The session opened on the target computer.
    """

    channel: grpc.Channel
    stub: controller_v1.ControllerAPIStub
    session_info: controller_v1_pbs.SessionInfo

    @classmethod
    def open(cls, target: ComputerTarget) -> ComputerTargetConnection:
        """
        Open a gRPC channel and session to `target`.

        Starts the target's local controller process first (a no-op for remote
        and service-managed targets), opens an insecure gRPC channel, starts a
        session, starts execution, and sets the configured display.

        On failure during session setup, the channel is closed and any started
        process is stopped before re-raising.
        """
        target.start()
        channel = grpc.insecure_channel(
            target.address,
            options=[
                ("grpc.max_send_message_length", 2**30),
                ("grpc.max_receive_message_length", 2**30),
                ("grpc.default_deadline", 300000),
            ],
        )
        stub = controller_v1.ControllerAPIStub(channel)
        try:
            session_response: controller_v1_pbs.Response_StartSession = (
                stub.StartSession(
                    controller_v1_pbs.Request_StartSession(
                        sessionGUID=target.session_guid,
                        immediateExecution=True,
                    )
                )
            )
            session_info = session_response.sessionInfo
            stub.StartExecution(
                controller_v1_pbs.Request_StartExecution(sessionInfo=session_info)
            )
            stub.SetActiveDisplay(
                controller_v1_pbs.Request_SetActiveDisplay(displayID=target.display)
            )
        except Exception as e:
            try:
                channel.close()
            finally:
                target.stop()
            error_msg = (
                f"Failed to connect to Agent OS target computer "
                f"{target.description!r} "
                f"(computer_id={target.computer_id!r}, "
                f"session_guid={target.session_guid}, "
                f"display={target.display}, "
                f"address={target.address}): {e}"
            )
            raise AskUiControllerError(error_msg) from e
        return cls(channel=channel, stub=stub, session_info=session_info)

    def close(self, target: ComputerTarget) -> None:
        """
        Close this connection to `target`.

        Stops execution, ends the session, closes the gRPC channel, and stops
        the target's local controller process (a no-op unless this client
        started one). Errors are logged but never raised, so a partial failure
        still releases the rest of the connection.
        """
        computer_id = target.computer_id
        try:
            self.stub.StopExecution(
                controller_v1_pbs.Request_StopExecution(sessionInfo=self.session_info)
            )
            self.stub.EndSession(
                controller_v1_pbs.Request_EndSession(sessionInfo=self.session_info)
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error stopping execution/session for controller %s", computer_id
            )
        try:
            self.channel.close()
        except Exception:  # noqa: BLE001
            logger.exception("Error closing channel for controller %s", computer_id)
        try:
            target.stop()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error stopping client-started controller process for %s", computer_id
            )


__all__ = ["ComputerTargetConnection"]
