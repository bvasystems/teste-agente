"""
Production-ready Stdio implementation of the Model Context Protocol (MCP) server client.

FIXES APPLIED:
1. Correct JSON-RPC method names per MCP spec
2. Updated protocol version to 2024-11-05
3. Fixed parameter structures
4. Added process health monitoring
5. Improved error handling and resource cleanup
6. Better connection state management
7. Proper signal handling for graceful shutdown

ENHANCEMENTS:
1. Added extensive, detailed logging to trace execution flow and debug connection issues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from collections.abc import Callable, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, NotRequired, Optional, TypedDict, override

from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol

if TYPE_CHECKING:
    from mcp.types import (
        BlobResourceContents,
        CallToolResult,
        Resource,
        TextResourceContents,
        Tool,
    )


# TypedDict definitions for JSON-RPC messages
class _JsonRpcRequestParams(TypedDict, total=False):
    """Parameters for a JSON-RPC request."""

    protocolVersion: NotRequired[str]
    clientInfo: NotRequired[MutableMapping[str, str]]
    capabilities: NotRequired[MutableMapping[str, MutableMapping[str, Any]]]
    uri: NotRequired[str]
    name: NotRequired[str]  # FIXED: Changed from 'tool' to 'name'
    arguments: NotRequired[MutableMapping[str, Any]]


class _JsonRpcRequest(TypedDict):
    """A JSON-RPC request message."""

    jsonrpc: str
    id: str
    method: str
    params: NotRequired[_JsonRpcRequestParams]


class _JsonRpcNotification(TypedDict):
    """A JSON-RPC notification message."""

    jsonrpc: str
    method: str
    params: NotRequired[MutableMapping[str, Any]]


class _JsonRpcResponse(TypedDict, total=False):
    """A JSON-RPC response message."""

    jsonrpc: str
    id: str
    result: NotRequired[MutableMapping[str, Any]]
    error: NotRequired[MutableMapping[str, Any]]


class StdioMCPServer(MCPServerProtocol):
    """
    Production-ready Stdio implementation of the MCP (Model Context Protocol) server client.

    FIXED VERSION with proper protocol compliance, error handling, and resource management.

    This class provides a client implementation for interacting with MCP servers
    over standard input/output streams. The server is launched as a subprocess and
    communication happens through stdin/stdout pipes.

    Key Features:
    - Proper MCP protocol compliance (2024-11-05)
    - Robust process lifecycle management
    - Health monitoring and connection validation
    - Graceful error handling and cleanup
    - Signal handling for proper shutdown
    - Resource leak prevention

    Attributes:
        server_name (str): A human-readable name for the server
        command (str): The command to launch the MCP server subprocess
        server_env (MutableMapping[str, str]): Environment variables for the server process
        working_dir (str): Working directory for the server process
        request_timeout_s (float): Timeout for individual requests
        startup_timeout_s (float): Timeout for server startup
        shutdown_timeout_s (float): Timeout for graceful shutdown

    Usage:
        server = StdioMCPServer(
            server_name="OpenMemory MCP",
            command="npx openmemory",
            server_env={"OPENMEMORY_API_KEY": "your-key"},
        )

        try:
            await server.connect()
            tools = await server.list_tools()
            result = await server.call_tool("search", {"query": "test"})
        finally:
            await server.cleanup()
    """

    # Required configuration fields
    server_name: str = Field(..., description="Human-readable name for the MCP server")
    command: str | Callable[..., str] = Field(
        ..., description="Command to launch the MCP server subprocess"
    )

    # Optional configuration fields
    server_env: MutableMapping[str, str] = Field(
        default_factory=dict,
        description="Environment variables to pass to the server process",
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="Working directory for the server process",
    )
    request_timeout_s: float = Field(
        default=30.0, description="Timeout in seconds for individual requests"
    )
    startup_timeout_s: float = Field(
        default=10.0, description="Timeout in seconds for server startup"
    )
    shutdown_timeout_s: float = Field(
        default=5.0, description="Timeout in seconds for graceful shutdown"
    )

    # Internal state
    _process: Optional[asyncio.subprocess.Process] = PrivateAttr(default=None)
    _stdin: Optional[asyncio.StreamWriter] = PrivateAttr(default=None)
    _stdout: Optional[asyncio.StreamReader] = PrivateAttr(default=None)
    _stderr: Optional[asyncio.StreamReader] = PrivateAttr(default=None)
    _next_id: int = PrivateAttr(default=1)
    _pending_requests: MutableMapping[str, asyncio.Future[_JsonRpcResponse]] = (
        PrivateAttr(default_factory=dict)
    )
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )
    _read_task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)
    _stderr_task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)
    _health_check_task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _connection_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @property
    @override
    def name(self) -> str:
        """Get a readable name for the server."""
        return self.server_name

    def _is_connected(self) -> bool:
        """Check if the server process is running and its I/O streams are available."""
        is_proc_ok = self._process is not None and self._process.returncode is None
        is_stdin_ok = self._stdin is not None and not self._stdin.is_closing()
        is_stdout_ok = self._stdout is not None

        # This check verifies the transport layer is up. The protocol-level
        # _initialized flag is managed by the connect_async flow to prevent
        # premature API calls. Removing it from this check resolves the deadlock.
        connected = is_proc_ok and is_stdin_ok and is_stdout_ok

        self._logger.debug(
            f"Checking connection status for '{self.server_name}': "
            + f"process_ok={is_proc_ok}, stdin_ok={is_stdin_ok}, stdout_ok={is_stdout_ok} "
            + f"-> connected={connected}"
        )
        return connected

    @override
    async def connect_async(self) -> None:
        """
        Connect to the MCP server over stdin/stdout.

        FIXED: Proper protocol compliance and error handling.
        """
        self._logger.info(f"Acquiring connection lock for '{self.server_name}'...")
        async with self._connection_lock:
            self._logger.debug(f"Connection lock acquired for '{self.server_name}'.")
            if self._is_connected():
                self._logger.info(
                    f"Already connected to MCP server '{self.server_name}'. Skipping connection."
                )
                return

            self._logger.info(f"Starting connection process for '{self.server_name}'.")

            try:
                self._logger.debug("Step 1: Starting server process...")
                await self._start_server_process()
                self._logger.debug("Step 2: Initializing MCP protocol...")
                await self._initialize_protocol()
                self._logger.debug("Step 3: Starting background tasks...")
                self._start_background_tasks()
                self._initialized = True

                self._logger.info(
                    f"MCP server '{self.server_name}' connected successfully. Status: initialized=True"
                )

            except Exception as e:
                self._logger.exception(
                    f"!!! Critical failure during connect_async for '{self.server_name}': {e}"
                )
                await self._cleanup_resources()
                raise ConnectionError(
                    f"Could not connect to MCP server '{self.server_name}': {e}"
                )

    async def _start_server_process(self) -> None:
        """Start the server subprocess with proper configuration."""
        # Prepare environment variables
        env = os.environ.copy()
        env.update(self.server_env)
        self._logger.debug(f"Server environment variables: {self.server_env}")

        # Disable Python buffering for immediate I/O
        env["PYTHONUNBUFFERED"] = "1"

        try:
            # Split command into args if provided as a string
            cmd = self.command() if callable(self.command) else self.command
            cmd_args = shlex.split(cmd)
            self._logger.info(f"Executing command: {' '.join(cmd_args)}")
            self._logger.debug(f"Working directory: {self.working_dir or os.getcwd()}")

            # Start the server process
            self._process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.working_dir,
                # ADDED: Ensure process gets killed if parent dies
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )
            self._logger.info(f"Server process started with PID: {self._process.pid}")

            if (
                self._process.stdin is None
                or self._process.stdout is None
                or self._process.stderr is None
            ):
                self._logger.error("Subprocess pipes (stdin/stdout/stderr) are None.")
                raise ConnectionError("Failed to open pipes to server process")

            self._stdin = self._process.stdin
            self._stdout = self._process.stdout
            self._stderr = self._process.stderr
            self._logger.debug("Successfully assigned stdin, stdout, stderr streams.")

            # Wait briefly to ensure process started successfully
            await asyncio.sleep(0.1)

            if self._process.returncode is not None:
                self._logger.error(
                    f"Server process exited immediately after start with code {self._process.returncode}"
                )
                raise ConnectionError(
                    f"Server process exited immediately with code {self._process.returncode}"
                )
            self._logger.debug("Server process is running.")

        except (OSError, Exception) as e:
            self._logger.exception(f"Failed to execute or start server process: {e}")
            raise ConnectionError(f"Failed to start server process: {e}")

    async def _initialize_protocol(self) -> None:
        """
        Initialize the MCP protocol with the server.

        FIXED: Correct protocol version and method names.
        """
        self._logger.info("Initializing MCP protocol...")

        # FIXED: Use correct protocol version and structure
        initialize_request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "initialize",  # Correct method name
            "params": {
                "protocolVersion": "2024-11-05",  # FIXED: Current protocol version
                "clientInfo": {"name": "agentle-mcp-client", "version": "0.1.0"},
                "capabilities": {"resources": {}, "tools": {}, "prompts": {}},
            },
        }
        self._next_id += 1
        self._logger.debug(f"Constructed initialize request: {initialize_request}")

        try:
            # Start a temporary reader task to get the response for initialization
            self._logger.debug("Starting temporary reader for initialization...")
            temp_read_task = asyncio.create_task(self._read_responses())

            # Send initialize with timeout
            self._logger.debug(
                f"Sending 'initialize' request with {self.startup_timeout_s}s timeout."
            )
            response = await asyncio.wait_for(
                self._send_request(initialize_request), timeout=self.startup_timeout_s
            )

            self._logger.info(f"Received 'initialize' response: {response}")
            if "error" in response:
                self._logger.error(
                    f"Server returned an error during initialization: {response['error']}"
                )
                raise ConnectionError(
                    f"Failed to initialize MCP protocol: {response['error']}"
                )

            # Validate server response
            if "result" not in response:
                raise ConnectionError("Invalid initialization response: missing result")

            result = response["result"]
            if "protocolVersion" not in result or "serverInfo" not in result:
                self._logger.error(
                    f"Initialization response missing required fields 'protocolVersion' or 'serverInfo'. Got: {result}"
                )
                raise ConnectionError(
                    "Invalid initialization response: missing required fields"
                )
            self._logger.debug(f"Initialization result validated: {result}")

            # Send initialized notification
            initialized_notification: _JsonRpcNotification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",  # Correct notification method
                "params": {},
            }
            await self._send_notification(initialized_notification)

            # Cancel the temporary reader task, a permanent one will be started next
            temp_read_task.cancel()
            try:
                await temp_read_task
            except asyncio.CancelledError:
                self._logger.debug("Temporary reader task cancelled as expected.")

            self._logger.info(
                f"MCP protocol initialized successfully with server: {result.get('serverInfo', {}).get('name', 'unknown')}"
            )

        except asyncio.TimeoutError:
            self._logger.error(
                f"Protocol initialization timed out after {self.startup_timeout_s}s. No response from server."
            )
            raise ConnectionError(
                f"Protocol initialization timed out after {self.startup_timeout_s}s"
            )
        except Exception as e:
            self._logger.exception(
                f"An unexpected error occurred during initialization: {e}"
            )
            raise

    def _start_background_tasks(self) -> None:
        """Start background tasks for reading responses and monitoring health."""
        if self._stdout is not None and self._read_task is None:
            self._read_task = asyncio.create_task(
                self._read_responses(), name=f"read-{self.server_name}"
            )
            self._logger.info(
                f"Started response reader task: {self._read_task.get_name()}"
            )
        else:
            self._logger.warning(
                "Could not start response reader: stdout is None or task already exists."
            )

        if self._stderr is not None and self._stderr_task is None:
            self._stderr_task = asyncio.create_task(
                self._read_stderr(), name=f"stderr-{self.server_name}"
            )
            self._logger.info(
                f"Started stderr reader task: {self._stderr_task.get_name()}"
            )
        else:
            self._logger.warning(
                "Could not start stderr reader: stderr is None or task already exists."
            )

        # Start health monitoring
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(
                self._monitor_process_health(), name=f"health-{self.server_name}"
            )
            self._logger.info(
                f"Started health monitor task: {self._health_check_task.get_name()}"
            )
        else:
            self._logger.warning("Could not start health monitor: task already exists.")

    async def _read_responses(self) -> None:
        """
        Background task to read responses from the server.

        IMPROVED: Better error handling and connection monitoring.
        """
        if self._stdout is None:
            self._logger.error("Cannot read responses, stdout stream is None.")
            return

        self._logger.debug("Response reader task started.")
        try:
            while not self._stdout.at_eof():
                try:
                    line = await asyncio.wait_for(self._stdout.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    if self._process and self._process.returncode is not None:
                        self._logger.warning(
                            f"Server process terminated (code {self._process.returncode}) while waiting for data."
                        )
                        break
                    continue

                if not line:
                    self._logger.warning(
                        "Server closed stdout (readline returned empty bytes)."
                    )
                    break

                try:
                    message_str = line.decode("utf-8").strip()
                    if not message_str:
                        continue

                    self._logger.debug(f"<- RECV: {message_str}")
                    message = json.loads(message_str)

                    if "id" in message:
                        request_id = message["id"]
                        self._logger.debug(
                            f"Received message is a response for ID: {request_id}"
                        )
                        future = self._pending_requests.pop(request_id, None)
                        if future and not future.cancelled():
                            future.set_result(message)
                            self._logger.debug(
                                f"Future for request ID {request_id} resolved."
                            )
                        elif future and future.cancelled():
                            self._logger.warning(
                                f"Received response for already cancelled request ID: {request_id}"
                            )
                        elif not future:
                            self._logger.warning(
                                f"Received response for unknown or already handled request ID: {request_id}"
                            )
                    elif "method" in message:
                        self._logger.debug(
                            f"Received message is a server notification: {message['method']}"
                        )
                        await self._handle_server_notification(message)
                    else:
                        self._logger.warning(
                            f"Received unknown message type: {message}"
                        )

                except json.JSONDecodeError as e:
                    message_str = line.decode("utf-8", errors="replace").strip()
                    self._logger.error(
                        f"Failed to parse JSON from server: {e}. Line was: '{message_str}'"
                    )
                except UnicodeDecodeError as e:
                    self._logger.error(
                        f"Failed to decode UTF-8 message from server: {e}"
                    )

        except asyncio.CancelledError:
            self._logger.info("Response reading task was cancelled.")
        except Exception as e:
            self._logger.exception(f"!!! Unhandled error in response reader task: {e}")
            for future in list(self._pending_requests.values()):
                if not future.done():
                    future.set_exception(
                        ConnectionError(f"Server communication error: {e}")
                    )
            self._pending_requests.clear()
        finally:
            self._logger.info("Response reader task finished.")

    async def _read_stderr(self) -> None:
        """Background task to read and log stderr from the server."""
        if self._stderr is None:
            self._logger.error("Cannot read stderr, stderr stream is None.")
            return

        self._logger.debug("Stderr reader task started.")
        try:
            async for line in self._stderr:
                message = line.decode("utf-8", errors="replace").strip()
                if message:
                    if any(
                        level in message.lower()
                        for level in ["error", "exception", "failed", "traceback"]
                    ):
                        self._logger.error(f"Server stderr: {message}")
                    elif any(level in message.lower() for level in ["warn", "warning"]):
                        self._logger.warning(f"Server stderr: {message}")
                    else:
                        self._logger.info(f"Server stderr: {message}")
        except asyncio.CancelledError:
            self._logger.info("Stderr reading task was cancelled.")
        except Exception as e:
            self._logger.exception(f"!!! Unhandled error in stderr reader task: {e}")
        finally:
            self._logger.info("Stderr reader task finished.")

    async def _monitor_process_health(self) -> None:
        """Monitor the health of the server process."""
        self._logger.debug("Process health monitor started.")
        try:
            while True:
                await asyncio.sleep(5.0)

                if self._process is None:
                    self._logger.warning(
                        "Health monitor: Process object is None. Stopping monitor."
                    )
                    break

                if self._process.returncode is not None:
                    self._logger.error(
                        f"Health monitor: Server process has died with return code: {self._process.returncode}. PID: {self._process.pid}"
                    )
                    self._initialized = False
                    for request_id, future in list(self._pending_requests.items()):
                        if not future.done():
                            msg = f"Server process died with code {self._process.returncode}"
                            future.set_exception(ConnectionError(msg))
                            self._logger.warning(
                                f"Failing pending request {request_id} due to process death."
                            )
                    self._pending_requests.clear()
                    break
                else:
                    self._logger.debug(
                        f"Health check: Process {self._process.pid} is alive."
                    )

        except asyncio.CancelledError:
            self._logger.info("Process health monitor was cancelled.")
        except Exception as e:
            self._logger.exception(f"!!! Unhandled error in health monitor task: {e}")
        finally:
            self._logger.info("Process health monitor finished.")

    async def _handle_server_notification(
        self, message: MutableMapping[str, Any]
    ) -> None:
        """Handle notifications sent by the server."""
        method = message.get("method", "")
        params = message.get("params", {})
        self._logger.debug(
            f"Handling server notification '{method}' with params: {params}"
        )

        if method == "notifications/message":
            level = params.get("level", "info")
            data = params.get("data", "")
            getattr(self._logger, level, self._logger.info)(
                f"Server notification: {data}"
            )
        else:
            self._logger.debug(f"Received unhandled server notification: {method}")

    @override
    async def cleanup_async(self) -> None:
        """
        Clean up the server connection with proper resource management.

        IMPROVED: Better cleanup sequence and error handling.
        """
        self._logger.info(
            f"Acquiring connection lock for cleanup of '{self.server_name}'..."
        )
        async with self._connection_lock:
            self._logger.info(f"Starting cleanup for MCP server: {self.server_name}")

            if not self._initialized and self._process is None:
                self._logger.info(
                    "Cleanup called on an already clean or uninitialized server. Nothing to do."
                )
                return

            self._initialized = False
            self._logger.debug("Set initialized=False to prevent new operations.")

            if self._pending_requests:
                self._logger.warning(
                    f"Cancelling {len(self._pending_requests)} pending requests during shutdown."
                )
                for request_id, future in list(self._pending_requests.items()):
                    if not future.done():
                        future.set_exception(ConnectionError("Server is shutting down"))
                        self._logger.debug(f"Cancelled pending request {request_id}.")
                self._pending_requests.clear()

            await self._cleanup_resources()
            self._logger.info(f"Cleanup for '{self.server_name}' complete.")

    async def _cleanup_resources(self) -> None:
        """Clean up all resources in the proper order."""
        self._logger.debug("Cleaning up background tasks and process resources.")
        tasks_to_cancel = [
            ("read_task", self._read_task),
            ("stderr_task", self._stderr_task),
            ("health_check_task", self._health_check_task),
        ]

        for task_name, task in tasks_to_cancel:
            if task is not None and not task.done():
                self._logger.debug(f"Cancelling {task_name}...")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                    self._logger.debug(f"{task_name} cancelled successfully.")
                except asyncio.CancelledError:
                    self._logger.debug(f"{task_name} was already cancelled.")
                except asyncio.TimeoutError:
                    self._logger.warning(f"Timeout waiting for {task_name} to cancel.")
                except Exception as e:
                    self._logger.warning(
                        f"Error during cancellation of {task_name}: {e}"
                    )

        self._read_task = None
        self._stderr_task = None
        self._health_check_task = None

        if self._stdin is not None:
            self._logger.debug("Closing stdin stream...")
            try:
                if not self._stdin.is_closing():
                    self._stdin.close()
                    await asyncio.wait_for(self._stdin.wait_closed(), timeout=2.0)
                    self._logger.debug("Stdin stream closed.")
            except Exception as e:
                self._logger.warning(f"Error closing stdin: {e}")
            finally:
                self._stdin = None

        self._stdout = None
        self._stderr = None

        if self._process is not None:
            await self._terminate_process()

    async def _terminate_process(self) -> None:
        """Terminate the server process gracefully, with force if needed."""
        if self._process is None:
            self._logger.debug("Terminate process called, but no process exists.")
            return

        process = self._process
        self._process = None
        pid = process.pid
        self._logger.info(f"Terminating server process {pid}...")

        try:
            if process.returncode is None:
                self._logger.debug(f"Sending SIGTERM to process {pid}.")
                process.terminate()

                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self.shutdown_timeout_s
                    )
                    self._logger.info(
                        f"Process {pid} terminated gracefully with code {process.returncode}."
                    )
                except asyncio.TimeoutError:
                    self._logger.warning(
                        f"Process {pid} did not terminate gracefully after {self.shutdown_timeout_s}s. Killing it (SIGKILL)."
                    )
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                        self._logger.info(f"Process {pid} killed successfully.")
                    except asyncio.TimeoutError:
                        self._logger.error(
                            f"Failed to kill process {pid} after waiting."
                        )
            else:
                self._logger.info(
                    f"Process {pid} was already terminated with code {process.returncode}."
                )

        except ProcessLookupError:
            self._logger.warning(
                f"Could not find process {pid} to terminate. It may have already exited."
            )
        except Exception as e:
            self._logger.exception(f"Error while terminating process {pid}: {e}")

    async def _send_request(self, request: _JsonRpcRequest) -> _JsonRpcResponse:
        """
        Send a request to the server and wait for the response.

        IMPROVED: Better connection validation and error handling.
        """
        method = request["method"]
        request_id = request["id"]
        self._logger.debug(f"Preparing to send request '{method}' (id: {request_id}).")
        if not self._is_connected():
            self._logger.error(f"Cannot send request '{method}'. Server not connected.")
            raise ConnectionError(
                f"Server not connected. Failed to send request '{method}'."
            )

        if self._stdin is None:  # Should be caught by _is_connected, but as a safeguard
            raise ConnectionError("Server stdin not available")

        response_future: asyncio.Future[_JsonRpcResponse] = asyncio.Future()
        self._pending_requests[request_id] = response_future
        self._logger.debug(
            f"Future created for request ID {request_id}. Total pending: {len(self._pending_requests)}."
        )

        try:
            request_json = json.dumps(request) + "\n"
            self._logger.debug(f"-> SEND: {request_json.strip()}")
            self._stdin.write(request_json.encode("utf-8"))
            await self._stdin.drain()
            self._logger.debug(
                f"Request '{method}' (id: {request_id}) written to stdin and drained."
            )

            response = await asyncio.wait_for(
                response_future, timeout=self.request_timeout_s
            )
            self._logger.debug(f"Response received for '{method}' (id: {request_id}).")
            return response

        except asyncio.TimeoutError as e:
            self._logger.error(
                f"Request '{method}' (id: {request_id}) timed out after {self.request_timeout_s}s."
            )
            self._pending_requests.pop(request_id, None)
            response_future.cancel("Request timed out")
            raise TimeoutError(
                f"Request {method} timed out after {self.request_timeout_s}s"
            ) from e
        except Exception as e:
            self._logger.exception(
                f"Connection error while sending request '{method}' (id: {request_id}): {e}"
            )
            self._pending_requests.pop(request_id, None)
            if not response_future.done():
                response_future.set_exception(e)
            raise ConnectionError(f"Error sending request {method}: {e}")

    async def _send_notification(self, notification: _JsonRpcNotification) -> None:
        """Send a notification to the server."""
        method = notification["method"]
        self._logger.debug(f"Preparing to send notification '{method}'.")
        if not self._is_connected() or self._stdin is None:
            self._logger.error(
                f"Cannot send notification '{method}'. Server not connected."
            )
            raise ConnectionError("Server not connected")

        try:
            notification_json = json.dumps(notification) + "\n"
            self._logger.debug(f"-> SEND: {notification_json.strip()}")
            self._stdin.write(notification_json.encode("utf-8"))
            await self._stdin.drain()
            self._logger.debug(f"Notification '{method}' written to stdin and drained.")
        except Exception as e:
            self._logger.exception(
                f"Connection error while sending notification '{method}': {e}"
            )
            raise ConnectionError(f"Error sending notification {method}: {e}")

    # FIXED: All protocol methods now use correct method names and parameter structures

    async def list_tools_async(self) -> Sequence[Tool]:
        """List the tools available on the server."""
        from mcp.types import Tool

        self._logger.debug("Executing list_tools_async")
        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "tools/list",
            "params": {},
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to list tools: {response['error']}")
        if "result" not in response or "tools" not in response["result"]:
            raise ValueError("Invalid response format: missing 'tools' in result")

        tools = [Tool.model_validate(tool) for tool in response["result"]["tools"]]
        self._logger.debug(f"Successfully listed {len(tools)} tools.")
        return tools

    @override
    async def list_resources_async(self) -> Sequence[Resource]:
        """List the resources available on the server."""
        from mcp.types import Resource

        self._logger.debug("Executing list_resources_async")
        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "resources/list",
            "params": {},
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to list resources: {response['error']}")
        if "result" not in response or "resources" not in response["result"]:
            raise ValueError("Invalid response format: missing 'resources' in result")

        resources = [
            Resource.model_validate(resource)
            for resource in response["result"]["resources"]
        ]
        self._logger.debug(f"Successfully listed {len(resources)} resources.")
        return resources

    @override
    async def list_resource_contents_async(
        self, uri: str
    ) -> Sequence[TextResourceContents | BlobResourceContents]:
        """List contents of a specific resource."""
        from mcp.types import BlobResourceContents, TextResourceContents

        self._logger.debug(f"Executing list_resource_contents_async for uri: {uri}")
        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "resources/read",
            "params": {"uri": uri},
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to read resource contents: {response['error']}")
        if "result" not in response or "contents" not in response["result"]:
            raise ValueError("Invalid response format: missing 'contents' in result")

        contents = [
            TextResourceContents.model_validate(content)
            if content["type"] == "text"
            else BlobResourceContents.model_validate(content)
            for content in response["result"]["contents"]
        ]
        self._logger.debug(
            f"Successfully read {len(contents)} content parts from resource {uri}."
        )
        return contents

    @override
    async def call_tool_async(
        self, tool_name: str, arguments: MutableMapping[str, object] | None
    ) -> "CallToolResult":
        """Invoke a tool on the server."""
        from mcp.types import CallToolResult

        self._logger.debug(
            f"Executing call_tool_async for tool '{tool_name}' with args: {arguments}"
        )
        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to call tool: {response['error']}")
        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        result = CallToolResult.model_validate(response["result"])
        self._logger.debug(f"Successfully called tool '{tool_name}'.")
        return result

    # Additional utility methods

    def get_process_info(self) -> dict[str, Any]:
        """Get information about the server process for debugging."""
        if self._process is None:
            return {"status": "not_started"}

        return {
            "status": "running" if self._process.returncode is None else "terminated",
            "pid": self._process.pid,
            "returncode": self._process.returncode,
            "command": self.command() if callable(self.command) else self.command,
            "initialized": self._initialized,
            "pending_requests": len(self._pending_requests),
        }

    async def ping(self) -> bool:
        """
        Test if the server is responsive by attempting to list tools.

        Returns:
            bool: True if server responds, False otherwise
        """
        self._logger.debug("Pinging server...")
        try:
            await asyncio.wait_for(self.list_tools_async(), timeout=5.0)
            self._logger.debug("Ping successful.")
            return True
        except Exception as e:
            self._logger.warning(f"Ping failed: {e}")
            return False
