"""Service lifecycle manager for Evalyn.

Manages starting and stopping services needed for test execution.
Only shuts down services that the agent itself started.
"""

import subprocess
import time
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evalyn.utils.log import get_logger

logger = get_logger("services")


@dataclass
class ServiceDef:
    """Definition of a service to manage."""

    name: str
    command: str
    port: Optional[int] = None
    health_check_url: Optional[str] = None
    startup_timeout_seconds: int = 30
    cwd: Optional[str] = None  # relative to repo root


@dataclass
class ManagedService:
    """A service that was started by the agent."""

    definition: ServiceDef
    process: subprocess.Popen


class ServiceManager:
    """Manages service lifecycle for test execution.

    Only starts services that aren't already running, and only stops
    services that it started itself.

    Args:
        repo_path: Path to the target repository.
        run_dir: Path to the current run directory (for logging).
    """

    def __init__(self, repo_path: Path, run_dir: Path):
        self.repo_path = repo_path
        self.run_dir = run_dir
        self._started: list[ManagedService] = []

    def start_services(self, services: list[dict]) -> None:
        """Start required services that aren't already running.

        Args:
            services: List of service definitions from config.
                Each dict should have: name, command, and optionally
                port, health_check_url, startup_timeout_seconds, cwd.
        """
        for svc_dict in services:
            svc = ServiceDef(
                name=svc_dict.get("name", "unknown"),
                command=svc_dict.get("command", ""),
                port=svc_dict.get("port"),
                health_check_url=svc_dict.get("health_check_url"),
                startup_timeout_seconds=svc_dict.get("startup_timeout_seconds", 30),
                cwd=svc_dict.get("cwd"),
            )

            if not svc.command:
                logger.warning(f"Service '{svc.name}' has no command, skipping")
                continue

            # Check if already running
            if svc.port and _is_port_in_use(svc.port):
                logger.info(
                    f"Service '{svc.name}' already running on port {svc.port}"
                )
                continue

            # Start the service
            self._start_service(svc)

    def stop_services(self) -> None:
        """Stop all services that the agent started."""
        for managed in reversed(self._started):
            try:
                logger.info(f"Stopping service: {managed.definition.name}")
                managed.process.terminate()
                try:
                    managed.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Service '{managed.definition.name}' didn't stop "
                        "gracefully, killing..."
                    )
                    managed.process.kill()
                    managed.process.wait(timeout=5)
                logger.info(f"Service '{managed.definition.name}' stopped")
            except Exception as e:
                logger.error(
                    f"Error stopping '{managed.definition.name}': {e}"
                )

        self._started.clear()

    def _start_service(self, svc: ServiceDef) -> None:
        """Start a single service."""
        cwd = str(self.repo_path / svc.cwd) if svc.cwd else str(self.repo_path)

        log_file = self.run_dir / f"service_{svc.name}.log"

        logger.info(f"Starting service: {svc.name} (command: {svc.command})")

        try:
            log_handle = open(log_file, "w")
            process = subprocess.Popen(
                svc.command,
                shell=True,
                cwd=cwd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=None,
            )

            managed = ManagedService(definition=svc, process=process)
            self._started.append(managed)

            # Wait for the service to be ready
            if svc.port:
                if _wait_for_port(svc.port, timeout=svc.startup_timeout_seconds):
                    logger.info(
                        f"Service '{svc.name}' ready on port {svc.port}"
                    )
                else:
                    logger.warning(
                        f"Service '{svc.name}' started but port {svc.port} "
                        f"not responding after {svc.startup_timeout_seconds}s"
                    )
            else:
                # No port to check — just wait a moment
                time.sleep(2)
                if process.poll() is not None:
                    logger.error(
                        f"Service '{svc.name}' exited immediately "
                        f"(exit code: {process.returncode})"
                    )
                else:
                    logger.info(f"Service '{svc.name}' started (no port check)")

        except Exception as e:
            logger.error(f"Failed to start service '{svc.name}': {e}")

    @property
    def started_count(self) -> int:
        """Number of services started by the agent."""
        return len(self._started)

    def get_base_url(self) -> Optional[str]:
        """Get the base URL of the first service with a port."""
        for managed in self._started:
            if managed.definition.port:
                return f"http://localhost:{managed.definition.port}"
        return None


def _is_port_in_use(port: int) -> bool:
    """Check if a TCP port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(port: int, timeout: int = 30) -> bool:
    """Wait for a port to become available."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if _is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False
