"""Tests for services/manager.py."""

from evalyn.services.manager import ServiceManager


class TestServiceManagerConfig:
    def test_accepts_template_service_aliases(self, tmp_path, monkeypatch):
        manager = ServiceManager(tmp_path, tmp_path)
        started = []
        monkeypatch.setattr(manager, "_start_service", started.append)

        manager.start_services([
            {
                "name": "backend",
                "start": "npm run dev",
                "health_check": "http://localhost:3000/health",
            }
        ])

        assert len(started) == 1
        assert started[0].name == "backend"
        assert started[0].command == "npm run dev"
        assert started[0].health_check_url == "http://localhost:3000/health"

    def test_command_key_takes_precedence_over_start_alias(self, tmp_path, monkeypatch):
        manager = ServiceManager(tmp_path, tmp_path)
        started = []
        monkeypatch.setattr(manager, "_start_service", started.append)

        manager.start_services([
            {
                "name": "backend",
                "command": "python -m http.server 3000",
                "start": "npm run dev",
            }
        ])

        assert started[0].command == "python -m http.server 3000"
