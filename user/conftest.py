import os
import sys
import types


# Ensure repository root is importable so modules like 'environment' resolve
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Provide a fallback for optional server.api.create_participant during tests
try:
    from server.api import create_participant as _create_participant  # noqa: F401
except Exception:
    server_mod = types.ModuleType("server")
    api_mod = types.ModuleType("server.api")

    def _fallback_create_participant(username):
        return None

    api_mod.create_participant = _fallback_create_participant  # type: ignore[attr-defined]
    server_mod.api = api_mod  # type: ignore[attr-defined]
    sys.modules["server"] = server_mod
    sys.modules["server.api"] = api_mod
