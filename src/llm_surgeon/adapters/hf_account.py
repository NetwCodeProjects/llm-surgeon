"""The Hugging Face login used for publishing, through the control environment's pinned huggingface_hub.

The token is read from the environment variable named by ``model.hf_token_env`` or from the login that
``hf auth login`` stores (huggingface_hub's own token file); it is never printed, logged or written by this
project. ``login`` runs the ``hf`` command installed in the control environment (``.venv``), so nothing is
downloaded to log in. Verified against huggingface_hub 2.3.0 (``get_token``, ``HfApi.whoami``,
``HfApi.model_info``); an OAuth login's ``whoami`` carries ``auth.type == "oauth"`` and ``expiresAt`` but no
token role.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..errors import ExitCode, PipelineError

# Files a new, empty repository holds.
EMPTY_REPO_FILES = {".gitattributes"}


def _hub():
    try:
        import huggingface_hub  # noqa: PLC0415
        import huggingface_hub.errors  # noqa: F401, PLC0415 - a submodule, not loaded by the package import
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PipelineError(
            ExitCode.ENVIRONMENT,
            "E_CONTROL_ENV_INCOMPLETE",
            "huggingface_hub is not installed in the control environment",
            hint="Run ./main.sh once without --dry-run to bootstrap .venv.",
        ) from exc
    return huggingface_hub


def hf_command() -> str | None:
    """The ``hf`` command next to the running Python (the control environment's)."""
    return shutil.which("hf", path=str(Path(sys.executable).parent))


class HubAccount:
    def token(self, env_name: str | None) -> str | None:
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]
        return _hub().get_token()  # HF_TOKEN, then the stored login

    def login(self) -> None:
        """Interactive ``hf auth login`` (the token is typed into hf's own hidden prompt)."""
        command = hf_command()
        if command is None:
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_HF_LOGIN_UNAVAILABLE",
                "The control environment has no 'hf' command to log in with",
                hint="Export a Hugging Face write token in HF_TOKEN instead.",
            )
        subprocess.run([command, "auth", "login"], check=False)

    def whoami(self, token: str) -> dict[str, Any] | None:
        """The account behind ``token``; None when the token is not accepted."""
        hub = _hub()
        try:
            return hub.HfApi().whoami(token=token)
        except hub.errors.HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                return None
            raise PipelineError(ExitCode.ACQUIRE, "E_HF_API", f"Hugging Face account check failed: {exc}") from exc

    def repo_state(self, repo_id: str, token: str) -> str:
        """'missing', 'empty' (private, nothing but .gitattributes), 'content' (private, with files) or 'public'."""
        hub = _hub()
        try:
            info = hub.HfApi().model_info(repo_id, token=token)
        except hub.errors.RepositoryNotFoundError:
            return "missing"
        if not info.private:
            return "public"
        files = {s.rfilename for s in info.siblings or []}
        return "empty" if files <= EMPTY_REPO_FILES else "content"

    def parameter_count(self, repo_id: str, revision: str | None, token: str | None) -> int | None:
        hub = _hub()
        try:
            info = hub.HfApi().model_info(repo_id, revision=revision, expand=["safetensors"], token=token)
        except Exception:  # noqa: BLE001 - only used for size estimates
            return None
        return info.safetensors.total if info.safetensors else None
