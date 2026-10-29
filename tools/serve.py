#!/usr/bin/env python3
"""serve.py — start a tts-serve engine server from its envs/<engine>/ venv.

Each directory under envs/ is a small uv project that builds a dedicated venv
for one server (see docs/04-engine-environments.md).  This launcher picks the
env by name, runs ``uv sync`` only when the venv needs it, then replaces
itself with ``<venv python> <server script>``, so the server runs exactly as
if it had been started by hand, minus the bookkeeping.

Usage:
    python3 tools/serve.py omnivoice              # sync if needed, then start
    python3 tools/serve.py omnivoice --sync       # always run uv sync first
    python3 tools/serve.py --list                 # envs, servers, venv status
    python3 tools/serve.py chatterbox --port 7501 # run next to another engine
    python3 tools/serve.py omnivoice --host 127.0.0.1   # local-only
    OMNIVOICE_PORT=8500 python3 tools/serve.py omnivoice

The server script comes from ``[tool.tts-serve] server`` in the env's
pyproject.toml, and ``--host`` / ``--port`` become ``<env-prefix>_HOST`` /
``<env-prefix>_PORT`` for the server (e.g. OMNIVOICE_HOST, OMNIVOICE_PORT).
Otherwise the working directory and environment are passed through unchanged,
so the server's <ENGINE>_* variables work as usual.

Standard library only, on purpose: this must run on the system python3 with
nothing installed (Python 3.11+, for tomllib).

Full behavioral spec: docs/04-engine-environments.md ('Launcher').
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS_DIRNAME = "envs"

# Touched inside the venv after every successful sync.  Its mtime is the
# baseline the env's inputs are compared against; living inside .venv/ keeps
# it gitignored and makes it vanish whenever uv recreates the venv.
SYNC_STAMP_NAME = ".tts-serve-synced"

UV_INSTALL_URL = "https://docs.astral.sh/uv/getting-started/installation/"

# sync_reason() value for a venv that doesn't exist yet; --list shows it as
# "not built" rather than "needs sync".
NOT_BUILT = "venv not built yet"

# [tool.tts-serve] env-prefix: the server's <PREFIX>_HOST / <PREFIX>_PORT
# prefix.  The prefixes don't follow env names (DOTS_TTS, QWEN3TTS_MLX, ...),
# so each env declares its own.
_ENV_PREFIX_RE = re.compile(r"[A-Z][A-Z0-9_]*")


class ServeError(Exception):
    """A user-facing failure that must exit with status 1."""


@dataclass
class EngineEnv:
    """One envs/<engine>/ project, resolved against the repo root."""

    name: str
    env_dir: Path
    server: Path
    # Files whose modification means the venv may no longer match its
    # definition: the env's pyproject.toml and .python-version, plus the
    # pyproject.toml of every local path source (engine checkout,
    # tts-engine-common).
    inputs: list[Path]
    # Prefix of the server's <PREFIX>_HOST / <PREFIX>_PORT variables; None if
    # the env doesn't declare one (then --host / --port can't be applied).
    env_prefix: str | None = None


def _display(path: Path, repo_root: Path) -> str:
    """A path as the user would type it from the repo root."""
    return os.path.relpath(path, repo_root)


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        raise ServeError(
            f"serve.py needs Python 3.11+ (for tomllib); this is Python {sys.version.split()[0]}."
        ) from None
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ServeError(f"Cannot read {path}: {exc}") from None


def discover_envs(repo_root: Path) -> list[str]:
    """Names of the directories under envs/ that hold a pyproject.toml, sorted."""
    envs_dir = repo_root / ENVS_DIRNAME
    if not envs_dir.is_dir():
        return []
    return sorted(p.name for p in envs_dir.iterdir() if (p / "pyproject.toml").is_file())


def _local_source_inputs(env_dir: Path, pyproject: dict) -> list[Path]:
    """pyproject.toml (or the file itself) of every existing local path source.

    ``[tool.uv.sources]`` values are either one source table or a list of
    marker-guarded tables; only tables with a ``path`` are local.  A missing
    path is skipped here: uv sync reports it far better than we could.
    """
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    found: list[Path] = []
    if not isinstance(sources, dict):
        return found
    for entry in sources.values():
        for item in entry if isinstance(entry, list) else [entry]:
            if not (isinstance(item, dict) and isinstance(item.get("path"), str)):
                continue
            target = (env_dir / item["path"]).resolve()
            candidate = target / "pyproject.toml" if target.is_dir() else target
            if candidate.is_file():
                found.append(candidate)
    return found


def load_env(repo_root: Path, name: str) -> EngineEnv:
    env_dir = repo_root / ENVS_DIRNAME / name
    pyproject_path = env_dir / "pyproject.toml"
    pyproject = _load_toml(pyproject_path)

    server = pyproject.get("tool", {}).get("tts-serve", {}).get("server")
    if not isinstance(server, str) or not server.strip():
        raise ServeError(
            f"{_display(pyproject_path, repo_root)} has no [tool.tts-serve] server entry "
            '(e.g. server = "impl/server_<name>.py").'
        )
    server_path = (repo_root / server).resolve()
    if not server_path.is_file():
        raise ServeError(
            f"Server script {server} (from {_display(pyproject_path, repo_root)}) does not exist."
        )

    env_prefix = pyproject.get("tool", {}).get("tts-serve", {}).get("env-prefix")
    if env_prefix is not None and not (
        isinstance(env_prefix, str) and _ENV_PREFIX_RE.fullmatch(env_prefix)
    ):
        raise ServeError(
            f"{_display(pyproject_path, repo_root)} has a malformed [tool.tts-serve] "
            f'env-prefix {env_prefix!r} (expected e.g. "OMNIVOICE").'
        )

    inputs = [pyproject_path]
    python_version = env_dir / ".python-version"
    if python_version.is_file():
        inputs.append(python_version)
    inputs.extend(_local_source_inputs(env_dir, pyproject))
    return EngineEnv(
        name=name, env_dir=env_dir, server=server_path, inputs=inputs, env_prefix=env_prefix
    )


def venv_python(env_dir: Path, windows: bool | None = None) -> Path:
    if windows is None:
        windows = os.name == "nt"
    if windows:
        return env_dir / ".venv" / "Scripts" / "python.exe"
    return env_dir / ".venv" / "bin" / "python"


def _stamp(env: EngineEnv) -> Path:
    return env.env_dir / ".venv" / SYNC_STAMP_NAME


def sync_reason(env: EngineEnv, repo_root: Path, force: bool = False) -> str | None:
    """Why the venv needs a ``uv sync`` before starting, or None if it doesn't."""
    if force:
        return "--sync given"
    if not venv_python(env.env_dir).is_file():
        return NOT_BUILT
    stamp = _stamp(env)
    if not stamp.is_file():
        return "not yet synced by serve.py"
    baseline = stamp.stat().st_mtime
    for path in env.inputs:
        if path.is_file() and path.stat().st_mtime > baseline:
            return f"{_display(path, repo_root)} changed"
    return None


def run_sync(env: EngineEnv) -> int:
    """Run ``uv sync`` for the env, output streaming to the terminal; return its exit code."""
    uv = shutil.which("uv")
    if uv is None:
        raise ServeError(f"uv is not installed (or not on PATH); see {UV_INSTALL_URL}")
    # An unrelated activated venv only makes uv warn that VIRTUAL_ENV "does not
    # match the project environment path"; the project venv is what we want.
    child_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run([uv, "sync", "--project", str(env.env_dir)], env=child_env).returncode


def env_status(repo_root: Path, name: str) -> tuple[str, str]:
    """(server display path, status) for one env, for --list."""
    try:
        env = load_env(repo_root, name)
    except ServeError as exc:
        return "-", f"error: {exc}"
    reason = sync_reason(env, repo_root)
    if reason is None:
        status = "ready"
    elif reason == NOT_BUILT:
        status = "not built"
    else:
        status = f"needs sync ({reason})"
    return _display(env.server, repo_root), status


def format_env_list(repo_root: Path, names: list[str]) -> str:
    if not names:
        return f"No environments found in {ENVS_DIRNAME}/."
    rows = [(name, *env_status(repo_root, name)) for name in names]
    name_width = max(len(r[0]) for r in rows)
    server_width = max(len(r[1]) for r in rows)
    lines = [f"Environments in {ENVS_DIRNAME}/:"]
    for name, server, status in rows:
        lines.append(f"  {name:<{name_width}}  {server:<{server_width}}  {status}")
    return "\n".join(lines)


def port_number(value: str) -> int:
    """argparse type for --port: an integer 1-65535 (anything else exits 2)."""
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be between 1 and 65535, got {port}")
    return port


def host_address(value: str) -> str:
    """argparse type for --host: non-empty, no whitespace (anything else exits 2).

    Whether the address is one this machine can bind is left to the server,
    which reports it with the real reason.
    """
    if not value or any(ch.isspace() for ch in value):
        raise argparse.ArgumentTypeError(f"invalid host: {value!r}")
    return value


def build_parser(available: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve.py",
        description=(
            "Start a tts-serve engine server from its envs/<engine>/ uv environment, "
            "running 'uv sync' first only when the venv needs it."
        ),
        epilog=(
            f"available environments: {', '.join(available) or 'none'}\n\n"
            "examples:\n"
            "  serve.py omnivoice\n"
            "  serve.py omnivoice --sync\n"
            "  serve.py chatterbox --port 7501\n"
            "  serve.py omnivoice --host 127.0.0.1\n"
            "  OMNIVOICE_PORT=8500 serve.py omnivoice\n"
            "  serve.py --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "engine",
        nargs="?",
        metavar="ENGINE",
        help="Name of a directory under envs/ (see --list).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run 'uv sync' before starting, even if the venv looks up to date.",
    )
    parser.add_argument(
        "--host",
        type=host_address,
        metavar="HOST",
        help=(
            "Address for the server to bind (every server defaults to 0.0.0.0; use "
            "127.0.0.1 for local-only). Passed as <env-prefix>_HOST, e.g. "
            "OMNIVOICE_HOST, overriding any value already set."
        ),
    )
    parser.add_argument(
        "--port",
        type=port_number,
        metavar="PORT",
        help=(
            "Port for the server to bind (every server defaults to 7500). Passed as "
            "<env-prefix>_PORT, e.g. OMNIVOICE_PORT, overriding any value already set."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the environments, their server scripts, and venv status, then exit.",
    )
    return parser


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    """Run the launcher; return the exit code (0 ok, 1 error; argparse exits 2).

    On success the process is replaced by the server, so this only returns
    when something failed (or when os.execve is stubbed out in tests).
    """
    # Resolved once so relpath-based display stays sane under a symlinked
    # checkout.  Never resolve the venv python itself: it is a symlink to the
    # base interpreter, and exec'ing the target would bypass the venv.
    repo_root = repo_root.resolve()
    available = discover_envs(repo_root)
    parser = build_parser(available)
    args = parser.parse_args(argv)

    if args.list:
        print(format_env_list(repo_root, available))
        return 0
    if args.engine is None:
        parser.error("the following arguments are required: ENGINE (or use --list)")
    if args.engine not in available:
        parser.error(
            f"unknown environment {args.engine!r} (available: {', '.join(available) or 'none'})"
        )

    try:
        env = load_env(repo_root, args.engine)
        # Checked before any sync: a sync that then can't start is wasted time.
        requested = [
            (flag, suffix, value)
            for flag, suffix, value in (
                ("--host", "HOST", args.host),
                ("--port", "PORT", args.port),
            )
            if value is not None
        ]
        if requested and env.env_prefix is None:
            flags = " and ".join(flag for flag, _, _ in requested)
            raise ServeError(
                f"{_display(env.env_dir / 'pyproject.toml', repo_root)} has no "
                f"[tool.tts-serve] env-prefix entry, so {flags} can't be applied "
                "(set it to the prefix of the server's <PREFIX>_HOST / <PREFIX>_PORT "
                'variables, e.g. "OMNIVOICE").'
            )
        overrides = {f"{env.env_prefix}_{suffix}": str(value) for _, suffix, value in requested}
        server_env = {**os.environ, **overrides}
        override_note = (
            " (" + ", ".join(f"{k}={v}" for k, v in overrides.items()) + ")" if overrides else ""
        )
        python = venv_python(env.env_dir)
        reason = sync_reason(env, repo_root, force=args.sync)
        if reason is not None:
            print(f"Syncing {ENVS_DIRNAME}/{env.name} ({reason}) ...", flush=True)
            code = run_sync(env)
            if code == 0:
                _stamp(env).touch()
            elif args.sync or not python.is_file():
                return _fail(f"uv sync failed (exit code {code}).")
            else:
                print(
                    f"Warning: uv sync failed (exit code {code}); "
                    "starting with the existing venv."
                )
        if not python.is_file():
            return _fail(f"{_display(python, repo_root)} does not exist after uv sync.")
    except ServeError as exc:
        return _fail(str(exc))

    print(
        f"Starting {_display(env.server, repo_root)} with {_display(python, repo_root)}"
        f"{override_note}",
        flush=True,
    )
    try:
        # Replace this process: signals (Ctrl+C) reach uvicorn directly, and
        # nothing of the launcher lingers.  Buffers were flushed above; exec
        # would discard them.
        os.execve(python, [str(python), str(env.server)], server_env)
    except OSError as exc:
        return _fail(f"Cannot start {_display(python, repo_root)}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
