#!/usr/bin/env python3
"""
headless_streamlit.py — A minimal Streamlit stub for running SPACE's core
computation modules WITHOUT the Streamlit runtime / web server.

Why this exists
---------------
The SPACE `space/*.py` modules import `streamlit as st` at module import time
and sprinkle `st.session_state`, `st.spinner`, `st.error`, `st.plotly_chart`,
etc. throughout the computational functions. That couples pure computation to a
web UI runtime. This stub stands in for the real `streamlit` module so you can
import ``space.analysis``, ``space.alignment``, ``space.pdb_processing``, etc.
and drive them from a Python script, a Jupyter notebook, or a REST service —
no browser, no server, no sockets.

Usage
-----
Always install the stub *before* importing any ``space.*`` module::

    import sys, types
    import headless_streamlit as _st
    sys.modules["streamlit"] = _st.get_stub()          # or:
    # sys.modules["streamlit"] = headless_streamlit.stub

    from space.analysis import run_al2co, list_unique_point_mutations
    ...

The stub is a drop-in replacement for the *subset* of the Streamlit API that
SPACE's modules actually call. Anything a script doesn't use degrades to a
no-op; anything the modules rely on for state (``session_state``) is a real
dict-backed object so the pipeline flows correctly without a UI.

Notes on behaviour
------------------
* ``session_state`` is a real ``_SessionState`` dict-like: attribute access,
  ``.get()``, ``.update()``, ``in``, and ``.add()`` (for the ``outputs`` set)
  all work. Nothing is persisted — it is per-process.
* UI widgets (``spinner``, ``progress``, ``error``, ``warning``, ``info``,
  ``success``, ``write``, ``markdown``, ``code``, ``plotly_chart``, ``pyplot``)
  are no-ops (optionally print to stderr when ``VERBOSE`` is set) unless the
  callable is a context manager (``spinner``), in which case it yields.
* ``selectbox`` / ``select_slider`` / ``radio`` / ``checkbox`` / ``slider`` /
  ``text_input`` / ``text_area`` / ``number_input`` / ``button`` /
  ``file_uploader`` / ``columns`` / ``expander`` degrade to their *default /
  first option* so code that reads a widget value keeps working. They emit a
  warning naming the widget so you know a UI interaction was silently collapsed
  to a default.
"""

from __future__ import annotations

import builtins
import contextlib
import os
import sys
import types
from typing import Any, Dict, List, Optional

VERBOSE = os.environ.get("HEADLESS_STREAMlit_VERBOSE", "0") == "1"


def _log(*args: Any) -> None:
    if VERBOSE:
        print("[headless-streamlit]", *args, file=sys.stderr, flush=True)


class _SessionState(dict):
    """Dict that also supports attribute access (``st.session_state.foo``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def add(self, value: Any) -> None:
        """Mimic Streamlit's ``session_state.<set>.add`` (used for outputs)."""
        if "outputs" not in self:
            self["outputs"] = set()
        self["outputs"].add(value)


# ---------------------------------------------------------------------------
# Widgets / UI helpers — collapse to defaults, never block.
# ---------------------------------------------------------------------------

def _warn(name: str, value: Any) -> Any:
    _log(f"widget {name} collapsed to default: {value!r}")
    return value


def selectbox(label: str, options: List[Any], index: int = 0, **kwargs: Any) -> Any:
    return _warn("selectbox", options[index] if options else None)


def select_slider(label: str, options: List[Any], value: Any = None, **kwargs: Any) -> Any:
    if value is None:
        value = options[0] if options else None
    return _warn("select_slider", value)


def radio(label: str, options: List[Any], index: int = 0, **kwargs: Any) -> Any:
    return _warn("radio", options[index] if options else None)


def checkbox(label: str, value: bool = False, **kwargs: Any) -> bool:
    return _warn("checkbox", value)


def slider(label: str, min_value: float = 0.0, max_value: float = 100.0,
           value: Any = None, **kwargs: Any) -> Any:
    if value is None:
        value = min_value
    return _warn("slider", value)


def text_input(label: str, value: str = "", **kwargs: Any) -> str:
    return _warn("text_input", value)


def text_area(label: str, value: str = "", **kwargs: Any) -> str:
    return _warn("text_area", value)


def number_input(label: str, min_value: Any = None, max_value: Any = None,
                 value: Any = None, **kwargs: Any) -> Any:
    if value is None:
        value = min_value if min_value is not None else 0
    return _warn("number_input", value)


def button(label: str, **kwargs: Any) -> bool:
    _log(f"button {label!r} treated as not-pressed")
    return False

def radio_selectbox_dummy(*args: Any, **kwargs: Any) -> Any:
    return None


def file_uploader(label: str, type: Any = None, **kwargs: Any) -> Optional[Any]:
    _log(f"file_uploader {label!r} returns None (no file in headless mode)")
    return None


def plotly_chart(fig: Any, **kwargs: Any) -> None:
    _log("plotly_chart ignored (no UI)")


def pyplot(*args: Any, **kwargs: Any) -> None:
    _log("pyplot ignored (no UI)")


def dataframe(*args: Any, **kwargs: Any) -> None:
    _log("dataframe ignored (no UI)")


@contextlib.contextmanager
def spinner(text: str = ""):
    _log(f"spinner: {text}")
    yield


def progress(value: float = 0.0) -> Any:
    _log(f"progress {value:.2f}")
    return _NullProgress()


class _NullProgress:
    def progress(self, value: float) -> None:
        _log(f"progress {value:.2f}")
    def empty(self) -> None:
        pass


def error(msg: Any) -> None:
    _log(f"error: {msg}")


def warning(msg: Any) -> None:
    _log(f"warning: {msg}")


def info(msg: Any) -> None:
    _log(f"info: {msg}")


def success(msg: Any) -> None:
    _log(f"success: {msg}")


def write(*args: Any, **kwargs: Any) -> None:
    _log("write:", args)


def markdown(*args: Any, **kwargs: Any) -> None:
    _log("markdown:", args)


def code(*args: Any, **kwargs: Any) -> None:
    _log("code:", args)


def caption(*args: Any, **kwargs: Any) -> None:
    _log("caption:", args)


def subheader(*args: Any, **kwargs: Any) -> None:
    _log("subheader:", args)


def header(*args: Any, **kwargs: Any) -> None:
    _log("header:", args)


def title(*args: Any, **kwargs: Any) -> None:
    _log("title:", args)


def expander(*args: Any, **kwargs: Any):
    @contextlib.contextmanager
    def _ctx():
        yield _ColumnLike()
    return _ctx()


def tabs(*args: Any, **kwargs: Any):
    return [_ColumnLike() for _ in args]


def columns(*args: Any, **kwargs: Any):
    n = args[0] if args and isinstance(args[0], (int, list, tuple)) else (len(args) if args else 1)
    if isinstance(n, (list, tuple)):
        n = len(n)
    return [_ColumnLike() for _ in range(int(n))]


class _ColumnLike:
    """Stand-in returned by columns()/expander/tabs so chained calls don't crash."""
    def __getattr__(self, name: str) -> Any:
        try:
            return globals()[name]
        except KeyError:
            def _noop(*a: Any, **k: Any) -> Any:
                _log(f"column.{name} ignored")
                return None
            return _noop


def empty(*args: Any, **kwargs: Any) -> Any:
    return _NullProgress()


def form(*args: Any, **kwargs: Any):
    @contextlib.contextmanager
    def _ctx():
        yield _ColumnLike()
    return _ctx()


def form_submit_button(*args: Any, **kwargs: Any) -> bool:
    return False


def set_page_config(*args: Any, **kwargs: Any) -> None:
    _log("set_page_config ignored")


def sidebar(*args: Any, **kwargs: Any) -> Any:
    return _ColumnLike()


class _column_config:
    @staticmethod
    def TextColumn(*args: Any, **kwargs: Any) -> Any:
        return None
    @staticmethod
    def NumberColumn(*args: Any, **kwargs: Any) -> Any:
        return None
    @staticmethod
    def ProgressColumn(*args: Any, **kwargs: Any) -> Any:
        return None


def get_stub() -> types.ModuleType:
    """Build and return a fresh ``streamlit``-shaped module."""
    m = types.ModuleType("streamlit")

    # ``stmol`` (and others) import ``streamlit.components.v1``; provide it so
    # third-party libs that assume a full Streamlit tree don't crash on import.
    components = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    def html(*a: Any, **k: Any) -> None:
        _log("components.html ignored")
    def iframe(*a: Any, **k: Any) -> None:
        _log("components.iframe ignored")
    v1.html = html
    v1.iframe = iframe
    components.v1 = v1
    m.components = components

    # Mark the stub as a package and pre-register submodules so
    # ``import streamlit.components.v1`` resolves without hitting disk.
    m.__path__ = []  # type: ignore[attr-defined]
    m.__package__ = "streamlit"
    components.__package__ = "streamlit.components"
    v1.__package__ = "streamlit.components.v1"
    m.__all__ = [name for name in dir(m) if not name.startswith("__")]

    # state
    m.session_state = _SessionState()
    m.session_state.update({
        "outputs": set(),
        "alignment_mapping": None,
        "result": None,
        "reference_seq": None,
        "proteins_list": None,
    })
    # widgets
    m.selectbox = selectbox
    m.select_slider = select_slider
    m.radio = radio
    m.checkbox = checkbox
    m.slider = slider
    m.text_input = text_input
    m.text_area = text_area
    m.number_input = number_input
    m.button = button
    m.file_uploader = file_uploader
    m.columns = columns
    m.tabs = tabs
    m.expander = expander
    m.form = form
    m.form_submit_button = form_submit_button
    m.empty = empty
    m.sidebar = sidebar
    m.columns = columns
    m.column_config = _column_config()
    # display (no-ops)
    m.plotly_chart = plotly_chart
    m.pyplot = pyplot
    m.dataframe = dataframe
    m.write = write
    m.markdown = markdown
    m.code = code
    m.caption = caption
    m.subheader = subheader
    m.header = header
    m.title = title
    m.spinner = spinner
    m.progress = progress
    m.error = error
    m.warning = warning
    m.info = info
    m.success = success
    m.set_page_config = set_page_config
    m.ColumnLike = _ColumnLike
    return m


# A module-level singleton so you can do:
#   sys.modules["streamlit"] = headless_streamlit.stub
stub = get_stub()


def install() -> types.ModuleType:
    """Install THIS stub as ``sys.modules['streamlit']`` and return it.

    Call this before importing any ``space.*`` module::

        import headless_streamlit
        headless_streamlit.install()
        from space.analysis import run_al2co

    Also registers submodules (``streamlit.components.v1``) so third-party
    imports like ``from stmol import showmol`` don't crash.
    """
    # Ensure the stub is fully built
    _stub = stub
    # Register child submodules for nested imports (e.g., stmol → components.v1)
    comp = _stub.components
    sys.modules.setdefault("streamlit", _stub)
    sys.modules.setdefault("streamlit.components", comp)
    sys.modules.setdefault("streamlit.components.v1", comp.v1)
    # Also set as the canonical streamlit to catch any late imports
    sys.modules["streamlit"] = _stub
    return _stub


if __name__ == "__main__":
    install()
    print("headless Streamlit stub installed as sys.modules['streamlit']")
    print("session_state:", dict(stub.session_state))