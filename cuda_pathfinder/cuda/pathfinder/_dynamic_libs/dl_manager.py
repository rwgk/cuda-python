# cuda/pathfinder/_dynamic_libs/dl_manager.py
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Literal

from .load_dl_common import LoadedDL, DynamicLibNotFoundError, load_dependencies

if os.name == "nt":
    from .load_dl_windows import (
        abs_path_for_dynamic_library,
        load_with_abs_path,
        load_with_system_search,
    )
    import ctypes
    import ctypes.wintypes as wt
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _FreeLibrary = _k32.FreeLibrary
    _FreeLibrary.argtypes = [wt.HMODULE]
    _FreeLibrary.restype = wt.BOOL
    def _os_close(abs_path: str, handle_uint: int) -> None:
        # Idempotent close via Windows loader; ignore failure on double-close.
        _FreeLibrary(ctypes.c_void_p(handle_uint))
else:
    from .load_dl_linux import (
        abs_path_for_dynamic_library,
        load_with_abs_path,
        load_with_system_search,
    )
    import ctypes, ctypes.util
    _libdl_path = ctypes.util.find_library("dl")
    _libdl = ctypes.CDLL(_libdl_path) if _libdl_path else None
    if _libdl:
        _libdl.dlclose.argtypes = [ctypes.c_void_p]
        _libdl.dlclose.restype = ctypes.c_int
    def _os_close(abs_path: str, handle_uint: int) -> None:
        # Idempotent close; ignore failure if already closed elsewhere.
        if _libdl:
            _libdl.dlclose(ctypes.c_void_p(handle_uint))

# -------- Handle types -------------------------------------------------------

@dataclass(frozen=True)
class BorrowedDL:
    """Non-owning view; do not close."""
    abs_path: str
    _handle_uint: int
    owner: Literal["pinned"] = "pinned"

class DLLease:
    """One OS reference (LoadLibrary/dlopen). Must be closed exactly once."""
    __slots__ = ("abs_path", "_handle_uint", "_closed")
    def __init__(self, abs_path: str, handle_uint: int) -> None:
        self.abs_path = abs_path
        self._handle_uint = handle_uint
        self._closed = False
    def close(self) -> None:
        if not self._closed:
            _os_close(self.abs_path, self._handle_uint)
            self._closed = True
    def __enter__(self) -> "DLLease": return self
    def __exit__(self, exc_type, exc, tb) -> None: self.close()

@dataclass(frozen=True)
class DLPinned:
    """Process-lifetime reference. Exposes a borrow-only view."""
    abs_path: str
    _handle_uint: int
    def borrow(self) -> BorrowedDL:
        return BorrowedDL(self.abs_path, self._handle_uint)

# -------- Registry (by real absolute path) ----------------------------------

@dataclass
class _Entry:
    pinned: Optional[DLPinned] = None

_registry: Dict[str, _Entry] = {}
_lock = threading.RLock()

def _key_for_abs_path(p: str) -> str:
    return os.path.realpath(p)

# -------- Public API ---------------------------------------------------------

def pin_nvidia_dynamic_lib(libname: str) -> DLPinned:
    """
    Ensure `libname` is loaded and pinned for the rest of the process.
    Safe to call multiple times; returns the same pinned object.
    """
    # Load direct dependencies first (keeps existing behavior)
    load_dependencies(libname, load_with_system_search)
    abs_path = abs_path_for_dynamic_library(libname)  # raises if not found
    key = _key_for_abs_path(abs_path)
    with _lock:
        ent = _registry.setdefault(key, _Entry())
        if ent.pinned is not None:
            return ent.pinned
        # One OS open that we never close (process lifetime)
        loaded: LoadedDL = load_with_abs_path(libname, abs_path)
        pinned = DLPinned(abs_path=loaded.abs_path or abs_path, _handle_uint=loaded._handle_uint)
        ent.pinned = pinned
        return pinned

def acquire_nvidia_dynamic_lib(libname: str) -> DLLease:
    """
    Acquire a *lease* (one OS ref). Use `with` or call `.close()`.
    Independent of whether the lib is pinned; if pinned, this just adds one more ref.
    """
    abs_path = abs_path_for_dynamic_library(libname)
    loaded: LoadedDL = load_with_abs_path(libname, abs_path)
    return DLLease(abs_path=loaded.abs_path or abs_path, handle_uint=loaded._handle_uint)

def borrow_pinned_nvidia_dynamic_lib(libname: str) -> BorrowedDL:
    """
    Borrow a handle from the pinned library. Raises if not pinned.
    (Linux has no zero-cost 'borrow' for arbitrary preloaded libs.)
    """
    abs_path = abs_path_for_dynamic_library(libname)
    key = _key_for_abs_path(abs_path)
    with _lock:
        ent = _registry.get(key)
        if not ent or not ent.pinned:
            raise RuntimeError(
                f"{libname} is not pinned; call pin_nvidia_dynamic_lib('{libname}') first."
            )
        return ent.pinned.borrow()

@dataclass(frozen=True)
class DLWeak:
    """Weak reference keyed by absolute path."""
    abs_path: str
    def lock(self, *, as_lease: bool = True) -> Optional[DLLease]:
        """
        If the lib is (still) loaded, return a DLLease (increments OS ref).
        If not loaded and as_lease=True, it will load and return a DLLease.
        If not loaded and as_lease=False, returns None.
        """
        try:
            if as_lease:
                # Loads if necessary, returning a real ref
                return acquire_nvidia_dynamic_lib(_libname_from_abs(self.abs_path))
            else:
                # Borrow only from pinned
                return None  # explicit: borrow requires a pin, provided by borrow_pinned...
        except DynamicLibNotFoundError:
            return None

# Helper to map back to a supported logical name, if you need it.
# Simplest is to accept abs_path here; if not needed, drop this helper.
def _libname_from_abs(abs_path: str) -> str:
    return os.path.basename(abs_path)  # or a real reverse map if you have one
