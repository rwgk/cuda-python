# cuda/pathfinder/_dynamic_libs/dl_manager.py
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Literal

from .load_dl_common import LoadedDL, DynamicLibNotFoundError, load_dependencies

if os.name == "nt":
    from .load_dl_windows import (
        load_with_abs_path,
        load_with_system_search,
        check_if_already_loaded_from_elsewhere,
    )
    import ctypes
    import ctypes.wintypes as wt
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _FreeLibrary = _k32.FreeLibrary
    _FreeLibrary.argtypes = [wt.HMODULE]
    _FreeLibrary.restype = wt.BOOL
    def _os_close(abs_path: str, handle_uint: int) -> None:
        _FreeLibrary(ctypes.c_void_p(handle_uint))
else:
    from .load_dl_linux import (
        load_with_abs_path,
        load_with_system_search,
        check_if_already_loaded_from_elsewhere,
    )
    import ctypes, ctypes.util
    _libdl_path = ctypes.util.find_library("dl")
    _libdl = ctypes.CDLL(_libdl_path) if _libdl_path else None
    if _libdl:
        _libdl.dlclose.argtypes = [ctypes.c_void_p]
        _libdl.dlclose.restype = ctypes.c_int
    def _os_close(abs_path: str, handle_uint: int) -> None:
        if _libdl:
            _libdl.dlclose(ctypes.c_void_p(handle_uint))


@dataclass(frozen=True)
class BorrowedDL:
    abs_path: str
    _handle_uint: int
    owner: Literal["pinned"] = "pinned"

class DLLease:
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
    abs_path: str
    _handle_uint: int
    def borrow(self) -> BorrowedDL:
        return BorrowedDL(self.abs_path, self._handle_uint)

@dataclass
class _Entry:
    pinned: Optional[DLPinned] = None

_registry: Dict[str, _Entry] = {}
_name_to_key: Dict[str, str] = {}
_lock = threading.RLock()

def _key_for_abs_path(p: str) -> str:
    return os.path.realpath(p)

def register_pinned(libname: str, loaded: LoadedDL) -> LoadedDL:
    """
    Record `loaded` as the pinned instance for `libname` without changing OS refcounts.
    Returns `loaded` unchanged.
    """
    if not loaded or not loaded.abs_path:
        raise DynamicLibNotFoundError(f"Failed to pin {libname!r}: no absolute path available")
    key = _key_for_abs_path(loaded.abs_path)
    with _lock:
        ent = _registry.setdefault(key, _Entry())
        if ent.pinned is None:
            ent.pinned = DLPinned(abs_path=loaded.abs_path, _handle_uint=loaded._handle_uint)
        _name_to_key[libname] = key
        return LoadedDL(ent.pinned.abs_path, loaded.was_already_loaded_from_elsewhere, ent.pinned._handle_uint)

def pin_nvidia_dynamic_lib(libname: str) -> LoadedDL:
    """
    Ensure `libname` is loaded and pinned for the rest of the process.
    Safe to call multiple times; returns the same logical handle.
    """
    # Fast path: already pinned
    with _lock:
        key = _name_to_key.get(libname)
        if key:
            ent = _registry.get(key)
            if ent and ent.pinned:
                p = ent.pinned
                return LoadedDL(p.abs_path, False, p._handle_uint)

    # Otherwise, load it (system search + dependencies)
    load_dependencies(libname, load_with_system_search)

    # If someone else loaded it, upgrade to an owned ref if needed
    pre = check_if_already_loaded_from_elsewhere(libname, have_abs_path=False)
    if pre is not None:
        if os.name == "nt" and pre.abs_path:
            owned = load_with_abs_path(libname, pre.abs_path)  # increments refcount
            return register_pinned(libname, owned)
        return register_pinned(libname, pre)  # Linux NOLOAD already increments

    loaded = load_with_system_search(libname)
    if loaded is None:
        raise DynamicLibNotFoundError(f"Failed to load dynamic library {libname!r} via system search")
    return register_pinned(libname, loaded)

def acquire_nvidia_dynamic_lib(libname: str) -> DLLease:
    """
    Acquire a *lease* (one OS ref). Use `with` or call `.close()`.
    Independent of whether the lib is pinned; if pinned, this just adds one more ref.
    """
    loaded = load_with_system_search(libname)
    if loaded is None:
        raise DynamicLibNotFoundError(f"Failed to load dynamic library {libname!r} via system search")
    return DLLease(abs_path=loaded.abs_path, handle_uint=loaded._handle_uint)

def borrow_pinned_nvidia_dynamic_lib(libname: str) -> BorrowedDL:
    """
    Borrow a handle from the pinned library. Raises if not pinned.
    """
    with _lock:
        key = _name_to_key.get(libname)
        if not key:
            raise RuntimeError(f"{libname} is not pinned; call pin_nvidia_dynamic_lib('{libname}') first.")
        ent = _registry.get(key)
        if not ent or not ent.pinned:
            raise RuntimeError(f"{libname} is not pinned; call pin_nvidia_dynamic_lib('{libname}') first.")
        return ent.pinned.borrow()

@dataclass(frozen=True)
class DLWeak:
    abs_path: str
    def lock(self, *, as_lease: bool = True) -> Optional[DLLease]:
        try:
            if as_lease:
                loaded = load_with_system_search(os.path.basename(self.abs_path))
                if loaded is None:
                    return None
                return DLLease(abs_path=loaded.abs_path, handle_uint=loaded._handle_uint)
            else:
                return None
        except DynamicLibNotFoundError:
            return None

def pin_from_abs_path(libname: str, abs_path: str) -> LoadedDL:
    """Load the library from a known absolute path and pin it."""
    loaded = load_with_abs_path(libname, abs_path)
    return register_pinned(libname, loaded)
