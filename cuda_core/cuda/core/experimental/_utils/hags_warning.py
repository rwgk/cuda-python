# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0

# Windows only:
# Issue a warning if
# Hardware‑Accelerated GPU Scheduling (HAGS)
# is disabled.

import sys

if sys.platform != "win32":

    def warn_if_hags_is_disabled() -> None:
        pass
else:
    import warnings
    import winreg

    def _hags_is_disabled() -> bool:
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            )
            value, _ = winreg.QueryValueEx(key, "HwSchMode")
            winreg.CloseKey(key)
            return value == 1  # Explicitly disabled
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(e)  # XXX XXX XXX NEED TO CONFIRM THAT THE ABOVE ACTUALLY WORKS
            # Silent if key/value is missing or unreadable
            pass
        return False

    def warn_if_hags_is_disabled() -> None:
        if _hags_is_disabled():
            assert 0
            warnings.warn(
                message=r"Hardware‑Accelerated GPU Scheduling (HAGS) is disabled:"
                r" CUDA event timings may not be reliable. Please consider enabling HAGS."
                r" (Hint: SYSTEM\CurrentControlSet\Control\GraphicsDrivers HwSchMode 0)",
                stacklevel=1,
            )


if __name__ == "__main__":
    warn_if_hags_is_disabled()
