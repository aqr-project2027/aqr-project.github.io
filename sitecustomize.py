from __future__ import annotations

import os
from pathlib import Path


def _add_local_dll_dir() -> None:
    dll_dir = Path(__file__).resolve().parent / ".dlls"
    if not dll_dir.exists():
        return
    os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))


_add_local_dll_dir()
