"""Runtime Panda URDF loading."""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET

def _resolve_urdf(value: str) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
    else:
        try:
            from mani_skill import PACKAGE_ASSET_DIR
        except ImportError as exc:
            raise RuntimeError(
                "ManiSkill is unavailable; pass --urdf-path explicitly or run in "
                "the ManiSkill development environment."
            ) from exc
        path = Path(PACKAGE_ASSET_DIR) / "robots" / "panda" / "panda_v2.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {path}")
    return path

def _sanitized_urdf_text(path: Path) -> str:
    """Remove SAPIEN-only dynamics attributes that pykinematics warns about."""

    root = ET.parse(path).getroot()
    allowed = {"damping", "friction"}
    for dynamics in root.findall(".//dynamics"):
        for key in list(dynamics.attrib):
            if key not in allowed:
                del dynamics.attrib[key]
    return ET.tostring(root, encoding="unicode")
