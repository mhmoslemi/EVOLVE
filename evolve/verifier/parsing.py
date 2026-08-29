"""Deterministic parsing helpers for generated scientific programs."""

from __future__ import annotations

import re
from typing import Optional


def extract_python_code(response: str) -> Optional[str]:
    """Extract the final Python block, including a truncated final block."""

    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "<think>" in response and "</think>" not in response:
        return None
    matches = re.findall(r"```python\s*\n?(.*?)```", response, re.DOTALL)
    if matches and matches[-1].strip():
        return matches[-1].strip()
    truncated = re.search(r"```python\s*\n?(.*)$", response, re.DOTALL)
    if truncated:
        code = re.sub(r"\n?```\s*$", "", truncated.group(1)).strip()
        if code:
            return code
    matches = re.findall(r"```\s*\n?(.*?)```", response, re.DOTALL)
    if matches and matches[-1].strip():
        return matches[-1].strip()
    stripped = response.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
        return stripped
    return None


__all__ = ["extract_python_code"]
