"""Keep published source, documentation, and filenames free of Chinese text.

Raster image labels require visual review; this check covers text content only.
"""

from pathlib import Path
import subprocess
import unicodedata

import pytest


ROOT = Path(__file__).resolve().parents[1]


def is_chinese_character(char: str) -> bool:
    name = unicodedata.name(char, "")
    point = ord(char)
    return (
        name.startswith(("CJK", "IDEOGRAPHIC", "KANGXI", "BOPOMOFO"))
        or 0x3000 <= point <= 0x303F
        or 0xFF01 <= point <= 0xFF60
        or 0xFFE0 <= point <= 0xFFE6
    )


@pytest.mark.parametrize("point", [0x4E2D, 0x6587, 0x3002, 0xFF0C, 0x20000])
def test_detects_chinese_text_and_punctuation(point):
    assert is_chinese_character(chr(point))


def test_allows_english_and_scientific_notation():
    assert not any(is_chinese_character(c) for c in "ECG-RePAIR: PR > 200 ms; 95% CI")


def test_repository_contains_no_chinese_text():
    if not (ROOT / ".git").exists():
        pytest.skip("Repository-wide language audit requires a Git checkout")
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    failures = []
    for relative in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        if any(is_chinese_character(c) for c in relative):
            failures.append(f"Non-English filename: {relative}")
        path = ROOT / relative
        if not path.is_file():
            continue
        raw = path.read_bytes()
        try:
            if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
                content = raw.decode("utf-32")
            elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                content = raw.decode("utf-16")
            elif b"\0" in raw:
                continue
            else:
                content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # Binary assets, including the workflow image, are reviewed separately.
            continue
        for number, line in enumerate(content.splitlines(), 1):
            points = sorted({f"U+{ord(c):04X}" for c in line if is_chinese_character(c)})
            if points:
                failures.append(f"{relative}:{number}: {', '.join(points)}")
    assert not failures, "Chinese text found in repository:\n" + "\n".join(failures)
