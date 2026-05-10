from __future__ import annotations

import argparse
import re
from pathlib import Path


SHARED_HEADING_RE = re.compile(r"^# Shared Methods\b.*$", re.MULTILINE)
AGENT_KB_HEADING_RE = re.compile(r"^# .+ Methods (?:&|—) Knowledge Base\s*$", re.MULTILINE)
DATED_SECTION_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}", re.MULTILINE)
PROVEN_SECTION_RE = re.compile(r"^## \[PROVEN\]", re.MULTILINE)
TOP_HEADING_RE = re.compile(r"(?m)^(?=#{1,2} )")
WORLD_CLASS_HEADING_RE = re.compile(r"^## \[PROVEN\] World-Class Analytics Default\s*$", re.MULTILINE)
WORLD_CLASS_BODY_RE = re.compile(
    r"\n*Every agent must treat world-class analytics as the default standard.*?"
    r"DATASET_RISK_REGISTER, DATA_QUALITY_AUDIT, BUSINESS_EDA_FRAME, FEATURE_GOVERNANCE, "
    r"PRODUCTION_READINESS, PATTERN_VALIDITY, BUSINESS_DECISION_BRIEF, VISUAL_QC, WORLD_CLASS_QC\.\s*",
    re.DOTALL,
)


DECISION_POINTER = """## Decision Quality Gate (mandatory)
ใช้กฎกลางจาก `knowledge_base/shared_methods.md` ก่อนทุก decision สำคัญ
- ตรวจหลักฐานจากไฟล์จริงและ output ล่าสุดก่อนเลือกวิธี
- เทียบอย่างน้อย 2 ทางเลือก หรืออธิบายว่าทำไมมีทางเดียว
- บันทึก `DECISION_CHECK` พร้อม Evidence, Risk Check, Confidence และ Verdict
- ถ้าหลักฐานไม่พอหรือ confidence ต่ำ ให้หยุดด้วย `STOP_AND_REPAIR`, `LOOP_BACK`, หรือ `ASK_USER`
"""


def normalize_section(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line).strip().lower()


def remove_embedded_shared_methods(text: str) -> str:
    """Remove copied shared_methods blocks from agent-specific KB files.

    The shared file is loaded separately by KnowledgeBase.load(), so embedded
    copies inside *_methods.md only add prompt bloat and stale instructions.
    """

    while True:
        match = SHARED_HEADING_RE.search(text)
        if not match:
            return text

        next_candidates = []
        for regex in (AGENT_KB_HEADING_RE, DATED_SECTION_RE, PROVEN_SECTION_RE):
            next_match = regex.search(text, match.end())
            if next_match:
                next_candidates.append(next_match.start())

        end = min(next_candidates) if next_candidates else len(text)
        text = text[: match.start()] + text[end:]


def dedupe_repeated_sections(text: str) -> str:
    """Drop later repeated markdown sections using normalized exact matching."""

    parts = [part for part in TOP_HEADING_RE.split(text) if part.strip()]
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        key = normalize_section(part)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(part.strip())
    return "\n\n".join(kept).strip() + "\n"


def remove_global_world_class_sections(text: str) -> str:
    """Remove copied global world-class blocks from agent KB files."""

    while True:
        match = WORLD_CLASS_HEADING_RE.search(text)
        if not match:
            return text
        next_match = re.search(r"(?m)^#{1,2} ", text[match.end() :])
        end = match.end() + next_match.start() if next_match else len(text)
        text = text[: match.start()] + text[end:]


def remove_global_world_class_body(text: str) -> str:
    return WORLD_CLASS_BODY_RE.sub("\n", text)


def ensure_decision_pointer(text: str) -> str:
    if "## Decision Quality Gate (mandatory)" in text:
        return text
    return DECISION_POINTER.strip() + "\n\n" + text.strip() + "\n"


def compact_duplicate_headings(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    previous_heading = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and stripped == previous_heading:
            continue
        out.append(line.rstrip())
        previous_heading = stripped if stripped.startswith("#") else ""
    compacted = "\n".join(out)
    compacted = re.sub(r"\n{3,}", "\n\n", compacted)
    return compacted.strip() + "\n"


def clean_file(path: Path) -> tuple[int, int]:
    before = path.read_text(encoding="utf-8", errors="ignore")
    after = remove_embedded_shared_methods(before)
    after = remove_global_world_class_sections(after)
    after = remove_global_world_class_body(after)
    after = dedupe_repeated_sections(after)
    after = ensure_decision_pointer(after)
    after = compact_duplicate_headings(after)
    if after != before:
        path.write_text(after, encoding="utf-8")
    return len(before), len(after)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("knowledge_dir", type=Path)
    args = parser.parse_args()

    for path in sorted(args.knowledge_dir.glob("*_methods.md")):
        if path.name == "shared_methods.md":
            continue
        before, after = clean_file(path)
        if before != after:
            print(f"{path.name}: {before} -> {after} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
