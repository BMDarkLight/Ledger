"""Splitting documents into the units that become retrieval receipts.

A chunk is the smallest thing Ledger can cite, so a boundary in the wrong place
either buries a fact among unrelated text or splits it away from the context
that makes it findable.

The corpus is reStructuredText, which carries its own structure, so sections are
respected instead of sliding a fixed window over the raw characters.
"""

import re
from dataclasses import dataclass

# An RST section header: a title line followed by a line of repeated punctuation
# at least as long as the title.
_UNDERLINE = re.compile(r"^([=\-~^\"'#*+`:._]){3,}\s*$")

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_PARAGRAPHS = 1

METADATA_SECTION = "Metadata"
"""The RFC-822 style preamble: PEP, Title, Author, Status, Type, Created.

Kept as its own chunk because a surprising share of real questions are about it
("who wrote it", "what status is it") and those facts appear nowhere else.
"""


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    ordinal: int
    section: str
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.ordinal}"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split RST into (section title, body) pairs, preamble first."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [(METADATA_SECTION, [])]

    i = 0
    while i < len(lines):
        line = lines[i]
        is_header = (
            i + 1 < len(lines)
            and line.strip()
            and _UNDERLINE.match(lines[i + 1])
            and len(lines[i + 1].strip()) >= len(line.strip())
        )
        if is_header:
            sections.append((line.strip(), []))
            i += 2
            continue
        sections[-1][1].append(line)
        i += 1

    return [(title, "\n".join(body).strip()) for title, body in sections if "".join(body).strip()]


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _pack(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedily fill windows, carrying `overlap` paragraphs across the boundary.

    The overlap exists so a fact stated at the end of one window is still
    reachable from the next one's context.
    """
    windows: list[str] = []
    current: list[str] = []
    size = 0

    for para in paragraphs:
        # A single oversized paragraph (a long code block, usually) becomes its
        # own chunk rather than being cut mid-line.
        if len(para) > max_chars and not current:
            windows.append(para)
            continue
        if current and size + len(para) > max_chars:
            windows.append("\n\n".join(current))
            current = current[-overlap:] if overlap else []
            size = sum(len(p) for p in current)
        current.append(para)
        size += len(para)

    if current:
        windows.append("\n\n".join(current))
    return windows


def chunk_document(
    doc_id: str,
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> list[Chunk]:
    """Chunk a document, prefixing each chunk with its section for context.

    The section title is prepended to the embedded text because "Author" or
    "Rationale" is often the only thing that distinguishes an otherwise generic
    passage, and because it makes a retrieved receipt readable on its own.
    """
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for window in _pack(_paragraphs(body), max_chars, overlap):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    ordinal=len(chunks),
                    section=section,
                    text=f"{section}\n\n{window}" if section != METADATA_SECTION else window,
                )
            )
    return chunks
