from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

PathLike = str | Path

_CHAPTER_LINE_RE = re.compile(r"^chapter\s+(?P<number>[a-z0-9ivxlcdm]+)\b(?P<suffix>.*)$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
_HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")
_ROMAN_VALUES = {
    "i": 1,
    "v": 5,
    "x": 10,
    "l": 50,
    "c": 100,
    "d": 500,
    "m": 1000,
}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def parse_pdf_to_markdown(pdf_path: PathLike, output_path: PathLike | None = None) -> str:
    """Extract PDF text and write a simple markdown file suitable for benchmark slicing."""
    source = Path(pdf_path)
    pages = _extract_pdf_pages(source)
    markdown = _pages_to_markdown(source.name, pages)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
    return markdown


def write_chapter_text_file(
    markdown_path: PathLike,
    chapter: str | int,
    output_path: PathLike | None = None,
) -> Path:
    """Write one markdown chapter as plain text and return the written path."""
    source = Path(markdown_path)
    chapter_text = extract_chapter_text(source.read_text(encoding="utf-8"), chapter)
    destination = Path(output_path) if output_path is not None else _default_chapter_output_path(source, chapter)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(chapter_text, encoding="utf-8")
    return destination


def extract_chapter_text(markdown: str, chapter: str | int) -> str:
    """Return a plain-text chapter from markdown using a heading such as 'Chapter 1'."""
    lines = markdown.splitlines()
    start_index: int | None = None
    start_level: int | None = None

    for index, line in enumerate(lines):
        heading = _parse_heading(line)
        if heading is None:
            continue
        level, title = heading
        if _chapter_matches(title, chapter):
            start_index = index
            start_level = level
            break

    if start_index is None or start_level is None:
        raise ValueError(f"Chapter not found: {chapter}")

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        heading = _parse_heading(lines[index])
        if heading is None:
            continue
        level, title = heading
        if level <= start_level and _is_chapter_heading(title):
            end_index = index
            break

    return _markdown_to_plain_text(lines[start_index:end_index])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PDF source text for Whispbook benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    markdown = subparsers.add_parser("markdown", help="Convert a PDF to simple markdown.")
    markdown.add_argument("pdf", type=Path)
    markdown.add_argument("-o", "--output", type=Path, default=None)

    chapter = subparsers.add_parser("chapter", help="Write one markdown chapter to a plain text file.")
    chapter.add_argument("markdown", type=Path)
    chapter.add_argument("-c", "--chapter", required=True)
    chapter.add_argument("-o", "--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "markdown":
        markdown = parse_pdf_to_markdown(args.pdf, args.output)
        if args.output is None:
            print(markdown, end="")
        else:
            print(args.output)
        return 0

    if args.command == "chapter":
        output_path = write_chapter_text_file(args.markdown, args.chapter, args.output)
        print(output_path)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _extract_pdf_pages(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to parse PDFs. Install project dependencies with `uv sync`."
        ) from exc

    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _pages_to_markdown(source_name: str, pages: Iterable[str]) -> str:
    lines: list[str] = [f"<!-- source: {source_name} -->"]
    for page_number, page_text in enumerate(pages, start=1):
        lines.extend(["", f"<!-- page: {page_number} -->", ""])
        lines.extend(_page_text_to_markdown_lines(page_text))

    return "\n".join(_collapse_blank_lines(lines)).strip() + "\n"


def _page_text_to_markdown_lines(page_text: str) -> list[str]:
    converted: list[str] = []
    for line in _normalize_pdf_text(page_text).splitlines():
        stripped = line.strip()
        if not stripped:
            converted.append("")
            continue
        if _is_chapter_heading(stripped):
            converted.extend([f"## {stripped}", ""])
            continue
        converted.append(stripped)
    return converted


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _markdown_to_plain_text(lines: list[str]) -> str:
    plain_lines: list[str] = []
    for line in lines:
        if _HTML_COMMENT_RE.match(line):
            continue
        heading = _parse_heading(line)
        if heading is not None:
            plain_lines.append(heading[1])
            continue
        plain_lines.append(line.strip())

    return "\n".join(_collapse_blank_lines(plain_lines)).strip() + "\n"


def _collapse_blank_lines(lines: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank
    return collapsed


def _parse_heading(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line.strip())
    if match is None:
        return None
    return len(match.group("marks")), match.group("title").strip()


def _is_chapter_heading(title: str) -> bool:
    return _CHAPTER_LINE_RE.match(_normalize_heading(title)) is not None


def _chapter_matches(title: str, chapter: str | int) -> bool:
    normalized_title = _normalize_heading(title)
    normalized_chapter = _normalize_heading(str(chapter))
    if normalized_title == normalized_chapter:
        return True

    title_number = _chapter_number(normalized_title)
    chapter_number = _chapter_number(normalized_chapter)
    if title_number is not None and chapter_number is not None:
        return title_number == chapter_number

    return normalized_title.startswith(f"chapter {normalized_chapter}")


def _chapter_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)

    match = _CHAPTER_LINE_RE.match(value)
    token = match.group("number") if match else value
    if token.isdigit():
        return int(token)
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    return _roman_to_int(token)


def _roman_to_int(value: str) -> int | None:
    if not value or any(char not in _ROMAN_VALUES for char in value):
        return None

    total = 0
    previous = 0
    for char in reversed(value):
        current = _ROMAN_VALUES[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def _normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _default_chapter_output_path(markdown_path: Path, chapter: str | int) -> Path:
    chapter_slug = re.sub(r"[^a-z0-9]+", "-", str(chapter).lower()).strip("-") or "chapter"
    return markdown_path.with_name(f"{markdown_path.stem}-{chapter_slug}.txt")


if __name__ == "__main__":
    raise SystemExit(main())
