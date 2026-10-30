from pathlib import Path

import pytest

from whispbook_gender_specdec.utils import pdf_parser


def test_parse_pdf_to_markdown_writes_chapter_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "sample-book.pdf"
    source_pdf.write_bytes(b"%PDF-1.7 fixture")
    output_md = tmp_path / "sample-book.md"

    monkeypatch.setattr(
        pdf_parser,
        "_extract_pdf_pages",
        lambda path: [
            "CHAPTER I\nNora speaks first.\n\nHelmer answers.",
            "CHAPTER II: Later\nMrs Linde enters.",
        ],
    )

    markdown = pdf_parser.parse_pdf_to_markdown(source_pdf, output_md)

    assert markdown == output_md.read_text()
    assert "<!-- source: sample-book.pdf -->" in markdown
    assert "<!-- page: 1 -->" in markdown
    assert "## CHAPTER I" in markdown
    assert "Nora speaks first.\n\nHelmer answers." in markdown
    assert "## CHAPTER II: Later" in markdown


def test_write_chapter_text_file_extracts_requested_chapter(tmp_path: Path) -> None:
    markdown_path = tmp_path / "book.md"
    markdown_path.write_text(
        "\n".join(
            [
                "<!-- source: book.pdf -->",
                "",
                "## CHAPTER I",
                "",
                "Nora speaks first.",
                "",
                "### Scene note",
                "",
                "Helmer answers.",
                "",
                "## CHAPTER II: Later",
                "",
                "Mrs Linde enters.",
            ]
        )
    )
    output_txt = tmp_path / "chapter-1.txt"

    written_path = pdf_parser.write_chapter_text_file(markdown_path, "1", output_txt)

    assert written_path == output_txt
    assert output_txt.read_text() == "CHAPTER I\n\nNora speaks first.\n\nScene note\n\nHelmer answers.\n"


def test_write_chapter_text_file_reports_missing_chapter(tmp_path: Path) -> None:
    markdown_path = tmp_path / "book.md"
    markdown_path.write_text("## CHAPTER I\n\nOnly one chapter.\n")

    with pytest.raises(ValueError, match="Chapter not found"):
        pdf_parser.write_chapter_text_file(markdown_path, "2", tmp_path / "chapter-2.txt")


def test_pdf_parser_cli_writes_chapter_text_file(tmp_path: Path) -> None:
    markdown_path = tmp_path / "book.md"
    markdown_path.write_text("## CHAPTER I\n\nNora speaks.\n\n## CHAPTER II\n\nHelmer speaks.\n")
    output_txt = tmp_path / "chapter.txt"

    exit_code = pdf_parser.main(
        [
            "chapter",
            str(markdown_path),
            "--chapter",
            "1",
            "--output",
            str(output_txt),
        ]
    )

    assert exit_code == 0
    assert output_txt.read_text() == "CHAPTER I\n\nNora speaks.\n"
