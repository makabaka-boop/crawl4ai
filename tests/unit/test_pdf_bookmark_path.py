"""Tests for bookmark/outline-path chapter extraction in the PDF processor.

Manual page numbers break across manual revisions: a chapter that started on
page 42 of version 1.3 may start on page 47 of version 1.4. ``bookmark_path``
selects a chapter structurally, from the PDF outline, instead of from a fixed
page offset. These tests pin the contract:

* the path matches bookmark titles *exactly*, level by level, and must resolve
  to exactly one bookmark (zero -> error, several -> error);
* the chapter starts on the matched bookmark's page and ends on the page
  before the first later bookmark at the same or a higher (shallower) level;
* when start and boundary bookmarks share a page that page is retained, and
  with no boundary the chapter runs to the end of the document;
* ``max_pages`` is counted from the chapter start page;
* only retained pages are processed, so images are extracted/saved for those
  pages only, while ``metadata.pages`` still reports the whole PDF;
* retained pages keep their original document-wide page numbers (HTML
  ``data-page`` included);
* no path (``None`` or empty) preserves the original from-page-1 behaviour.

The fixtures build real, minimal PDFs: hand-written text pages (no fpdf2
dependency) cloned through a pypdf ``PdfWriter`` that attaches the outline.
"""

import re

import pytest

pypdf = pytest.importorskip("pypdf")
from pypdf import PdfReader, PdfWriter

from crawl4ai.processors.pdf.processor import (
    NaivePDFProcessorStrategy,
    PDFBookmarkPathError,
)
from crawl4ai.processors.pdf import PDFContentScrapingStrategy

N_PAGES = 12


# ── fixture builders ────────────────────────────────────────────────────


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_lines(page_number: int):
    return [
        f"Chapter Heading Number {page_number}",
        "Assessment Overview",
        f"this is the unique body of page {page_number}. " * 10,
    ]


def _write_text_pdf(path, n_pages: int = N_PAGES) -> None:
    """Write a minimal valid multi-page PDF with plain-text content."""
    objects = []
    kids = " ".join(f"{3 + 3 * i} 0 R" for i in range(n_pages))
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")
    for i in range(n_pages):
        page_obj = 3 + 3 * i
        content_obj = page_obj + 1
        font_obj = page_obj + 2
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>"
        )
        stream_lines = ["BT /F1 12 Tf 72 720 Td"]
        for j, line in enumerate(_page_lines(i + 1)):
            if j:
                stream_lines.append("0 -16 Td")
            stream_lines.append(f"({_escape_pdf_text(line)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1")
        objects.append((f"<< /Length {len(stream)} >>\nstream\n", stream, "\nendstream"))
        objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode("latin-1")
        if isinstance(obj, tuple):
            out += obj[0].encode("latin-1") + obj[1] + obj[2].encode("latin-1")
        else:
            out += obj.encode("latin-1")
        out += b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("latin-1")
    path.write_bytes(bytes(out))


def _attach_outline(src_path, dst_path, spec):
    """Clone the text PDF and attach an outline.

    ``spec`` is a list of (title, 0-based page, parent_key_or_None, key)
    tuples-ish; easier: a list of dicts with keys title/page/parent/key.
    """
    writer = PdfWriter(clone_from=str(src_path))
    nodes = {}
    for item in spec:
        parent = nodes.get(item["parent"]) if item.get("parent") else None
        nodes[item["key"]] = writer.add_outline_item(
            item["title"], item["page"], parent=parent
        )
    with open(dst_path, "wb") as f:
        writer.write(f)


def _build_outlined_pdf(path):
    """12 pages. Layout (1-based page numbers in comments):

    Chapter Alpha ........................ page 2
      Alpha 1 ............................ page 3
      Alpha 2 ............................ page 4
    Chapter Beta ......................... page 7
      Beta 1 ............................. page 7  (same page as Beta)
      Beta 2 ............................. page 7  (same page as Beta 1,
                                                    its boundary)
      Beta 3 ............................. page 9
    Chapter Gamma ........................ page 11 (last chapter, to end)
    """
    text_pdf = path.with_name(path.stem + "_text.pdf")
    _write_text_pdf(text_pdf)
    _attach_outline(
        text_pdf,
        path,
        [
            {"key": "alpha", "title": "Chapter Alpha", "page": 1},
            {"key": "a1", "title": "Alpha 1", "page": 2, "parent": "alpha"},
            {"key": "a2", "title": "Alpha 2", "page": 3, "parent": "alpha"},
            {"key": "beta", "title": "Chapter Beta", "page": 6},
            {"key": "b1", "title": "Beta 1", "page": 6, "parent": "beta"},
            {"key": "b2", "title": "Beta 2", "page": 6, "parent": "beta"},
            {"key": "b3", "title": "Beta 3", "page": 8, "parent": "beta"},
            {"key": "gamma", "title": "Chapter Gamma", "page": 10},
        ],
    )


def _build_duplicate_titles_pdf(path):
    """Two top-level chapters share a title, and both have a 'Child' entry."""
    writer = PdfWriter()
    for _ in range(10):
        writer.add_blank_page(width=300, height=400)
    first = writer.add_outline_item("Same Title", 0)
    writer.add_outline_item("Child", 1, parent=first)
    second = writer.add_outline_item("Same Title", 4)
    writer.add_outline_item("Child", 5, parent=second)
    with open(path, "wb") as f:
        writer.write(f)


@pytest.fixture
def outlined_pdf(tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    _build_outlined_pdf(pdf_path)
    return pdf_path


def _strategy(**kwargs):
    kwargs.setdefault("extract_images", False)
    return NaivePDFProcessorStrategy(**kwargs)


def _page_numbers(result):
    return [page.page_number for page in result.pages]


# ── range selection (serial + batch) ────────────────────────────────────


@pytest.mark.parametrize(
    "path, expected_pages",
    [
        (["Chapter Alpha"], [2, 3, 4, 5, 6]),       # ends before Beta (page 7)
        (["Chapter Alpha", "Alpha 1"], [3]),         # ends before Alpha 2
        (["Chapter Alpha", "Alpha 2"], [4, 5, 6]),   # ends before Beta
        (["Chapter Beta"], [7, 8, 9, 10]),           # ends before Gamma (page 11)
        (["Chapter Beta", "Beta 1"], [7]),           # boundary Beta 2 shares page
        (["Chapter Beta", "Beta 2"], [7, 8]),        # ends before Beta 3 (page 9)
        (["Chapter Beta", "Beta 3"], [9, 10]),       # ends before Gamma
        (["Chapter Gamma"], [11, 12]),               # no boundary: to the end
    ],
)
def test_bookmark_path_selects_exact_pages_serial(outlined_pdf, path, expected_pages):
    result = _strategy(bookmark_path=path).process(outlined_pdf)
    assert _page_numbers(result) == expected_pages


@pytest.mark.parametrize(
    "path, expected_pages",
    [
        (["Chapter Alpha"], [2, 3, 4, 5, 6]),
        (["Chapter Beta", "Beta 1"], [7]),
        (["Chapter Gamma"], [11, 12]),
    ],
)
def test_bookmark_path_selects_exact_pages_batched(outlined_pdf, path, expected_pages):
    result = _strategy(batch_size=3, bookmark_path=path).process_batch(outlined_pdf)
    assert _page_numbers(result) == expected_pages


def test_tuple_path_accepted(outlined_pdf):
    result = _strategy(bookmark_path=("Chapter Beta", "Beta 3")).process(outlined_pdf)
    assert _page_numbers(result) == [9, 10]


def test_serial_and_batch_agree(outlined_pdf):
    path = ["Chapter Alpha"]
    serial = _strategy(bookmark_path=path).process(outlined_pdf)
    batched = _strategy(batch_size=3, bookmark_path=path).process_batch(outlined_pdf)

    assert _page_numbers(serial) == _page_numbers(batched)
    assert [p.markdown for p in serial.pages] == [p.markdown for p in batched.pages]
    assert [p.html for p in serial.pages] == [p.html for p in batched.pages]


def test_pages_stay_in_document_order(outlined_pdf):
    result = _strategy(bookmark_path=["Chapter Beta"]).process(outlined_pdf)
    numbers = _page_numbers(result)
    assert numbers == sorted(numbers)
    # body content matches the page marker it rides on
    for page in result.pages:
        assert f"unique body of page {page.page_number}." in page.raw_text


# ── max_pages is counted from the chapter start ─────────────────────────


def test_max_pages_counts_from_chapter_start(outlined_pdf):
    result = _strategy(max_pages=2, bookmark_path=["Chapter Alpha"]).process(outlined_pdf)
    assert _page_numbers(result) == [2, 3]


def test_max_pages_batched_counts_from_chapter_start(outlined_pdf):
    result = _strategy(
        batch_size=2, max_pages=3, bookmark_path=["Chapter Gamma"]
    ).process_batch(outlined_pdf)
    # Gamma only has 2 pages, so the cap does not cut anything here
    assert _page_numbers(result) == [11, 12]

    result = _strategy(
        batch_size=2, max_pages=2, bookmark_path=["Chapter Beta"]
    ).process_batch(outlined_pdf)
    assert _page_numbers(result) == [7, 8]


# ── metadata still describes the whole PDF ──────────────────────────────


def test_metadata_reports_total_pdf_pages(outlined_pdf):
    result = _strategy(bookmark_path=["Chapter Beta", "Beta 2"]).process(outlined_pdf)
    assert result.metadata.pages == N_PAGES
    assert len(result.pages) == 2  # but only the chapter's pages were extracted


# ── images are only extracted for retained pages ────────────────────────


class _RecordingProcessor(NaivePDFProcessorStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_pages = []

    def _extract_images(self, page, image_dir, page_number):
        self.image_pages.append(page_number)
        return []


def test_images_only_processed_for_retained_pages_serial(outlined_pdf, tmp_path):
    processor = _RecordingProcessor(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=tmp_path / "imgs",
        bookmark_path=["Chapter Beta"],
    )
    processor.process(outlined_pdf)
    assert processor.image_pages == [7, 8, 9, 10]


def test_images_only_processed_for_retained_pages_batched(outlined_pdf, tmp_path):
    processor = _RecordingProcessor(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=tmp_path / "imgs",
        batch_size=3,
        bookmark_path=["Chapter Alpha", "Alpha 2"],
    )
    processor.process_batch(outlined_pdf)
    assert sorted(processor.image_pages) == [4, 5, 6]


# ── HTML keeps original page numbers (end-to-end via scraping strategy) ──


def test_scrap_html_keeps_original_page_numbers(outlined_pdf):
    strategy = PDFContentScrapingStrategy(
        extract_images=False,
        bookmark_path=["Chapter Beta", "Beta 3"],
    )
    result = strategy.scrap(outlined_pdf.as_uri(), "")

    data_pages = [int(n) for n in re.findall(r'data-page="(\d+)"', result.cleaned_html)]
    assert data_pages == [9, 10]
    assert result.metadata["pages"] == N_PAGES


def test_scrap_marks_images_with_original_page_numbers(outlined_pdf):
    strategy = PDFContentScrapingStrategy(
        extract_images=False,
        bookmark_path=["Chapter Alpha", "Alpha 1"],
    )
    result = strategy.scrap(f"file://{outlined_pdf}", "")
    assert 'data-page="3"' in result.cleaned_html
    assert 'data-page="2"' not in result.cleaned_html


# ── error conditions ────────────────────────────────────────────────────


def test_missing_top_level_title_raises(outlined_pdf):
    with pytest.raises(PDFBookmarkPathError, match="not found at level 1"):
        _strategy(bookmark_path=["No Such Chapter"]).process(outlined_pdf)


def test_missing_nested_title_raises(outlined_pdf):
    with pytest.raises(PDFBookmarkPathError, match="not found at level 2"):
        _strategy(
            bookmark_path=["Chapter Alpha", "No Such Section"]
        ).process(outlined_pdf)


def test_duplicate_top_level_title_is_ambiguous(tmp_path):
    pdf_path = tmp_path / "dup.pdf"
    _build_duplicate_titles_pdf(pdf_path)
    with pytest.raises(PDFBookmarkPathError, match="matches 2 bookmarks"):
        _strategy(bookmark_path=["Same Title"]).process(pdf_path)


def test_path_without_outline_raises(tmp_path):
    pdf_path = tmp_path / "plain.pdf"
    _write_text_pdf(pdf_path)
    with pytest.raises(PDFBookmarkPathError, match="no bookmarks"):
        _strategy(bookmark_path=["Anything"]).process(pdf_path)


def test_invalid_bookmark_page_and_bad_boundary():
    """Drive _resolve_page_range with a duck-typed reader for broken outlines."""

    class _Dest:
        def __init__(self, title, page_index):
            self.title = title
            self._page = page_index

    class _FakeReader:
        def __init__(self, outline, page_count):
            self.outline = outline
            self.pages = [object() for _ in range(page_count)]

        def get_destination_page_number(self, dest):
            if dest._page is None:
                raise RuntimeError("destination has no page")
            return dest._page

    # matched bookmark points past the end of the document
    reader = _FakeReader([_Dest("A", 99)], page_count=10)
    strategy = _strategy(bookmark_path=["A"])
    with pytest.raises(PDFBookmarkPathError, match="invalid page"):
        strategy._resolve_page_range(reader)

    # matched bookmark points to a negative index
    reader = _FakeReader([_Dest("A", -1)], page_count=10)
    with pytest.raises(PDFBookmarkPathError, match="invalid page"):
        strategy._resolve_page_range(reader)

    # destination cannot be resolved at all
    reader = _FakeReader([_Dest("A", None)], page_count=10)
    with pytest.raises(PDFBookmarkPathError, match="does not resolve to a page"):
        strategy._resolve_page_range(reader)

    # boundary bookmark points beyond the document
    reader = _FakeReader([_Dest("A", 1), _Dest("B", 50)], page_count=10)
    with pytest.raises(PDFBookmarkPathError, match="invalid page"):
        strategy._resolve_page_range(reader)

    # boundary precedes the chapter start
    reader = _FakeReader([_Dest("A", 5), _Dest("B", 2)], page_count=10)
    with pytest.raises(PDFBookmarkPathError, match="before the matched chapter start"):
        strategy._resolve_page_range(reader)


def test_invalid_path_arguments():
    # bare string must not be iterated into single-character levels
    with pytest.raises(TypeError, match="sequence of bookmark title"):
        NaivePDFProcessorStrategy(bookmark_path="Chapter Alpha")
    # non-string elements are rejected
    with pytest.raises(ValueError, match="non-empty string"):
        NaivePDFProcessorStrategy(bookmark_path=["Chapter Alpha", 3])
    with pytest.raises(ValueError, match="non-empty string"):
        NaivePDFProcessorStrategy(bookmark_path=[""])


# ── default behaviour is unchanged ──────────────────────────────────────


def test_no_path_processes_every_page_from_page_one(outlined_pdf):
    result = _strategy().process(outlined_pdf)
    assert _page_numbers(result) == list(range(1, N_PAGES + 1))


def test_empty_path_processes_every_page(outlined_pdf):
    result = _strategy(bookmark_path=[]).process_batch(outlined_pdf)
    assert _page_numbers(result) == list(range(1, N_PAGES + 1))


def test_default_max_pages_still_applies_without_path(outlined_pdf):
    result = _strategy(max_pages=3).process(outlined_pdf)
    assert _page_numbers(result) == [1, 2, 3]
