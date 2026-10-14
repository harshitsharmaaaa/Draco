import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

logger = logging.getLogger(__name__)

PageType = Literal["digital", "scanned", "ambiguous"]


class PdfDetectionError(Exception):
    """The PDF could not be inspected (empty, malformed, or encrypted)."""


@dataclass(frozen=True)
class DetectorConfig:
    # Non-whitespace characters at or above which a page has a usable text layer.
    min_digital_chars: int = 100
    # Non-whitespace characters at or below which a page is treated as having no text.
    max_scanned_chars: int = 10
    # An image with at least this many pixels (width * height) is treated as
    # "large": it could hold page-sized content such as a scanned table.
    large_image_pixels: int = 250_000
    # Inspect at most this many pages, evenly spaced. None inspects every page,
    # which page-level routing needs (about 5 ms per page).
    max_pages: int | None = None

    def __post_init__(self):
        if self.max_pages is not None and self.max_pages < 1:
            raise ValueError("DetectorConfig max_pages must be None or >= 1.")
        if self.min_digital_chars < 0 or self.max_scanned_chars < 0:
            raise ValueError("DetectorConfig char thresholds must be >= 0.")
        if self.min_digital_chars <= self.max_scanned_chars:
            raise ValueError(
                "DetectorConfig min_digital_chars must exceed max_scanned_chars."
            )
        if self.large_image_pixels < 1:
            raise ValueError("DetectorConfig large_image_pixels must be >= 1.")


@dataclass(frozen=True)
class PageResult:
    index: int  # zero-based page index in the document
    page_type: PageType
    char_count: int
    image_count: int
    large_image_count: int
    reason: str


@dataclass(frozen=True)
class PdfInspection:
    page_count: int
    pages: list[PageResult] = field(default_factory=list)

    @property
    def sampled(self) -> bool:
        return len(self.pages) < self.page_count

    def count(self, page_type: PageType) -> int:
        return sum(1 for p in self.pages if p.page_type == page_type)

    @property
    def digital_ratio(self) -> float:
        if not self.pages:
            return 0.0
        return self.count("digital") / len(self.pages)

    @property
    def all_digital(self) -> bool:
        return bool(self.pages) and self.count("digital") == len(self.pages)


def _sample_indices(page_count: int, max_pages: int | None) -> list[int]:
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be None or >= 1.")
    if max_pages is None or page_count <= max_pages:
        return list(range(page_count))
    # Evenly spaced, so late pages are covered as well as the first ones.
    return sorted({i * page_count // max_pages for i in range(max_pages)})


def _image_stats(page, large_image_pixels: int) -> tuple[int, int]:
    """Count images from the page's XObject dictionary without decoding them.

    Only top-level image XObjects are seen; images nested in form XObjects or
    inline images are not.
    """
    resources = page.get("/Resources")
    xobjects = resources.get_object().get("/XObject") if resources else None
    if not xobjects:
        return 0, 0

    total = large = 0
    for ref in xobjects.get_object().values():
        xobj = ref.get_object()
        if xobj.get("/Subtype") != "/Image":
            continue
        total += 1
        if xobj.get("/Width", 0) * xobj.get("/Height", 0) >= large_image_pixels:
            large += 1
    return total, large


def _classify_page(index: int, page, config: DetectorConfig) -> PageResult:
    try:
        text = page.extract_text() or ""
        # Whitespace, control chars and U+FFFD do not count as usable text.
        chars = sum(1 for c in text if not c.isspace() and c.isprintable() and c != "�")
        images, large_images = _image_stats(page, config.large_image_pixels)
    except Exception as exc:
        return PageResult(index, "ambiguous", 0, 0, 0, f"inspection failed: {type(exc).__name__}")

    def result(page_type: PageType, reason: str) -> PageResult:
        return PageResult(index, page_type, chars, images, large_images, reason)

    if chars >= config.min_digital_chars:
        if large_images:
            return result("ambiguous", "text layer plus large image; image may hold extra text")
        return result("digital", "usable text layer")

    if chars <= config.max_scanned_chars:
        if images:
            return result("scanned", "no usable text layer and page has images")
        return result("ambiguous", "no text and no images; may be blank or vector-drawn text")

    return result("ambiguous", "text layer too small to be conclusive")


def inspect_pdf(
    pdf_path: str | Path, config: DetectorConfig | None = None
) -> PdfInspection:
    """Classify each inspected page as digital, scanned, or ambiguous.

    Heuristic only: a text layer proves that some text is native, not that the
    whole page is. No rendering or OCR is performed.

    Raises PdfDetectionError if the PDF is empty, malformed, or encrypted with
    a non-empty password.
    """
    config = config or DetectorConfig()

    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            if reader.is_encrypted and not reader.decrypt(""):
                raise PdfDetectionError("PDF is encrypted and requires a password")
            page_count = len(reader.pages)
            if page_count == 0:
                raise PdfDetectionError("PDF contains no pages")
            indices = _sample_indices(page_count, config.max_pages)
            pages = [_classify_page(i, reader.pages[i], config) for i in indices]
    except PdfDetectionError:
        raise
    except Exception as exc:
        raise PdfDetectionError(f"could not read PDF: {type(exc).__name__}: {exc}") from exc

    inspection = PdfInspection(page_count=page_count, pages=pages)
    logger.info(
        "Inspected %d of %d pages: digital=%d scanned=%d ambiguous=%d",
        len(pages),
        page_count,
        inspection.count("digital"),
        inspection.count("scanned"),
        inspection.count("ambiguous"),
    )
    return inspection
