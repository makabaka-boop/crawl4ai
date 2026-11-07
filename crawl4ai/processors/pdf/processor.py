import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from time import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union, Sequence, Tuple
import base64
import tempfile
from .utils import *
from .utils import (
    apply_png_predictor,
    clean_pdf_text,
    clean_pdf_text_to_html,
)

# Remove direct pypdf imports from the top
# import pypdf
# from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFBookmarkPathError(ValueError):
    """Raised when a bookmark_path cannot select exactly one chapter.

    Covers the three failure modes callers can act on:
      * no bookmark matches a title at some level (path not found)
      * several bookmarks match a title at the same level (ambiguous)
      * a matched/boundary bookmark does not resolve to a valid page
    """

@dataclass
class PDFMetadata:
    title: Optional[str] = None
    author: Optional[str] = None
    producer: Optional[str] = None
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    pages: int = 0
    encrypted: bool = False
    file_size: Optional[int] = None

@dataclass
class PDFPage:
    page_number: int
    raw_text: str = ""
    markdown: str = ""
    html: str = ""
    images: List[Dict] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    layout: List[Dict] = field(default_factory=list)

@dataclass
class PDFProcessResult:
    metadata: PDFMetadata
    pages: List[PDFPage]
    processing_time: float = 0.0
    version: str = "1.0"

class PDFProcessorStrategy(ABC):
    @abstractmethod
    def process(self, pdf_path: Path) -> PDFProcessResult:
        pass

class NaivePDFProcessorStrategy(PDFProcessorStrategy):
    def __init__(self, image_dpi: int = 144, image_quality: int = 85, extract_images: bool = True,
                 save_images_locally: bool = False, image_save_dir: Optional[Path] = None, batch_size: int = 4,
                 max_pages: Optional[int] = None, bookmark_path: Optional[Sequence[str]] = None):
        # Import check at initialization time
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with 'pip install crawl4ai[pdf]'")
            
        self.image_dpi = image_dpi
        self.image_quality = image_quality
        self.extract_images = extract_images
        self.save_images_locally = save_images_locally
        self.image_save_dir = image_save_dir
        self.batch_size = batch_size
        # None means unbounded, which is the right default for a library caller
        # opening a file they chose. Callers accepting untrusted PDFs (the
        # Docker server) pass a cap so page count can't be used to burn CPU.
        self.max_pages = max_pages
        # Optional outline (bookmark) path, e.g. ["User Guide", "Networking",
        # "Wi-Fi Setup"]. Each entry must match a bookmark title exactly, at the
        # corresponding outline depth. When set, only the pages of the selected
        # chapter are processed (see _resolve_page_range). None/empty keeps the
        # original "process from page 1" behaviour.
        self.bookmark_path = self._validate_bookmark_path(bookmark_path)
        self._temp_dir = None

    @staticmethod
    def _validate_bookmark_path(bookmark_path: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
        if bookmark_path is None:
            return None
        # A bare str would otherwise be silently iterated into single chars.
        if isinstance(bookmark_path, str) or not isinstance(bookmark_path, (list, tuple)):
            raise TypeError(
                "bookmark_path must be a sequence of bookmark title strings, "
                f"got {type(bookmark_path).__name__}"
            )
        if len(bookmark_path) == 0:
            return None
        for level, title in enumerate(bookmark_path):
            if not isinstance(title, str) or not title:
                raise ValueError(
                    f"bookmark_path title at level {level + 1} must be a non-empty string"
                )
        return tuple(bookmark_path)

    def _page_limit(self, total_pages: int) -> int:
        """Number of pages to actually process, honouring max_pages."""
        if self.max_pages is None:
            return total_pages
        capped = min(total_pages, self.max_pages)
        if capped < total_pages:
            logger.warning(
                f"PDF has {total_pages} pages; processing first {capped} "
                f"(max_pages={self.max_pages})"
            )
        return capped

    # ── bookmark (outline) based chapter selection ───────────────────────

    @staticmethod
    def _flatten_outline(reader) -> List[Tuple[Any, int]]:
        """Flatten the PDF outline into ``(bookmark, depth)`` tuples.

        pypdf exposes ``reader.outline`` as a nested list: a bookmark is a
        ``Destination`` and the list immediately following it holds its
        children. Flattening in document order turns that nesting into depths.
        """
        flat: List[Tuple[Any, int]] = []

        def walk(items, depth):
            for item in items:
                if isinstance(item, list):
                    walk(item, depth + 1)
                elif getattr(item, "title", None) is not None:
                    flat.append((item, depth))

        outline = getattr(reader, "outline", None)
        if outline:
            walk(outline, 0)
        return flat

    def _match_bookmark_path(
        self, flat_outline: List[Tuple[Any, int]]
    ) -> int:
        """Find the unique bookmark matching ``self.bookmark_path``.

        Each path entry is matched against a bookmark title exactly (string
        equality) at the depth of its level, restricted to the subtree of the
        previously matched parent. Returns the index in ``flat_outline``.
        """
        path = self.bookmark_path
        search_start = 0
        matched_index = -1

        for level, wanted_title in enumerate(path):
            candidates = []
            i = search_start
            while i < len(flat_outline):
                bookmark, depth = flat_outline[i]
                # A node shallower than the level we are matching closes the
                # parent subtree (depth can never be negative at level 0).
                if depth < level:
                    break
                if depth == level and bookmark.title == wanted_title:
                    candidates.append(i)
                i += 1

            breadcrumb = " > ".join(path[:level]) or "<outline root>"
            if not candidates:
                raise PDFBookmarkPathError(
                    f"Bookmark path not found at level {level + 1}: no bookmark "
                    f"titled {wanted_title!r} under {breadcrumb!r}"
                )
            if len(candidates) > 1:
                raise PDFBookmarkPathError(
                    f"Bookmark path matches {len(candidates)} bookmarks at level "
                    f"{level + 1} for title {wanted_title!r} under {breadcrumb!r}; "
                    "refusing to pick one ambiguously"
                )

            matched_index = candidates[0]
            search_start = matched_index + 1

        return matched_index

    def _bookmark_page_number(
        self, reader, bookmark, total_pages: int, *, role: str
    ) -> int:
        """Resolve a bookmark to a 0-based page index, validating the result."""
        title = getattr(bookmark, "title", "<untitled>")
        try:
            page_index = reader.get_destination_page_number(bookmark)
        except Exception as e:
            raise PDFBookmarkPathError(
                f"Bookmark {title!r} ({role}) does not resolve to a page: {e}"
            ) from e

        if not isinstance(page_index, int) or not (0 <= page_index < total_pages):
            raise PDFBookmarkPathError(
                f"Bookmark {title!r} ({role}) points to invalid page "
                f"{page_index!r}; document has {total_pages} pages"
            )
        return page_index

    def _resolve_page_range(self, reader) -> Optional[Tuple[int, int]]:
        """Return ``(start_index, end_index_exclusive)`` for bookmark_path.

        Returns ``None`` when no bookmark path was configured (original
        behaviour: process from the first page).

        The chapter starts on the page of the matched bookmark and runs to the
        page before the first later bookmark at the same or a higher (shallower)
        outline level. If that boundary bookmark shares the start page, the
        shared page is kept. With no boundary the chapter runs to the end.
        ``max_pages`` is counted from the chapter start page.
        """
        if not self.bookmark_path:
            return None

        total_pages = len(reader.pages)
        flat_outline = self._flatten_outline(reader)
        if not flat_outline:
            raise PDFBookmarkPathError(
                "PDF has no bookmarks/outline; cannot resolve bookmark_path "
                f"{list(self.bookmark_path)!r}"
            )

        matched_index = self._match_bookmark_path(flat_outline)
        matched_bookmark, matched_depth = flat_outline[matched_index]
        start_index = self._bookmark_page_number(
            reader, matched_bookmark, total_pages, role="chapter start"
        )

        # First later bookmark at the same level or higher (shallower) ends it.
        boundary_bookmark = None
        for i in range(matched_index + 1, len(flat_outline)):
            if flat_outline[i][1] <= matched_depth:
                boundary_bookmark = flat_outline[i][0]
                break

        if boundary_bookmark is None:
            end_index = total_pages
        else:
            boundary_page = self._bookmark_page_number(
                reader, boundary_bookmark, total_pages, role="chapter boundary"
            )
            if boundary_page < start_index:
                raise PDFBookmarkPathError(
                    f"Chapter boundary bookmark {boundary_bookmark.title!r} is on "
                    f"page {boundary_page + 1}, before the matched chapter start "
                    f"page {start_index + 1}"
                )
            # boundary_page is exclusive; when both bookmarks share a page keep
            # that page instead of producing an empty chapter.
            end_index = max(start_index + 1, boundary_page)

        # max_pages is counted from the chapter start page and only retained
        # pages are ever processed.
        retained = end_index - start_index
        if self.max_pages is not None and retained > self.max_pages:
            logger.warning(
                f"Bookmarked chapter spans {retained} pages starting at page "
                f"{start_index + 1}; processing first {self.max_pages} "
                f"(max_pages={self.max_pages})"
            )
            end_index = start_index + self.max_pages

        return start_index, end_index

    def process(self, pdf_path: Path) -> PDFProcessResult:
        # Import inside method to allow dependency to be optional
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with 'pip install crawl4ai[pdf]'")
            
        start_time = time()
        result = PDFProcessResult(
            metadata=PDFMetadata(),
            pages=[],
            version="1.1"
        )

        try:
            with pdf_path.open('rb') as file:
                reader = PdfReader(file)
                # Metadata is always extracted from the whole document, so it
                # keeps reporting the full page count even when a single
                # chapter is processed.
                result.metadata = self._extract_metadata(pdf_path, reader)
                
                # Handle image directory
                image_dir = None
                if self.extract_images and self.save_images_locally:
                    if self.image_save_dir:
                        image_dir = Path(self.image_save_dir)
                        image_dir.mkdir(exist_ok=True, parents=True)
                    else:
                        self._temp_dir = tempfile.mkdtemp(prefix='pdf_images_')
                        image_dir = Path(self._temp_dir)

                # None bookmark path -> process from page 1 as before.
                page_range = self._resolve_page_range(reader)
                if page_range is None:
                    start_index, end_index = 0, self._page_limit(len(reader.pages))
                else:
                    start_index, end_index = page_range

                for page_num in range(start_index, end_index):
                    page = reader.pages[page_num]
                    # Keep the original (document-wide) 1-based page number so
                    # HTML page markers and per-page formatting stay accurate.
                    pdf_page = self._process_page(page, image_dir, page_num + 1)
                    result.pages.append(pdf_page)

        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise
        finally:
            # Cleanup temp directory if it was created
            if self._temp_dir and not self.image_save_dir:
                import shutil
                try:
                    shutil.rmtree(self._temp_dir)
                except Exception as e:
                    logger.error(f"Failed to cleanup temp directory: {str(e)}")

        result.processing_time = time() - start_time
        return result

    def process_batch(self, pdf_path: Path) -> PDFProcessResult:
        """Like process() but processes PDF pages in parallel batches"""
        # Import inside method to allow dependency to be optional
        try:
            from pypdf import PdfReader
            import pypdf  # For type checking
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with 'pip install crawl4ai[pdf]'")
            
        import concurrent.futures
        import threading
        
        # Initialize pypdf thread support
        if not hasattr(threading.current_thread(), "_children"): 
            threading.current_thread()._children = set()
        
        start_time = time()
        result = PDFProcessResult(
            metadata=PDFMetadata(),
            pages=[],
            version="1.1" 
        )

        try:
            # Get metadata and page count from main thread
            with pdf_path.open('rb') as file:
                reader = PdfReader(file)
                # Metadata always reflects the whole document, even when a
                # bookmark path restricts which pages are processed.
                result.metadata = self._extract_metadata(pdf_path, reader)
                page_range = self._resolve_page_range(reader)
                if page_range is None:
                    start_index, end_index = 0, self._page_limit(len(reader.pages))
                else:
                    start_index, end_index = page_range

            page_indices = list(range(start_index, end_index))

            # Handle image directory setup
            image_dir = None
            if self.extract_images and self.save_images_locally:
                if self.image_save_dir:
                    image_dir = Path(self.image_save_dir)
                    image_dir.mkdir(exist_ok=True, parents=True)
                else:
                    self._temp_dir = tempfile.mkdtemp(prefix='pdf_images_')
                    image_dir = Path(self._temp_dir)

            def process_page_safely(page_num: int):
                # Each thread opens its own file handle
                with pdf_path.open('rb') as file:
                    thread_reader = PdfReader(file)
                    page = thread_reader.pages[page_num]
                    # Document-wide 1-based page number, preserved even when a
                    # bookmark path skips earlier pages.
                    return self._process_page(page, image_dir, page_num + 1)

            # Process pages in parallel batches
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                futures = []
                for page_num in page_indices:
                    future = executor.submit(process_page_safely, page_num)
                    futures.append((page_num, future))

                # Collect results in document order (absolute page index order)
                result.pages = [None] * len(page_indices)
                for position, (page_num, future) in enumerate(futures):
                    try:
                        pdf_page = future.result()
                        result.pages[position] = pdf_page
                    except Exception as e:
                        logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
                        raise

        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise
        finally:
            # Cleanup temp directory if it was created
            if self._temp_dir and not self.image_save_dir:
                import shutil
                try:
                    shutil.rmtree(self._temp_dir)
                except Exception as e:
                    logger.error(f"Failed to cleanup temp directory: {str(e)}")

        result.processing_time = time() - start_time
        return result

    def _process_page(self, page, image_dir: Optional[Path], page_number: int) -> PDFPage:
        pdf_page = PDFPage(
            page_number=page_number,
        )

        # Text and font extraction
        def visitor_text(text, cm, tm, font_dict, font_size):
            pdf_page.raw_text += text
            pdf_page.layout.append({
                "type": "text",
                "text": text,
                "x": tm[4],
                "y": tm[5],
            })
        
        page.extract_text(visitor_text=visitor_text)

        # Image extraction
        if self.extract_images:
            pdf_page.images = self._extract_images(page, image_dir, page_number)

        # Link extraction
        pdf_page.links = self._extract_links(page)
        
        # Add markdown content
        pdf_page.markdown = clean_pdf_text(page_number, pdf_page.raw_text)
        pdf_page.html = clean_pdf_text_to_html(page_number, pdf_page.raw_text)

        return pdf_page

    def _extract_images(self, page, image_dir: Optional[Path], page_number: int) -> List[Dict]:
        # Import pypdf for type checking only when needed
        try:
            from pypdf.generic import IndirectObject
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with 'pip install crawl4ai[pdf]'")
            
        if not self.extract_images:
            return []

        images = []
        try:
            resources = page.get("/Resources")
            if resources:  # Check if resources exist
                resources = resources.get_object()  # Resolve IndirectObject
                if '/XObject' in resources:
                    xobjects = resources['/XObject'].get_object()
                    img_count = 0
                    for obj_name in xobjects:
                        xobj = xobjects[obj_name]
                        if hasattr(xobj, 'get_object') and callable(xobj.get_object):
                            xobj = xobj.get_object()
                            if xobj.get('/Subtype') == '/Image':
                                try:
                                    img_count += 1
                                    img_filename = f"page_{page_number}_img_{img_count}"
                                    data = xobj.get_data()
                                    filters = xobj.get('/Filter', [])
                                    if not isinstance(filters, list):
                                        filters = [filters]

                                    # Resolve IndirectObjects in properties
                                    width = xobj.get('/Width', 0)
                                    height = xobj.get('/Height', 0)
                                    color_space = xobj.get('/ColorSpace', '/DeviceRGB')
                                    if isinstance(color_space, IndirectObject):
                                        color_space = color_space.get_object()

                                    # Handle different image encodings
                                    success = False
                                    image_format = 'bin'
                                    image_data = None
                                    
                                    if '/FlateDecode' in filters:
                                        try:
                                            decode_parms = xobj.get('/DecodeParms', {})
                                            if isinstance(decode_parms, IndirectObject):
                                                decode_parms = decode_parms.get_object()
                                            
                                            predictor = decode_parms.get('/Predictor', 1)
                                            bits = xobj.get('/BitsPerComponent', 8)
                                            colors = 3 if color_space == '/DeviceRGB' else 1

                                            if predictor >= 10:
                                                data = apply_png_predictor(data, width, bits, colors)

                                            # Create PIL Image
                                            from PIL import Image
                                            mode = 'RGB' if color_space == '/DeviceRGB' else 'L'
                                            img = Image.frombytes(mode, (width, height), data)
                                            
                                            if self.save_images_locally:
                                                final_path = (image_dir / img_filename).with_suffix('.png')
                                                img.save(final_path)
                                                image_data = str(final_path)
                                            else:
                                                import io
                                                img_byte_arr = io.BytesIO()
                                                img.save(img_byte_arr, format='PNG')
                                                image_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                                            
                                            success = True
                                            image_format = 'png'
                                        except Exception as e:
                                            logger.error(f"FlateDecode error: {str(e)}")

                                    elif '/DCTDecode' in filters:
                                        # JPEG image
                                        try:
                                            if self.save_images_locally:
                                                final_path = (image_dir / img_filename).with_suffix('.jpg')
                                                with open(final_path, 'wb') as f:
                                                    f.write(data)
                                                image_data = str(final_path)
                                            else:
                                                image_data = base64.b64encode(data).decode('utf-8')
                                            success = True
                                            image_format = 'jpeg'
                                        except Exception as e:
                                            logger.error(f"JPEG save error: {str(e)}")

                                    elif '/CCITTFaxDecode' in filters:
                                        try:
                                            if data[:4] != b'II*\x00':
                                                # Add TIFF header if missing
                                                tiff_header = b'II*\x00\x08\x00\x00\x00\x0e\x00\x00\x01\x03\x00\x01\x00\x00\x00' + \
                                                            width.to_bytes(4, 'little') + \
                                                            b'\x01\x03\x00\x01\x00\x00\x00' + \
                                                            height.to_bytes(4, 'little') + \
                                                            b'\x01\x12\x00\x03\x00\x00\x00\x01\x00\x01\x00\x00\x01\x17\x00\x04\x00\x00\x00\x01\x00\x00\x00J\x01\x1B\x00\x05\x00\x00\x00\x01\x00\x00\x00R\x01\x28\x00\x03\x00\x00\x00\x01\x00\x02\x00\x00'
                                                data = tiff_header + data
                                            
                                            if self.save_images_locally:
                                                final_path = (image_dir / img_filename).with_suffix('.tiff')
                                                with open(final_path, 'wb') as f:
                                                    f.write(data)
                                                image_data = str(final_path)
                                            else:
                                                image_data = base64.b64encode(data).decode('utf-8')
                                            success = True
                                            image_format = 'tiff'
                                        except Exception as e:
                                            logger.error(f"CCITT save error: {str(e)}")

                                    elif '/JPXDecode' in filters:
                                        # JPEG 2000
                                        try:
                                            if self.save_images_locally:
                                                final_path = (image_dir / img_filename).with_suffix('.jp2')
                                                with open(final_path, 'wb') as f:
                                                    f.write(data)
                                                image_data = str(final_path)
                                            else:
                                                image_data = base64.b64encode(data).decode('utf-8')
                                            success = True
                                            image_format = 'jpeg2000'
                                        except Exception as e:
                                            logger.error(f"JPEG2000 save error: {str(e)}")

                                    if success and image_data:
                                        image_info = {
                                            "format": image_format,
                                            "width": width,
                                            "height": height,
                                            "color_space": str(color_space),
                                            "bits_per_component": xobj.get('/BitsPerComponent', 1)
                                        }
                                        
                                        if self.save_images_locally:
                                            image_info["path"] = image_data
                                        else:
                                            image_info["data"] = image_data
                                            
                                        images.append(image_info)
                                    else:
                                        # Fallback: Save raw data
                                        if self.save_images_locally:
                                            final_path = (image_dir / img_filename).with_suffix('.bin')
                                            with open(final_path, 'wb') as f:
                                                f.write(data)
                                            logger.warning(f"Saved raw image data to {final_path}")
                                        else:
                                            image_data = base64.b64encode(data).decode('utf-8')
                                            images.append({
                                                "format": "bin",
                                                "width": width,
                                                "height": height,
                                                "color_space": str(color_space),
                                                "bits_per_component": xobj.get('/BitsPerComponent', 1),
                                                "data": image_data
                                            })

                                except Exception as e:
                                    logger.error(f"Error processing image: {str(e)}")
        except Exception as e:
            logger.error(f"Image extraction error: {str(e)}")
        
        return images

    def _extract_links(self, page) -> List[str]:
        links = []
        if '/Annots' in page:
            try:
                for annot in page['/Annots']:
                    a = annot.get_object()
                    if '/A' in a and '/URI' in a['/A']:
                        links.append(a['/A']['/URI'])
            except Exception as e:
                print(f"Link error: {str(e)}")
        return links

    def _extract_metadata(self, pdf_path: Path, reader = None) -> PDFMetadata:
        # Import inside method to allow dependency to be optional 
        if reader is None:
            try:
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
            except ImportError:
                raise ImportError("pypdf is required for PDF processing. Install with 'pip install crawl4ai[pdf]'")

        meta = reader.metadata or {}
        created = self._parse_pdf_date(meta.get('/CreationDate', ''))
        modified = self._parse_pdf_date(meta.get('/ModDate', ''))
        
        return PDFMetadata(
            title=meta.get('/Title'),
            author=meta.get('/Author'),
            producer=meta.get('/Producer'),
            created=created,
            modified=modified,
            pages=len(reader.pages),
            encrypted=reader.is_encrypted,
            file_size=pdf_path.stat().st_size
        )

    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        try:
            match = re.match(r'D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', date_str)
            if not match:
                return None
                
            return datetime(
                year=int(match[1]),
                month=int(match[2]),
                day=int(match[3]),
                hour=int(match[4]),
                minute=int(match[5]),
                second=int(match[6])
            )
        except:
            return None

# Usage example
if __name__ == "__main__":
    import json
    from pathlib import Path
    
    try:
        # Import pypdf only when running the file directly
        import pypdf
        from pypdf import PdfReader
    except ImportError:
        print("pypdf is required for PDF processing. Install with 'pip install crawl4ai[pdf]'")
        exit(1)
        
    current_dir = Path(__file__).resolve().parent
    pdf_path = f'{current_dir}/test.pdf'
    
    strategy = NaivePDFProcessorStrategy()
    result = strategy.process(Path(pdf_path))
    
    # Convert to JSON
    json_output = asdict(result)
    print(json.dumps(json_output, indent=2, default=str))
    
    with open(f'{current_dir}/test.html', 'w') as f:
        for page in result.pages:
            f.write(f'<h1>Page {page["page_number"]}</h1>')
            f.write(page['html'])
    with open(f'{current_dir}/test.md', 'w') as f:
        for page in result.pages:
            f.write(f'# Page {page["page_number"]}\n\n')
            f.write(clean_pdf_text(page["page_number"], page['raw_text']))
            f.write('\n\n')
