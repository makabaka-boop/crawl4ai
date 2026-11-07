"""
Tests for HTTP strategy file download detection and handling.

Tests the Content-Type/Content-Disposition detection logic in AsyncHTTPCrawlerStrategy
that saves non-HTML responses to disk and populates downloaded_files.
"""

import os
import sys
import asyncio
import gzip
import tempfile
import shutil
import json
import socket

import pytest
from aiohttp import web

# Add parent to path so crawl4ai is importable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from crawl4ai.async_crawler_strategy import (
    AsyncHTTPCrawlerStrategy,
    DownloadSizeLimitError,
    HTTPCrawlerError,
)
from crawl4ai.async_configs import HTTPCrawlerConfig, CrawlerRunConfig


# ---------------------------------------------------------------------------
# Test HTTP server
# ---------------------------------------------------------------------------

async def handle_html(request):
    return web.Response(text="<html><body>Hello</body></html>", content_type="text/html")

async def handle_csv(request):
    csv_data = "id,name,value\n1,alpha,100\n2,beta,200\n3,gamma,300\n"
    return web.Response(
        text=csv_data,
        content_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="data.csv"'},
    )

async def handle_csv_no_disposition(request):
    return web.Response(text="col1,col2\na,b\nc,d\n", content_type="text/csv")

async def handle_json(request):
    return web.Response(
        text=json.dumps({"key": "value", "items": [1, 2, 3]}),
        content_type="application/json",
    )

async def handle_pdf(request):
    pdf_bytes = b"%PDF-1.4 fake pdf content " + (b"\x00\xff" * 500)
    return web.Response(
        body=pdf_bytes,
        content_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="report.pdf"'},
    )

async def handle_binary_no_name(request):
    return web.Response(body=b"\x89PNG\r\n" + b"\x00" * 100, content_type="image/png")

async def handle_plain_text(request):
    return web.Response(text="Just plain text content.", content_type="text/plain")

async def handle_xml(request):
    return web.Response(
        text='<?xml version="1.0"?><root><item>test</item></root>',
        content_type="application/xml",
    )

async def handle_attachment_html(request):
    return web.Response(
        text="<html><body>download me</body></html>",
        content_type="text/html",
        headers={"Content-Disposition": 'attachment; filename="page.html"'},
    )

async def handle_csv_url_filename(request):
    return web.Response(text="x,y\n1,2\n", content_type="text/csv")


# --- Payloads for the size-cap / streaming tests ---------------------------

BIG_PDF_TOTAL = 5 * 1024 * 1024  # 5 MiB
_BIG_PDF_CHUNK = 64 * 1024
GZIP_PAYLOAD = bytes(range(256)) * 200  # 51,200 decompressed bytes, repetitive
GZIP_COMPRESSED = gzip.compress(GZIP_PAYLOAD, compresslevel=9)


async def handle_big_pdf(request):
    """Stream a large PDF with chunked transfer-encoding.

    Records how many bytes it managed to push before the client went away so
    tests can prove an over-cap client stops pulling the body.
    """
    state = request.app["state"]
    resp = web.StreamResponse(
        headers={
            "Content-Type": "application/pdf",
            "Content-Disposition": 'attachment; filename="big.pdf"',
        }
    )
    await resp.prepare(request)
    written = 0
    body = b"%PDF-1.4\n" + b"\x00" * (_BIG_PDF_CHUNK - 8)
    try:
        while written < BIG_PDF_TOTAL:
            n = min(len(body), BIG_PDF_TOTAL - written)
            await resp.write(body[:n])
            written += n
            state["big_pdf_sent"] = written
    except (ConnectionResetError, BrokenPipeError, RuntimeError):
        # Client aborted after hitting the size cap — expected.
        state["big_pdf_aborted"] = True
    return resp


async def handle_gzip_binary(request):
    """Pre-compressed binary: wire size is tiny, decompressed size is 51 KiB."""
    return web.Response(
        body=GZIP_COMPRESSED,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Disposition": 'attachment; filename="data.bin"',
            "Content-Encoding": "gzip",
        },
    )


async def handle_truncated_binary(request):
    """Advertise 1 MiB but sever the connection after a single 4 KiB write."""
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Disposition": 'attachment; filename="truncated.bin"',
            "Content-Length": str(1024 * 1024),
        },
    )
    await resp.prepare(request)
    await resp.write(b"X" * 4096)
    await asyncio.sleep(0.05)
    request.transport.close()
    return resp


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _TestServer:
    """Minimal test server lifecycle manager."""

    def __init__(self):
        self.port = _find_free_port()
        self.runner = None
        self.state = {"big_pdf_sent": 0, "big_pdf_aborted": False}

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def url(self, path):
        return f"{self.base_url}{path}"

    async def start(self):
        app = web.Application()
        app["state"] = self.state
        app.router.add_get("/page.html", handle_html)
        app.router.add_get("/data.csv", handle_csv)
        app.router.add_get("/inline.csv", handle_csv_no_disposition)
        app.router.add_get("/api/data.json", handle_json)
        app.router.add_get("/report.pdf", handle_pdf)
        app.router.add_get("/image", handle_binary_no_name)
        app.router.add_get("/readme.txt", handle_plain_text)
        app.router.add_get("/feed.xml", handle_xml)
        app.router.add_get("/attachment.html", handle_attachment_html)
        app.router.add_get("/files/export.csv", handle_csv_url_filename)
        app.router.add_get("/big.pdf", handle_big_pdf)
        app.router.add_get("/gzip.bin", handle_gzip_binary)
        app.router.add_get("/truncated.bin", handle_truncated_binary)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await site.start()

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()


# ---------------------------------------------------------------------------
# Helper to run an async crawl test
# ---------------------------------------------------------------------------

async def _crawl(server, path, downloads_dir=None):
    config = HTTPCrawlerConfig(downloads_path=downloads_dir)
    strategy = AsyncHTTPCrawlerStrategy(browser_config=config)
    return await strategy.crawl(server.url(path))


async def _crawl_with_cap(server, path, downloads_dir, max_file_size):
    config = HTTPCrawlerConfig(
        downloads_path=downloads_dir, max_file_size=max_file_size
    )
    strategy = AsyncHTTPCrawlerStrategy(browser_config=config)
    return await strategy.crawl(server.url(path))


def _part_files(dl_dir):
    return [f for f in os.listdir(dl_dir) if f.endswith(".part")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHTMLPassthrough:
    """Normal HTML responses should behave exactly as before."""

    def test_html_response_unchanged(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/page.html", dl_dir)

                assert "<html>" in result.html
                assert result.downloaded_files is None
                assert result.status_code == 200
                assert len(os.listdir(dl_dir)) == 0
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())


class TestTextFileDownloads:
    """Text-based file downloads (CSV, JSON, XML, plain text)."""

    def test_csv_with_disposition(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/data.csv", dl_dir)

                assert result.downloaded_files is not None
                assert len(result.downloaded_files) == 1
                filepath = result.downloaded_files[0]
                assert filepath.endswith("data.csv")
                assert os.path.isfile(filepath)
                assert "alpha" in result.html
                assert "id,name,value" in result.html
                with open(filepath) as f:
                    assert "alpha" in f.read()
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_csv_without_disposition(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/inline.csv", dl_dir)

                assert result.downloaded_files is not None
                assert len(result.downloaded_files) == 1
                assert result.downloaded_files[0].endswith("inline.csv")
                assert "col1,col2" in result.html
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_json_download(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/api/data.json", dl_dir)

                assert result.downloaded_files is not None
                filepath = result.downloaded_files[0]
                assert filepath.endswith("data.json")
                assert '"key"' in result.html
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_plain_text(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/readme.txt", dl_dir)

                assert result.downloaded_files is not None
                assert "Just plain text content." in result.html
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_xml_download(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/feed.xml", dl_dir)

                assert result.downloaded_files is not None
                assert "<root>" in result.html
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_csv_filename_from_url(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/files/export.csv", dl_dir)

                assert result.downloaded_files is not None
                assert result.downloaded_files[0].endswith("export.csv")
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())


class TestBinaryFileDownloads:
    """Binary file downloads (PDF, images) — html should be empty."""

    def test_pdf_download(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/report.pdf", dl_dir)

                assert result.downloaded_files is not None
                filepath = result.downloaded_files[0]
                assert filepath.endswith("report.pdf")
                assert os.path.isfile(filepath)
                assert result.html == ""
                with open(filepath, "rb") as f:
                    data = f.read()
                    assert data.startswith(b"%PDF")
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_binary_no_filename(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/image", dl_dir)

                assert result.downloaded_files is not None
                filepath = result.downloaded_files[0]
                assert filepath.endswith(".png")
                assert os.path.isfile(filepath)
                assert result.html == ""
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())


class TestEdgeCases:
    """Edge cases and backward compatibility."""

    def test_attachment_html_treated_as_download(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/attachment.html", dl_dir)

                assert result.downloaded_files is not None
                assert result.downloaded_files[0].endswith("page.html")
                assert "download me" in result.html
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_default_downloads_path(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                config = HTTPCrawlerConfig()  # no downloads_path
                strategy = AsyncHTTPCrawlerStrategy(browser_config=config)
                result = await strategy.crawl(srv.url("/data.csv"))

                assert result.downloaded_files is not None
                filepath = result.downloaded_files[0]
                assert ".crawl4ai/downloads" in filepath
                if os.path.isfile(filepath):
                    os.unlink(filepath)
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_response_headers_contain_content_type(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/data.csv", dl_dir)
                assert "text/csv" in result.response_headers.get("Content-Type", "")
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_status_code_preserved(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/report.pdf", dl_dir)
                assert result.status_code == 200
            finally:
                await srv.stop()

        asyncio.get_event_loop().run_until_complete(_test())


class TestDetectionHelpers:
    """Unit tests for the detection helper methods."""

    def test_is_file_download(self):
        s = AsyncHTTPCrawlerStrategy()
        assert s._is_file_download("text/csv", "") is True
        assert s._is_file_download("application/pdf", "") is True
        assert s._is_file_download("image/png", "") is True
        assert s._is_file_download("text/html", "") is False
        assert s._is_file_download("text/html", "attachment; filename=x") is True
        assert s._is_file_download("", "") is False

    def test_is_text_content(self):
        s = AsyncHTTPCrawlerStrategy()
        assert s._is_text_content("text/csv") is True
        assert s._is_text_content("text/plain") is True
        assert s._is_text_content("application/json") is True
        assert s._is_text_content("application/pdf") is False
        assert s._is_text_content("image/png") is False
        assert s._is_text_content("text/tab-separated-values") is True

    def test_extract_filename_from_disposition(self):
        s = AsyncHTTPCrawlerStrategy()
        assert s._extract_filename('attachment; filename="data.csv"', "http://x/y", "text/csv") == "data.csv"
        assert s._extract_filename("attachment; filename=report.pdf", "http://x/y", "application/pdf") == "report.pdf"

    def test_extract_filename_from_url(self):
        s = AsyncHTTPCrawlerStrategy()
        assert s._extract_filename("", "http://example.com/files/export.csv", "text/csv") == "export.csv"

    def test_extract_filename_fallback(self):
        s = AsyncHTTPCrawlerStrategy()
        name = s._extract_filename("", "http://example.com/download", "application/pdf")
        assert name.startswith("download_")
        assert name.endswith(".pdf")


class TestMaxFileSizeConfig:
    """Validation and plumbing of the HTTPCrawlerConfig knob."""

    def test_defaults_to_none(self):
        assert HTTPCrawlerConfig().max_file_size is None

    @pytest.mark.parametrize("bad_value", [0, -1, 1.5, "1024", True])
    def test_rejects_invalid_values(self, bad_value):
        with pytest.raises(ValueError):
            HTTPCrawlerConfig(max_file_size=bad_value)

    def test_accepted_and_serialized(self):
        cfg = HTTPCrawlerConfig(max_file_size=12345)
        assert cfg.max_file_size == 12345
        assert cfg.to_dict()["max_file_size"] == 12345
        assert cfg.clone().max_file_size == 12345
        assert HTTPCrawlerConfig.from_kwargs({"max_file_size": 7}).max_file_size == 7


class TestBinarySizeLimit:
    """Stream-to-disk binary downloads with an optional per-file cap."""

    def test_within_limit_publishes_final_file(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                # Cap exactly equals the decompressed payload: boundary is
                # inclusive, so the file must be saved.
                result = await _crawl_with_cap(
                    srv, "/gzip.bin", dl_dir, len(GZIP_PAYLOAD)
                )

                assert result.downloaded_files is not None
                filepath = result.downloaded_files[0]
                assert filepath.endswith("data.bin")
                assert os.path.isfile(filepath)
                with open(filepath, "rb") as f:
                    # Written bytes are the decompressed payload.
                    assert f.read() == GZIP_PAYLOAD
                assert _part_files(dl_dir) == []
                assert result.html == ""
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_gzip_measured_by_decompressed_size(self, tmp_path):
        # The wire body is well below the cap, but the expanded body is not —
        # the cap must be enforced against decompressed bytes.
        assert len(GZIP_COMPRESSED) < 2048, (
            "test precondition: compressed fixture must be smaller than cap"
        )

        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                with pytest.raises(DownloadSizeLimitError) as exc_info:
                    await _crawl_with_cap(srv, "/gzip.bin", dl_dir, 2048)

                err = exc_info.value
                assert err.filename == "data.bin"
                assert err.limit == 2048
                assert err.received == len(GZIP_PAYLOAD)
                assert "exceeds the maximum file size limit" in str(err)
                assert not os.path.exists(os.path.join(dl_dir, "data.bin"))
                assert _part_files(dl_dir) == []
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_over_limit_aborts_stops_transfer_and_cleans(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                cap = 64 * 1024
                with pytest.raises(DownloadSizeLimitError) as exc_info:
                    await _crawl_with_cap(srv, "/big.pdf", dl_dir, cap)

                err = exc_info.value
                assert err.filename == "big.pdf"
                assert err.limit == cap
                assert err.received > cap
                # No final file, no unfinished temp left behind.
                assert not os.path.exists(os.path.join(dl_dir, "big.pdf"))
                assert _part_files(dl_dir) == []

                # Give the server a beat to notice the client went away.
                await asyncio.sleep(0.2)
                # The client must have stopped reading long before 5 MiB.
                assert srv.state["big_pdf_sent"] < BIG_PDF_TOTAL
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_over_limit_keeps_existing_same_name_file(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                existing = os.path.join(dl_dir, "big.pdf")
                with open(existing, "wb") as f:
                    f.write(b"previously-downloaded-content")

                with pytest.raises(DownloadSizeLimitError):
                    await _crawl_with_cap(srv, "/big.pdf", dl_dir, 64 * 1024)

                with open(existing, "rb") as f:
                    assert f.read() == b"previously-downloaded-content"
                assert _part_files(dl_dir) == []
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_text_attachment_ignores_limit(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                # Cap smaller than the tiny CSV body — text attachments keep
                # the original buffered behavior regardless.
                result = await _crawl_with_cap(srv, "/data.csv", dl_dir, 10)

                assert result.downloaded_files is not None
                assert result.downloaded_files[0].endswith("data.csv")
                assert "id,name,value" in result.html
                assert _part_files(dl_dir) == []
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_html_page_ignores_limit(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl_with_cap(srv, "/page.html", dl_dir, 1)

                assert "<html>" in result.html
                assert result.downloaded_files is None
                assert os.listdir(dl_dir) == []
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_without_limit_keeps_buffered_full_download(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                result = await _crawl(srv, "/big.pdf", dl_dir)

                assert result.downloaded_files is not None
                filepath = result.downloaded_files[0]
                assert os.path.getsize(filepath) == BIG_PDF_TOTAL
                assert _part_files(dl_dir) == []
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_truncated_transfer_reports_failure_and_cleans(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                existing = os.path.join(dl_dir, "truncated.bin")
                with open(existing, "wb") as f:
                    f.write(b"keep-me")

                with pytest.raises(HTTPCrawlerError) as exc_info:
                    await _crawl_with_cap(srv, "/truncated.bin", dl_dir, 10 * 1024 * 1024)

                assert not isinstance(exc_info.value, DownloadSizeLimitError)
                message = str(exc_info.value).lower()
                assert "interrupt" in message or "timed out" in message
                # Existing same-name file survives; temp file is gone.
                with open(existing, "rb") as f:
                    assert f.read() == b"keep-me"
                assert _part_files(dl_dir) == []
            finally:
                await srv.stop()

        asyncio.run(_test())

    def test_save_failure_reports_and_cleans_temp(self, tmp_path):
        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                # Final path occupied by a directory: os.replace must fail.
                os.makedirs(os.path.join(dl_dir, "report.pdf"))
                with pytest.raises(HTTPCrawlerError) as exc_info:
                    await _crawl_with_cap(srv, "/report.pdf", dl_dir, 10 * 1024 * 1024)

                assert "finalize" in str(exc_info.value).lower()
                assert os.path.isdir(os.path.join(dl_dir, "report.pdf"))
                assert _part_files(dl_dir) == []
            finally:
                await srv.stop()

        asyncio.run(_test())


class TestSizeLimitCrawlResult:
    """The AsyncWebCrawler result must explain an aborted oversized download."""

    def test_crawl_result_reports_size_limit(self, tmp_path):
        from crawl4ai import AsyncWebCrawler, CacheMode

        async def _test():
            srv = _TestServer()
            await srv.start()
            try:
                dl_dir = str(tmp_path / "dl")
                os.makedirs(dl_dir)
                strategy = AsyncHTTPCrawlerStrategy(
                    browser_config=HTTPCrawlerConfig(
                        downloads_path=dl_dir, max_file_size=64 * 1024
                    )
                )
                async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
                    result = await crawler.arun(
                        url=srv.url("/big.pdf"),
                        config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
                    )

                assert result.success is False
                assert "exceeds the maximum file size limit" in (
                    result.error_message or ""
                )
                assert not result.downloaded_files
                assert not os.path.exists(os.path.join(dl_dir, "big.pdf"))
                assert _part_files(dl_dir) == []
                await asyncio.sleep(0.1)
            finally:
                await srv.stop()

        asyncio.run(_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
