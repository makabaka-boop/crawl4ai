"""
Tests for the optional conditional-refresh mode for sitemap indexes
(SeedingConfig.sitemap_conditional_refresh).

Behavior under test:
* Every child currently listed by the index is requested on each run.
* Cached ETag / Last-Modified are sent as If-None-Match / If-Modified-Since.
* 304 reuses the cached URL list; 200 replaces the whole list (even if empty).
* Children removed from the index stop contributing URLs (cache pruned).
* Download/parse failures log a warning, reuse the stale list and never abort
  the other children.
* Merged results are de-duplicated; force=True bypasses all caches.
"""
import gzip
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Stub rank_bm25 (optional dep) the same way the namespace tests do.
class _FakeBM25:
    def __init__(self, corpus):
        self._scores = [1.0] * len(corpus)

    def get_scores(self, tokens):
        return self._scores


sys.modules.setdefault("rank_bm25", SimpleNamespace(BM25Okapi=_FakeBM25))

import httpx  # noqa: E402

from crawl4ai.async_url_seeder import (  # noqa: E402
    AsyncUrlSeeder,
    _read_sub_sitemap_cache,
)

NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
INDEX_URL = "https://example.com/sitemap.xml"
CHILD_A = "https://example.com/a.xml"
CHILD_B = "https://example.com/b.xml"
CHILD_C = "https://example.com/c.xml"


def urlset(*urls: str) -> bytes:
    items = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return f'<?xml version="1.0"?><urlset xmlns="{NS}">{items}</urlset>'.encode()


def empty_urlset() -> bytes:
    return f'<?xml version="1.0"?><urlset xmlns="{NS}"></urlset>'.encode()


def sitemap_index(*urls: str) -> bytes:
    items = "".join(f"<sitemap><loc>{u}</loc></sitemap>" for u in urls)
    return (
        f'<?xml version="1.0"?><sitemapindex xmlns="{NS}">{items}'
        f"</sitemapindex>"
    ).encode()


class FakeHeaders(dict):
    def __init__(self, mapping=None):
        super().__init__({k.lower(): v for k, v in (mapping or {}).items()})

    def get(self, key, default=None):
        return super().get(key.lower(), default)


class FakeHTTPError(Exception):
    pass


class FakeResponse:
    def __init__(self, url, status_code=200, body=b"", headers=None):
        self.url = url
        self.status_code = status_code
        self.content = body
        self.headers = FakeHeaders(headers)

    @property
    def text(self):
        return self.content.decode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeHTTPError(f"HTTP {self.status_code} for {self.url}")


class FakeClient:
    """Routes map URL -> FakeResponse | callable(kwargs) | Exception."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        route = self.routes[url]
        if isinstance(route, BaseException):
            raise route
        if callable(route):
            return route(**kwargs)
        return route

    async def head(self, url, **kwargs):
        return FakeResponse(url, status_code=404)


@pytest.fixture
def make_seeder(tmp_path):
    def _make(routes):
        client = FakeClient(routes)
        seeder = AsyncUrlSeeder(
            client=client,
            base_directory=tmp_path,
            cache_root=tmp_path / "head_cache",
        )
        seeder._seed_concurrency = 8
        seeder._cache_ttl_hours = 24
        seeder._validate_sitemap_lastmod = True
        seeder._sitemap_conditional_refresh = True
        return seeder, client

    return _make


def _cache_file(seeder, index_url, child_url):
    return seeder._sub_sitemap_cache_file(
        seeder._sub_sitemap_cache_dir(index_url), child_url)


# --------------------------------------------------------------------------- #
# Per-child conditional refresh
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_first_fetch_sends_no_validators_and_persists(make_seeder):
    resp = FakeResponse(
        CHILD_A, body=urlset("https://example.com/a"),
        headers={"ETag": '"v1"', "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"},
    )
    seeder, client = make_seeder({CHILD_A: resp})
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)

    urls = await seeder._refresh_sub_sitemap(CHILD_A, cache_file, force=False)

    assert urls == ["https://example.com/a"]
    assert client.calls[0][1]["headers"] == {}
    cached = _read_sub_sitemap_cache(cache_file)
    assert cached is not None
    assert cached["etag"] == '"v1"'
    assert cached["last_modified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert cached["urls"] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_304_reuses_cached_list_and_sends_validators(make_seeder):
    seeder, client = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/a"),
            headers={"ETag": '"v1"', "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"}),
    })
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)
    await seeder._refresh_sub_sitemap(CHILD_A, cache_file, force=False)

    def conditional_route(**kwargs):
        sent = kwargs.get("headers") or {}
        assert sent.get("If-None-Match") == '"v1"'
        assert sent.get("If-Modified-Since") == "Mon, 01 Jan 2024 00:00:00 GMT"
        return FakeResponse(CHILD_A, status_code=304, body=b"")

    client.routes[CHILD_A] = conditional_route
    urls = await seeder._refresh_sub_sitemap(CHILD_A, cache_file, force=False)

    assert urls == ["https://example.com/a"]
    cached = _read_sub_sitemap_cache(cache_file)
    assert cached["urls"] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_last_modified_only_sends_if_modified_since(make_seeder):
    seeder, client = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/a"),
            headers={"Last-Modified": "Tue, 02 Jan 2024 00:00:00 GMT"}),
    })
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)
    await seeder._refresh_sub_sitemap(CHILD_A, cache_file, force=False)

    seen = {}

    def conditional_route(**kwargs):
        seen.update(kwargs.get("headers") or {})
        return FakeResponse(CHILD_A, status_code=304)

    client.routes[CHILD_A] = conditional_route
    urls = await seeder._refresh_sub_sitemap(CHILD_A, cache_file, force=False)

    assert urls == ["https://example.com/a"]
    assert seen == {"If-Modified-Since": "Tue, 02 Jan 2024 00:00:00 GMT"}


@pytest.mark.asyncio
async def test_200_replaces_whole_list_including_empty(make_seeder):
    seeder, client = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/old"),
            headers={"ETag": '"v1"'}),
    })
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)
    assert await seeder._refresh_sub_sitemap(CHILD_A, cache_file) == [
        "https://example.com/old"]

    # Newer representation with an EMPTY urlset: the empty list must replace
    # the old one rather than being treated as a cache miss.
    client.routes[CHILD_A] = FakeResponse(
        CHILD_A, body=empty_urlset(), headers={"ETag": '"v2"'})
    urls = await seeder._refresh_sub_sitemap(CHILD_A, cache_file)
    assert urls == []
    cached = _read_sub_sitemap_cache(cache_file)
    assert cached["urls"] == []
    assert cached["etag"] == '"v2"'

    # A later 304 keeps reusing that empty list.
    client.routes[CHILD_A] = lambda **kw: FakeResponse(CHILD_A, status_code=304)
    assert await seeder._refresh_sub_sitemap(CHILD_A, cache_file) == []


@pytest.mark.asyncio
async def test_download_failure_reuses_stale_list(make_seeder):
    seeder, client = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/keep"),
            headers={"ETag": '"v1"'}),
    })
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)
    await seeder._refresh_sub_sitemap(CHILD_A, cache_file)

    # Network error -> stale list.
    client.routes[CHILD_A] = httpx.ConnectError("simulated outage")
    assert await seeder._refresh_sub_sitemap(CHILD_A, cache_file) == [
        "https://example.com/keep"]

    # HTTP 5xx -> stale list as well.
    client.routes[CHILD_A] = FakeResponse(CHILD_A, status_code=500)
    assert await seeder._refresh_sub_sitemap(CHILD_A, cache_file) == [
        "https://example.com/keep"]


@pytest.mark.asyncio
async def test_parse_failure_reuses_stale_list(make_seeder):
    seeder, client = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/keep"),
            headers={"ETag": '"v1"'}),
    })
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)
    await seeder._refresh_sub_sitemap(CHILD_A, cache_file)

    client.routes[CHILD_A] = FakeResponse(CHILD_A, body=b"")
    assert await seeder._refresh_sub_sitemap(CHILD_A, cache_file) == [
        "https://example.com/keep"]


@pytest.mark.asyncio
async def test_failure_without_stale_list_contributes_nothing(make_seeder):
    seeder, client = make_seeder({CHILD_A: httpx.ConnectError("boom")})
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)

    assert await seeder._refresh_sub_sitemap(CHILD_A, cache_file) == []


@pytest.mark.asyncio
async def test_force_refresh_omits_conditional_headers(make_seeder):
    seeder, client = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/v1"),
            headers={"ETag": '"v1"'}),
    })
    cache_file = _cache_file(seeder, INDEX_URL, CHILD_A)
    await seeder._refresh_sub_sitemap(CHILD_A, cache_file)

    captured = {}

    def force_route(**kwargs):
        captured.update(kwargs.get("headers") or {})
        return FakeResponse(
            CHILD_A, body=urlset("https://example.com/v2"),
            headers={"ETag": '"v2"'})

    client.routes[CHILD_A] = force_route
    urls = await seeder._refresh_sub_sitemap(CHILD_A, cache_file, force=True)

    assert urls == ["https://example.com/v2"]
    assert captured == {}
    assert _read_sub_sitemap_cache(cache_file)["etag"] == '"v2"'


@pytest.mark.asyncio
async def test_gzipped_child_is_decompressed(make_seeder):
    body = gzip.compress(urlset("https://example.com/gz"))
    seeder, _ = make_seeder({
        "https://example.com/a.xml.gz": FakeResponse(
            "https://example.com/a.xml.gz", body=body,
            headers={"ETag": '"gz1"'}),
    })
    child = "https://example.com/a.xml.gz"
    cache_file = _cache_file(seeder, INDEX_URL, child)
    urls = await seeder._refresh_sub_sitemap(child, cache_file)
    assert urls == ["https://example.com/gz"]


# --------------------------------------------------------------------------- #
# Index-level merge / prune / resilience
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_index_merge_dedup_preserves_index_order(make_seeder):
    seeder, _ = make_seeder({
        CHILD_A: FakeResponse(CHILD_A, body=urlset(
            "https://example.com/a", "https://example.com/shared")),
        CHILD_B: FakeResponse(CHILD_B, body=urlset(
            "https://example.com/shared", "https://example.com/b")),
        CHILD_C: FakeResponse(CHILD_C, body=urlset("https://example.com/c")),
    })

    urls = [u async for u in seeder._from_sitemap_index_conditional(
        INDEX_URL, [CHILD_A, CHILD_B, CHILD_C])]

    assert urls == [
        "https://example.com/a",
        "https://example.com/shared",
        "https://example.com/b",
        "https://example.com/c",
    ]


@pytest.mark.asyncio
async def test_removed_child_is_dropped_and_its_cache_pruned(make_seeder):
    seeder, client = make_seeder({
        CHILD_A: FakeResponse(CHILD_A, body=urlset("https://example.com/a")),
        CHILD_B: FakeResponse(CHILD_B, body=urlset("https://example.com/b")),
    })
    cache_a = _cache_file(seeder, INDEX_URL, CHILD_A)
    cache_b = _cache_file(seeder, INDEX_URL, CHILD_B)

    first = [u async for u in seeder._from_sitemap_index_conditional(
        INDEX_URL, [CHILD_A, CHILD_B])]
    assert sorted(first) == ["https://example.com/a", "https://example.com/b"]
    assert cache_a.exists() and cache_b.exists()

    # Index no longer lists CHILD_A.
    second = [u async for u in seeder._from_sitemap_index_conditional(
        INDEX_URL, [CHILD_B])]

    assert second == ["https://example.com/b"]
    assert not cache_a.exists()
    assert cache_b.exists()
    # The removed child was never even requested.
    assert all(url != CHILD_A for url, _ in client.calls[len(client.calls) - 1:])


@pytest.mark.asyncio
async def test_failed_child_keeps_stale_while_others_continue(make_seeder):
    def child_b_conditional(**kwargs):
        assert (kwargs.get("headers") or {}).get("If-None-Match") == '"b1"'
        return FakeResponse(CHILD_B, status_code=304)

    seeder, client = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/a1"),
            headers={"ETag": '"a1"'}),
        CHILD_B: FakeResponse(
            CHILD_B, body=urlset("https://example.com/b1"),
            headers={"ETag": '"b1"'}),
        CHILD_C: FakeResponse(
            CHILD_C, body=urlset("https://example.com/c1"),
            headers={"ETag": '"c1"'}),
    })

    async def _collect(children):
        return [u async for u in seeder._from_sitemap_index_conditional(
            INDEX_URL, children)]

    await _collect([CHILD_A, CHILD_B, CHILD_C])

    # A dies, B is unchanged (304), C gains c2.
    client.routes[CHILD_A] = httpx.ConnectError("a is down")
    client.routes[CHILD_B] = child_b_conditional
    client.routes[CHILD_C] = FakeResponse(
        CHILD_C, body=urlset("https://example.com/c1", "https://example.com/c2"),
        headers={"ETag": '"c2"'})

    urls = await _collect([CHILD_A, CHILD_B, CHILD_C])
    assert urls == [
        "https://example.com/a1",   # stale list reused
        "https://example.com/b1",   # 304 reuse
        "https://example.com/c1",
        "https://example.com/c2",   # fresh replacement
    ]


# --------------------------------------------------------------------------- #
# End-to-end through _from_sitemaps
# --------------------------------------------------------------------------- #

def _install_index(seeder, index_body):
    async def fake_resolve_head(url):
        return url

    seeder._resolve_head = fake_resolve_head
    seeder.client.routes[INDEX_URL] = FakeResponse(
        INDEX_URL, body=index_body)


@pytest.mark.asyncio
async def test_from_sitemaps_conditional_end_to_end(make_seeder):
    def child1_route(**kwargs):
        headers = kwargs.get("headers") or {}
        if headers.get("If-None-Match") == '"p-v1"':
            return FakeResponse(CHILD_A, status_code=304)
        return FakeResponse(
            CHILD_A, body=urlset(
                "https://example.com/p1", "https://example.com/p2"),
            headers={"ETag": '"p-v1"'})

    seeder, client = make_seeder({
        CHILD_A: child1_route,
        CHILD_B: FakeResponse(
            CHILD_B, body=urlset("https://example.com/p3"),
            headers={"ETag": '"q-v1"'}),
    })
    _install_index(seeder, sitemap_index(CHILD_A, CHILD_B))

    async def collect(pattern="*"):
        return [u async for u in seeder._from_sitemaps(
            "example.com", pattern, force=False)]

    first = await collect()
    assert sorted(first) == [
        "https://example.com/p1",
        "https://example.com/p2",
        "https://example.com/p3",
    ]

    # CHILD B gains p4; CHILD A answers 304 and must not be re-downloaded.
    client.routes[CHILD_B] = FakeResponse(
        CHILD_B, body=urlset(
            "https://example.com/p3", "https://example.com/p4"),
        headers={"ETag": '"q-v2"'})
    second = await collect()
    assert sorted(second) == [
        "https://example.com/p1",
        "https://example.com/p2",
        "https://example.com/p3",
        "https://example.com/p4",
    ]

    # Existing pattern filtering still applies to the merged result.
    filtered = await collect(pattern="*p1*")
    assert filtered == ["https://example.com/p1"]


@pytest.mark.asyncio
async def test_from_sitemaps_force_bypasses_conditional_cache(make_seeder):
    captured = []

    def child_route(**kwargs):
        captured.append(kwargs.get("headers") or {})
        return FakeResponse(
            CHILD_A, body=urlset("https://example.com/v2"),
            headers={"ETag": '"v2"'})

    seeder, _ = make_seeder({
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/v1"),
            headers={"ETag": '"v1"'}),
    })
    _install_index(seeder, sitemap_index(CHILD_A))

    async def collect(force):
        return [u async for u in seeder._from_sitemaps(
            "example.com", "*", force=force)]

    assert await collect(False) == ["https://example.com/v1"]

    seeder.client.routes[CHILD_A] = child_route
    assert await collect(True) == ["https://example.com/v2"]
    assert captured == [{}]  # no validators sent on force refresh


@pytest.mark.asyncio
async def test_plain_urlset_falls_through_normal_flow(make_seeder):
    # Conditional mode only applies to sitemap indexes; a plain urlset must
    # keep working through the existing aggregate-cache flow.
    seeder, client = make_seeder({})
    _install_index(seeder, urlset("https://example.com/page-1"))

    first = [u async for u in seeder._from_sitemaps("example.com", "*")]
    assert first == ["https://example.com/page-1"]

    # Second run is served from the aggregate cache: discovery GET happens once
    # per run, but parsing still works regardless; just verify stable results.
    second = [u async for u in seeder._from_sitemaps("example.com", "*")]
    assert second == ["https://example.com/page-1"]


@pytest.mark.asyncio
async def test_robots_discovered_index_uses_conditional_refresh(make_seeder):
    robots_url = "https://example.com/robots.txt"
    custom_index = "https://example.com/custom-index.xml"
    seeder, client = make_seeder({
        robots_url: FakeResponse(
            robots_url, body=b"Sitemap: https://example.com/custom-index.xml\n"),
        custom_index: FakeResponse(
            custom_index, body=sitemap_index(CHILD_A)),
        CHILD_A: FakeResponse(
            CHILD_A, body=urlset("https://example.com/from-robots"),
            headers={"ETag": '"r1"'}),
    })

    # No well-known sitemap resolves; discovery must fall back to robots.txt.
    async def none_resolve(url):
        return None

    seeder._resolve_head = none_resolve

    urls = [u async for u in seeder._from_sitemaps("example.com", "*")]
    assert urls == ["https://example.com/from-robots"]
    assert _cache_file(seeder, custom_index, CHILD_A).exists()
