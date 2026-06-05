from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from .base import BaseProvider
from ..core.console import print_source_banner
from ..core.http import NetworkManager
from ..core.models import TrackMetadata, DownloadResult
from ..core.errors import SpotiflacError, ErrorKind, TrackNotFoundError
from ..core.tagger import embed_metadata, EmbedOptions
from ..core.download_validation import validate_downloaded_track
from ..core.musicbrainz import AsyncMBFetch, mb_result_to_tags

logger = logging.getLogger(__name__)

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ──────────────────────────────────────────────────────────────────────────────
# Two independent backends for JOOX.
#
# GD Studio  — music-api.gdstudio.xyz/api.php
#   Parameterised GET: types, source, id/name, br, count, pages, size.
#   Returns JSON objects; `url` field holds the signed CDN link.
#   Licence: GD音乐台 (music.gdstudio.xyz) — CC BY-NC 4.0, study use only.
#
# HEMusic    — music.wjhe.top/api/music/joox/*
#   REST-style paths: /search, /url.
#   Search returns data.data[]; each item carries `fileLinks` sorted by
#   quality.  The /url endpoint resolves to a CDN redirect.
# ──────────────────────────────────────────────────────────────────────────────

_GDSTUDIO_BASE = "https://music-api.gdstudio.xyz/api.php"
_HEMUSIC_BASE  = "https://music.wjhe.top/api/music/joox"
_SOURCE        = "joox"

# Bitrate thresholds (kbps) used by the GD Studio backend.
# We request 999 (maximum) and only accept the file when the API reports
# at least CD-lossless quality (≥ 740 kbps for 16-bit FLAC).
_BR_REQUEST    = 999   # sent as the `br` parameter
_BR_MIN_FLAC   = 740   # minimum acceptable — below this we raise, not fall back

# HEMusic quality floor: the API returns a numeric `quality` field per
# fileLink; we skip any link whose reported quality is below this value.
_HE_MIN_QUALITY = 740


class JooxProvider(BaseProvider):
    """
    Provider for JOOX lossless audio.

    Tries two independent backends in order:

    1. GD Studio  (music-api.gdstudio.xyz) — primary, fast, well-structured.
    2. HEMusic    (music.wjhe.top)         — fallback, richer fileLink catalogue.

    Both backends are queried only for FLAC.  If neither can serve a lossless
    file the track is marked as unavailable rather than falling back to lossy.

    API credits:
      · GD音乐台 (music.gdstudio.xyz) — CC BY-NC 4.0, study use only.
      · HEMusic  (music.wjhe.top)    — personal / educational use only.
    """

    name = "joox"

    def __init__(self, timeout_s: int = 30) -> None:
        super().__init__(timeout_s=timeout_s)
        self._session = NetworkManager.get_sync_client()
        self._session.headers.update({
            "User-Agent": _DEFAULT_UA,
            "Referer":    "https://music.wjhe.top/",   # required by HEMusic
        })

    # ══════════════════════════════════════════════════════════════════════════
    # GD Studio backend
    # ══════════════════════════════════════════════════════════════════════════

    def _gd_search(self, query: str, count: int = 10) -> list[dict]:
        """Search JOOX via GD Studio.  Returns raw result items."""
        try:
            resp = self._session.get(
                _GDSTUDIO_BASE,
                params={
                    "types":  "search",
                    "source": _SOURCE,
                    "name":   query,
                    "count":  count,
                    "pages":  1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", data.get("result", []))
        except Exception as exc:
            logger.debug("[joox/gd] Search failed for %r: %s", query, exc)
        return []

    def _gd_get_stream(self, track_id: str) -> tuple[str, int]:
        """
        Request a lossless stream URL from GD Studio.

        Returns (url, actual_br).  url is empty string when the backend
        cannot serve a file that meets the FLAC quality floor.
        """
        try:
            resp = self._session.get(
                _GDSTUDIO_BASE,
                params={
                    "types":  "url",
                    "source": _SOURCE,
                    "id":     track_id,
                    "br":     _BR_REQUEST,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data      = resp.json()
            url       = data.get("url", "")
            actual_br = int(data.get("br", 0))

            if not url:
                logger.debug("[joox/gd] Empty URL for id=%s", track_id)
                return "", 0

            if actual_br < _BR_MIN_FLAC:
                logger.debug(
                    "[joox/gd] id=%s returned br=%d — below FLAC floor (%d), skipping",
                    track_id, actual_br, _BR_MIN_FLAC,
                )
                return "", actual_br

            return url, actual_br

        except Exception as exc:
            logger.debug("[joox/gd] Stream fetch failed for id=%s: %s", track_id, exc)
        return "", 0

    def _gd_get_pic_url(self, pic_id: str, size: int = 500) -> str:
        """Fetch album-art URL from GD Studio."""
        if not pic_id:
            return ""
        try:
            resp = self._session.get(
                _GDSTUDIO_BASE,
                params={
                    "types":  "pic",
                    "source": _SOURCE,
                    "id":     pic_id,
                    "size":   size,
                },
                timeout=8,
            )
            resp.raise_for_status()
            return resp.json().get("url", "")
        except Exception as exc:
            logger.debug("[joox/gd] Pic fetch failed for pic_id=%s: %s", pic_id, exc)
        return ""

    def _gd_get_lyric(self, lyric_id: str) -> str:
        """Fetch LRC lyrics via GD Studio (original lang, falls back to translation)."""
        if not lyric_id:
            return ""
        try:
            resp = self._session.get(
                _GDSTUDIO_BASE,
                params={
                    "types":  "lyric",
                    "source": _SOURCE,
                    "id":     lyric_id,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("lyric", "") or data.get("tlyric", "")
        except Exception as exc:
            logger.debug("[joox/gd] Lyric fetch failed for id=%s: %s", lyric_id, exc)
        return ""

    def _gd_get_album_tracks(self, album_id: str) -> list[dict]:
        """Fetch track list for a JOOX album via GD Studio."""
        try:
            resp = self._session.get(
                _GDSTUDIO_BASE,
                params={
                    "types":  "search",
                    "source": f"{_SOURCE}_album",
                    "name":   album_id,
                    "count":  100,
                    "pages":  1,
                },
                timeout=12,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.debug("[joox/gd] Album fetch failed for id=%s: %s", album_id, exc)
        return []

    # ══════════════════════════════════════════════════════════════════════════
    # HEMusic backend
    # ══════════════════════════════════════════════════════════════════════════

    def _he_search(self, query: str, page_size: int = 20, page: int = 1) -> list[dict]:
        """
        Search JOOX via HEMusic.

        The API returns::

            { "data": { "data": [ <item>, … ] } }

        Each item carries a ``fileLinks`` list — entries with ``quality`` ≥
        _HE_MIN_QUALITY are FLAC-capable.
        """
        import time
        try:
            resp = self._session.get(
                f"{_HEMUSIC_BASE}/search",
                params={
                    "key":       query,
                    "pageIndex": page,
                    "pageSize":  page_size,
                    "_":         str(int(time.time() * 1000)),
                },
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = (
                payload.get("data", {}).get("data", [])
                if isinstance(payload, dict)
                else []
            )
            return items if isinstance(items, list) else []
        except Exception as exc:
            logger.debug("[joox/he] Search failed for %r: %s", query, exc)
        return []

    def _he_get_stream(self, track_id: str, quality: int, fmt: str = "flac") -> str:
        """
        Resolve a CDN URL via HEMusic's /url endpoint.

        Follows the redirect (HEAD) to obtain the final signed URL.
        Returns empty string on any failure.
        """
        endpoint = f"{_HEMUSIC_BASE}/url"
        params   = {"ID": track_id, "quality": quality, "format": fmt}
        try:
            resp = self._session.head(
                endpoint,
                params=params,
                timeout=10,
                allow_redirects=True,
            )
            resp.raise_for_status()
            final_url = str(resp.url)
            # A redirect to an error page or empty body means unavailable
            if not final_url or "error" in final_url.lower():
                return ""
            return final_url
        except Exception as exc:
            logger.debug(
                "[joox/he] Stream resolve failed for id=%s q=%s: %s",
                track_id, quality, exc,
            )
        return ""

    def _he_best_flac_link(self, item: dict) -> tuple[str, int, str]:
        """
        Pick the highest-quality FLAC fileLink from a HEMusic search result.

        Returns (track_id, quality, format) or ("", 0, "") when none qualify.
        """
        track_id   = str(item.get("ID", ""))
        file_links = item.get("fileLinks", [])
        if not track_id or not file_links:
            return "", 0, ""

        flac_links = [
            fl for fl in file_links
            if isinstance(fl, dict)
            and float(fl.get("quality", 0)) >= _HE_MIN_QUALITY
            and str(fl.get("format", "")).lower() in {"flac", ""}
        ]
        if not flac_links:
            return "", 0, ""

        best = max(flac_links, key=lambda fl: float(fl.get("quality", 0)))
        return track_id, int(float(best["quality"])), str(best.get("format", "flac"))

    def _he_get_cover(self, track_id: str, size: int = 500) -> str:
        """Resolve cover-art URL via HEMusic, following redirect."""
        endpoint = f"{_HEMUSIC_BASE}/url"
        try:
            resp = self._session.head(
                endpoint,
                params={"ID": track_id, "quality": size, "format": "jpg"},
                timeout=8,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return str(resp.url)
        except Exception as exc:
            logger.debug("[joox/he] Cover fetch failed for id=%s: %s", track_id, exc)
        return ""

    # ══════════════════════════════════════════════════════════════════════════
    # Shared helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _gd_item_to_metadata(self, item: dict) -> TrackMetadata:
        """Convert a GD Studio search result into TrackMetadata."""
        track_id = str(item.get("id", ""))
        title    = item.get("name", "Unknown")

        raw_artists = item.get("artist", [])
        if isinstance(raw_artists, list):
            artist_str = ", ".join(
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in raw_artists
            ).strip(", ") or "Unknown"
        else:
            artist_str = str(raw_artists) or "Unknown"

        album        = item.get("album", "Unknown")
        pic_id       = str(item.get("pic_id", ""))
        duration_ms  = int(item.get("duration_ms", item.get("duration", 0)) or 0)
        cover_url    = self._gd_get_pic_url(pic_id) if pic_id else ""

        return TrackMetadata(
            id           = f"joox_{track_id}",
            title        = title,
            artists      = artist_str,
            album        = album,
            album_artist = artist_str,
            duration_ms  = duration_ms,
            cover_url    = cover_url,
            external_url = "",
            extra_info   = {
                "provider":     "joox",
                "backend":      "gdstudio",
                "raw_track_id": track_id,
                "pic_id":       pic_id,
                "lyric_id":     str(item.get("lyric_id", track_id)),
            },
        )

    def _he_item_to_metadata(self, item: dict) -> TrackMetadata:
        """Convert a HEMusic search result into TrackMetadata."""
        track_id = str(item.get("ID", ""))
        title    = item.get("name") or item.get("title") or "Unknown"

        singers = item.get("singers", [])
        if isinstance(singers, list):
            artist_str = ", ".join(
                s.get("name", "") if isinstance(s, dict) else str(s)
                for s in singers
            ).strip(", ") or "Unknown"
        else:
            artist_str = str(singers) or "Unknown"

        album       = (item.get("album") or {}).get("name", "Unknown")
        duration_ms = int((item.get("duration") or 0)) * 1000   # HEMusic gives seconds
        cover_url   = self._he_get_cover(track_id) if track_id else ""

        return TrackMetadata(
            id           = f"joox_he_{track_id}",
            title        = title,
            artists      = artist_str,
            album        = album,
            album_artist = artist_str,
            duration_ms  = duration_ms,
            cover_url    = cover_url,
            external_url = "",
            extra_info   = {
                "provider":     "joox",
                "backend":      "hemusic",
                "raw_track_id": track_id,
                "lyric_id":     track_id,
                "pic_id":       "",
            },
        )

    def _embed_lyrics(self, dest: Path, lyrics: str, extension: str) -> None:
        """Inject lyrics into an already-tagged file."""
        if not lyrics or not lyrics.strip():
            return
        try:
            if extension == ".flac":
                from mutagen.flac import FLAC as _FLAC
                audio = _FLAC(str(dest))
                if "LYRICS" not in audio:
                    audio["LYRICS"] = lyrics
                    audio.save()
            else:
                from mutagen.id3 import ID3, USLT
                try:
                    audio = ID3(str(dest))
                except Exception:
                    audio = ID3()
                if not audio.get("USLT::eng"):
                    audio.add(USLT(encoding=3, lang="eng", desc="", text=lyrics))
                    audio.save(str(dest), v2_version=3)
            logger.debug("[joox] Lyrics embedded (%d chars)", len(lyrics))
        except Exception as exc:
            logger.warning("[joox] Lyrics embed failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    # Public interface — get_url
    # ══════════════════════════════════════════════════════════════════════════

    def get_url(self, url: str) -> tuple[str, list[TrackMetadata]]:
        """
        Resolve a JOOX track/album URL or free-text query to a list of
        TrackMetadata objects.

        Accepted inputs
        ---------------
        * A numeric JOOX track ID (5+ digits).
        * A URL or path segment containing ``_album`` + a numeric album ID.
        * A free-text search query (falls back to both backends).
        """
        match = re.search(r"(\d{5,})", url)

        # ── album ──────────────────────────────────────────────────────────
        if match and "_album" in url.lower():
            album_id = match.group(1)
            items = self._gd_get_album_tracks(album_id)
            if items:
                tracks     = [self._gd_item_to_metadata(it) for it in items]
                album_name = tracks[0].album if tracks else "Unknown Album"
                return album_name, tracks

        # ── single track by ID ─────────────────────────────────────────────
        if match:
            track_id = match.group(1)
            items = self._gd_search(track_id, count=1)
            if items:
                meta = self._gd_item_to_metadata(items[0])
                return meta.title, [meta]

        # ── free-text search — try GD Studio first, then HEMusic ───────────
        query = url.strip()

        items = self._gd_search(query, count=20)
        if items:
            tracks = [self._gd_item_to_metadata(it) for it in items]
            return f"Search: {query}", tracks

        items = self._he_search(query, page_size=20)
        if items:
            tracks = [self._he_item_to_metadata(it) for it in items]
            return f"Search: {query}", tracks

        raise SpotiflacError(
            ErrorKind.TRACK_NOT_FOUND,
            f"No results for: {query}",
            self.name,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Public interface — download_track
    # ══════════════════════════════════════════════════════════════════════════

    def download_track(
            self,
            metadata:            TrackMetadata,
            output_dir:          str,
            *,
            quality:             str              = "LOSSLESS",
            filename_format:     str              = "{title} - {artist}",
            position:            int              = 1,
            include_track_num:   bool             = False,
            use_album_track_num: bool             = False,
            first_artist_only:   bool             = False,
            allow_fallback:      bool             = True,
            embed_lyrics:        bool             = False,
            lyrics_providers:    list[str] | None = None,
            enrich_metadata:     bool             = False,
            enrich_providers:    list[str] | None = None,
            qobuz_token:         str | None       = None,
            is_album:            bool             = False,
            **kwargs:            Any,
    ) -> DownloadResult:
        """
        Download a JOOX track as FLAC.

        Resolution order
        ----------------
        1. GD Studio backend  (preferred — fast, structured).
        2. HEMusic backend    (fallback — richer FLAC catalogue).

        The track is rejected if neither backend can serve a file at or above
        the FLAC quality floor (_BR_MIN_FLAC / _HE_MIN_QUALITY).  No lossy
        fallback is attempted.
        """
        try:
            extra   = metadata.extra_info or {}
            backend = extra.get("backend", "gdstudio")

            # ── resolve the track on the appropriate backend ──────────────
            if backend == "hemusic":
                dl_url, quality_label, extension, raw_track_id = \
                    self._resolve_via_hemusic(metadata, extra)
            else:
                dl_url, quality_label, extension, raw_track_id = \
                    self._resolve_via_gdstudio(metadata, extra)

            # ── build destination path ────────────────────────────────────
            dest = self._build_output_path(
                metadata, output_dir, filename_format,
                position, include_track_num, use_album_track_num, first_artist_only,
                extension=extension,
            )

            if self._file_exists(dest):
                fmt = extension.lstrip(".")
                return DownloadResult.skipped_result(self.name, str(dest), fmt=fmt)

            # ── MusicBrainz async prefetch ────────────────────────────────
            mb_fetcher = AsyncMBFetch(metadata.isrc) if getattr(metadata, "isrc", None) else None

            # ── source banner ─────────────────────────────────────────────
            api_base = _HEMUSIC_BASE if backend == "hemusic" else _GDSTUDIO_BASE
            print_source_banner("joox", api_base, quality_label)

            # ── download ──────────────────────────────────────────────────
            logger.info(
                "[joox/%s] Downloading %r (id=%s, %s)",
                backend, metadata.title, raw_track_id, quality_label,
            )
            self._http.stream_to_file(dl_url, str(dest), self._progress_cb)

            # ── validate ──────────────────────────────────────────────────
            expected_s = (metadata.duration_ms or 0) // 1000
            valid, err_msg = validate_downloaded_track(str(dest), expected_s)
            if not valid:
                if dest.exists():
                    os.remove(str(dest))
                return DownloadResult.fail(self.name, f"Validation failed: {err_msg}")

            # ── lyrics ────────────────────────────────────────────────────
            fetched_lyrics: str | None = None
            if embed_lyrics:
                lyric_id = extra.get("lyric_id", raw_track_id)
                if backend == "hemusic":
                    # HEMusic does not expose a lyrics endpoint; delegate to
                    # embed_metadata's own lyrics_providers pipeline.
                    pass
                else:
                    fetched_lyrics = self._gd_get_lyric(lyric_id) or None

            # ── MusicBrainz tags ──────────────────────────────────────────
            mb_tags: dict[str, str] = {}
            if mb_fetcher:
                mb_tags = mb_result_to_tags(mb_fetcher.future.result())

            # ── cover art resolution ──────────────────────────────────────
            cover_url = metadata.cover_url
            if not cover_url:
                if backend == "hemusic":
                    cover_url = self._he_get_cover(raw_track_id)
                else:
                    pic_id = extra.get("pic_id", "")
                    if pic_id:
                        cover_url = self._gd_get_pic_url(pic_id)

            if cover_url and cover_url != metadata.cover_url:
                metadata = metadata.model_copy(update={"cover_url": cover_url})

            # ── embed metadata ────────────────────────────────────────────
            opts = EmbedOptions(
                first_artist_only  = first_artist_only,
                cover_url          = cover_url or metadata.cover_url,
                extra_tags         = mb_tags,
                embed_lyrics       = embed_lyrics,
                lyrics_providers   = lyrics_providers or [],
                enrich             = enrich_metadata,
                enrich_providers   = enrich_providers,
                enrich_qobuz_token = qobuz_token or "",
                is_album           = is_album,
            )
            embed_metadata(str(dest), metadata, opts, session=self._session)

            # ── inject GD Studio lyrics when embed_metadata found nothing ─
            if fetched_lyrics:
                self._embed_lyrics(dest, fetched_lyrics, extension)

            fmt = extension.lstrip(".")
            return DownloadResult.ok(self.name, str(dest), fmt=fmt)

        except SpotiflacError as exc:
            logger.error("[%s] %s", self.name, exc)
            return DownloadResult.fail(self.name, str(exc))
        except Exception as exc:
            logger.exception("[%s] Unexpected error", self.name)
            return DownloadResult.fail(self.name, f"Unexpected: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # Private resolution helpers (one per backend)
    # ══════════════════════════════════════════════════════════════════════════

    def _resolve_via_gdstudio(
        self,
        metadata: TrackMetadata,
        extra: dict,
    ) -> tuple[str, str, str, str]:
        """
        Resolve the download URL using the GD Studio backend.

        Returns (dl_url, quality_label, extension, raw_track_id).
        Raises SpotiflacError / TrackNotFoundError on failure.
        Falls through to HEMusic if no lossless URL is available.
        """
        raw_track_id = extra.get("raw_track_id", "")

        if not raw_track_id:
            query = f"{metadata.title} {metadata.first_artist}".strip()
            logger.info("[joox/gd] Searching for: %s", query)
            items = self._gd_search(query, count=5)
            if not items:
                raise TrackNotFoundError(self.name, f"Not found on GD Studio: {query}")
            raw_track_id = str(items[0].get("id", ""))
            if not raw_track_id:
                raise TrackNotFoundError(self.name, "Empty track ID from GD Studio")
            extra = extra | {
                "raw_track_id": raw_track_id,
                "pic_id":       str(items[0].get("pic_id", "")),
                "lyric_id":     str(items[0].get("lyric_id", raw_track_id)),
            }

        dl_url, actual_br = self._gd_get_stream(raw_track_id)

        if dl_url:
            quality_label = f"FLAC {actual_br} kbps"
            return dl_url, quality_label, ".flac", raw_track_id

        # GD Studio could not serve FLAC — try HEMusic
        logger.info(
            "[joox/gd] No FLAC for id=%s (br=%d) — trying HEMusic",
            raw_track_id, actual_br,
        )
        return self._resolve_via_hemusic(metadata, extra)

    def _resolve_via_hemusic(
        self,
        metadata: TrackMetadata,
        extra: dict,
    ) -> tuple[str, str, str, str]:
        """
        Resolve the download URL using the HEMusic backend.

        Returns (dl_url, quality_label, extension, raw_track_id).
        Raises SpotiflacError when no qualifying FLAC link is found.
        """
        raw_track_id = extra.get("raw_track_id", "")

        # ── search if we have no ID or came from GD Studio metadata ───────
        if not raw_track_id or extra.get("backend") != "hemusic":
            query = f"{metadata.title} {metadata.first_artist}".strip()
            logger.info("[joox/he] Searching for: %s", query)
            items = self._he_search(query, page_size=10)
            if not items:
                raise SpotiflacError(
                    ErrorKind.UNAVAILABLE,
                    f"No FLAC available on either backend for: {query}",
                    self.name,
                )
            # Pick the first item that has a qualifying FLAC fileLink
            chosen_id = chosen_quality = chosen_fmt = None
            for item in items:
                tid, q, fmt = self._he_best_flac_link(item)
                if tid:
                    chosen_id, chosen_quality, chosen_fmt = tid, q, fmt
                    break

            if not chosen_id:
                raise SpotiflacError(
                    ErrorKind.UNAVAILABLE,
                    f"HEMusic returned results but none have FLAC ≥ {_HE_MIN_QUALITY} kbps",
                    self.name,
                )
            raw_track_id  = chosen_id
            best_quality  = chosen_quality
            best_fmt      = chosen_fmt
        else:
            # ID already known from a previous HEMusic search result
            items = self._he_search(
                f"{metadata.title} {metadata.first_artist}".strip(),
                page_size=5,
            )
            chosen_id = chosen_quality = chosen_fmt = None
            for item in items:
                tid, q, fmt = self._he_best_flac_link(item)
                if tid == raw_track_id:
                    chosen_id, chosen_quality, chosen_fmt = tid, q, fmt
                    break
            if not chosen_id:
                raise SpotiflacError(
                    ErrorKind.UNAVAILABLE,
                    f"HEMusic: no FLAC link for id={raw_track_id}",
                    self.name,
                )
            best_quality = chosen_quality
            best_fmt     = chosen_fmt

        dl_url = self._he_get_stream(raw_track_id, best_quality, best_fmt)
        if not dl_url:
            raise SpotiflacError(
                ErrorKind.UNAVAILABLE,
                f"HEMusic: stream resolve failed for id={raw_track_id}",
                self.name,
            )

        quality_label = f"FLAC {best_quality} kbps (HEMusic)"
        return dl_url, quality_label, ".flac", raw_track_id