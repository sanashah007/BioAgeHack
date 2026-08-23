"""Phase 1 -- data acquisition.

Downloads are cached on disk and skipped if already present and non-trivially
sized, so re-running the pipeline is cheap and offline-safe after the first run.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "bioage-pipeline/0.1 (research; contact: local)"}
_CHUNK = 1 << 20


def download(url: str, dest: Path, *, min_bytes: int = 1024, force: bool = False) -> Path:
    """Stream a URL to disk with caching and an atomic rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes and not force:
        log.info("cached  %-22s %10.1f MB", dest.name, dest.stat().st_size / 1e6)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("get     %s", url)
    with requests.get(url, stream=True, timeout=300, headers=_HEADERS) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(_CHUNK):
                fh.write(chunk)

    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} returned only {size} bytes -- likely an error page")

    # CDC serves an HTML landing page (HTTP 200) for retired URL forms rather
    # than a 404, so a size check alone is not enough for XPT/DAT targets.
    if dest.suffix.lower() in {".xpt", ".dat"}:
        with open(tmp, "rb") as fh:
            head = fh.read(512).lstrip()
        if head[:14].upper().startswith(b"<!DOCTYPE") or head[:5].upper() == b"<HTML":
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{url} served HTML, not data -- URL form is wrong")

    tmp.replace(dest)
    log.info("saved   %-22s %10.1f MB", dest.name, size / 1e6)
    return dest


# pandas.read_sas decodes the SAS transport representation of zero as this
# denormal rather than as 0.0. It is not cosmetic: NHANES codes the survey
# weight of a participant who was interviewed but never MEC-examined as ZERO,
# and 398 of the 10,348 cycle-D records are in that state. A `weight > 0` filter
# -- the obvious way to drop them -- silently KEEPS all 398, because
# 5.4e-79 > 0 is True. They then enter every weighted curve fit carrying
# effectively no weight but inflating the reported n and the minimum-sample
# checks. The same artifact disables non-wear detection in the accelerometer
# data (see wearable._denorm), where the rule keys off "counts == 0".
_XPT_DENORMAL_EPS = 1e-70


def read_xpt(path: Path) -> pd.DataFrame:
    """Read a SAS transport file into a DataFrame with SEQN as a nullable int."""
    df = pd.read_sas(path, format="xport")
    for col in df.columns:
        # pandas decodes XPT character columns as bytes; normalise to str.
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
        elif pd.api.types.is_float_dtype(df[col]):
            df.loc[df[col].abs() < _XPT_DENORMAL_EPS, col] = 0.0
    if C.JOIN_KEY in df.columns:
        df[C.JOIN_KEY] = df[C.JOIN_KEY].astype("Int64")
    return df


def fetch_nhanes(names: list[str] | None = None, *, force: bool = False) -> dict[str, Path]:
    """Download the registered NHANES files (default: everything except PAXRAW)."""
    if names is None:
        names = [n for n, f in C.NHANES_FILES.items() if f.kind != "activity"]
    out: dict[str, Path] = {}
    for name in names:
        spec = C.NHANES_FILES[name]
        try:
            out[name] = download(spec.url, spec.local, min_bytes=2048, force=force)
        except Exception as exc:  # a single missing component must not kill the run
            log.warning("FAILED  %-22s %s", name, exc)
    return out


def fetch_paxraw(*, force: bool = False) -> Path:
    """Download and unzip the 471 MB minute-level accelerometer archive."""
    spec = C.NHANES_FILES["PAXRAW_D"]
    zpath = download(spec.url, spec.local, min_bytes=10_000_000, force=force)

    target_dir = C.RAW / "nhanes" / "paxraw"
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        log.info("zip contains: %s", members)
        inner = max(members, key=lambda m: zf.getinfo(m).file_size)
        out = target_dir / Path(inner).name
        if out.exists() and out.stat().st_size == zf.getinfo(inner).file_size and not force:
            log.info("cached  %s (%.1f GB)", out.name, out.stat().st_size / 1e9)
            return out
        log.info("extracting %s (%.1f GB uncompressed)", inner, zf.getinfo(inner).file_size / 1e9)
        with zf.open(inner) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
    return out


def fetch_mortality(*, force: bool = False) -> Path:
    """Download the Public-Use Linked Mortality File for this cycle."""
    return download(
        f"{C.LMF_BASE}/{C.LMF_FILE}",
        C.RAW / "mortality" / C.LMF_FILE,
        min_bytes=10_000,
        force=force,
    )
