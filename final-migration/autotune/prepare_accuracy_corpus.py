#!/usr/bin/env python3
"""Resolve, acquire, and deterministically sample KataGo public training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_INDEX_URL = "https://katagoarchive.org/kata1/trainingdata/index.html"
DEFAULT_SEED = 20260803
DEFAULT_SAMPLES = 8192
ARCHIVE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})npzs\.tgz$")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "katago-autotune-sdk/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def safe_extract_training_archive(archive: pathlib.Path, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as payload:
        members = payload.getmembers()
        for member in members:
            relative = pathlib.PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe path in training archive: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"unsupported member in training archive: {member.name}")
        payload.extractall(output, members=members, filter="data")


def validate_corpus(
    corpus: pathlib.Path,
    manifest_path: pathlib.Path,
    *,
    expected_archive: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_url: str | None = None,
    expected_samples: int = DEFAULT_SAMPLES,
    expected_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if not corpus.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("the corpus and its manifest must both exist")
    manifest = json.loads(manifest_path.read_text())
    checks = {
        "num_samples": manifest.get("num_samples") == expected_samples,
        "seed": manifest.get("seed") == expected_seed,
        "output_name": manifest.get("output_npz") == corpus.name,
        "output_sha256": manifest.get("output_npz_sha256") == sha256_file(corpus),
        "source_archive_name": bool(
            ARCHIVE_NAME_RE.fullmatch(str(manifest.get("source_archive", "")))
        ),
        "source_archive_sha256": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_archive_sha256", "")))
        ),
    }
    if expected_archive is not None:
        checks["expected_archive"] = manifest.get("source_archive") == expected_archive
    if expected_archive_sha256 is not None:
        checks["expected_archive_sha256"] = (
            manifest.get("source_archive_sha256") == expected_archive_sha256
        )
    if expected_url is not None:
        checks["expected_url"] = manifest.get("source_archive_url") == expected_url
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"accuracy corpus validation failed: {', '.join(failed)}")
    return manifest


def write_result(
    path: pathlib.Path | None,
    *,
    corpus: pathlib.Path,
    manifest: pathlib.Path,
    source_archive: str,
    source_url: str,
    reused: bool,
) -> None:
    result = {
        "schema": 1,
        "corpus": str(corpus.resolve()),
        "manifest": str(manifest.resolve()),
        "corpus_sha256": sha256_file(corpus),
        "source_archive": source_archive,
        "source_url": source_url,
        "reused": reused,
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--python", type=pathlib.Path, default=pathlib.Path(sys.executable))
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--archive-cache-dir", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--archive", type=pathlib.Path)
    parser.add_argument("--archive-url")
    parser.add_argument("--archive-sha256")
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--corpus", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--result-json", type=pathlib.Path)
    args = parser.parse_args()

    if (args.corpus is None) != (args.manifest is None):
        parser.error("--corpus and --manifest must be supplied together")
    if args.num_samples != DEFAULT_SAMPLES:
        parser.error(f"the release correctness gate requires exactly {DEFAULT_SAMPLES} rows")

    expected_archive: str | None = None
    if args.archive_url:
        expected_archive = pathlib.PurePosixPath(
            urllib.parse.urlparse(args.archive_url).path
        ).name
    elif args.archive is not None:
        expected_archive = args.archive.name

    # A release normally carries the already sampled pair. Validate it before
    # any network access so extraction remains fully offline. Production setup
    # and packaging rebuild only from the URL and hashes in corpus.lock.sh.
    if args.corpus is not None:
        try:
            manifest = validate_corpus(
                args.corpus,
                args.manifest,
                expected_archive=expected_archive,
                expected_archive_sha256=args.archive_sha256,
                expected_url=args.archive_url,
                expected_samples=args.num_samples,
                expected_seed=args.seed,
            )
        except FileNotFoundError:
            pass
        else:
            source_archive = str(manifest["source_archive"])
            source_url = str(
                manifest.get("source_archive_url")
                or urllib.parse.urljoin(args.index_url, source_archive)
            )
            write_result(
                args.result_json,
                corpus=args.corpus,
                manifest=args.manifest,
                source_archive=source_archive,
                source_url=source_url,
                reused=True,
            )
            return 0

    if args.archive_url:
        source_url = args.archive_url
        archive_name = pathlib.PurePosixPath(urllib.parse.urlparse(source_url).path).name
    elif args.archive is not None:
        archive_name = args.archive.name
        source_url = urllib.parse.urljoin(args.index_url, archive_name)
    else:
        parser.error("reconstruction requires --archive-url or --archive")
    if not ARCHIVE_NAME_RE.fullmatch(archive_name):
        raise ValueError(f"invalid KataGo training archive name: {archive_name}")

    date = ARCHIVE_NAME_RE.fullmatch(archive_name).group(1)  # type: ignore[union-attr]
    output_dir = args.output_dir.resolve()
    corpus = (args.corpus or output_dir / f"{date}-19x19-{args.num_samples}-seed{args.seed}-full19.npz").resolve()
    manifest_path = (
        args.manifest
        or output_dir / f"{date}-19x19-{args.num_samples}-seed{args.seed}-full19.manifest.json"
    ).resolve()
    if corpus.is_file() and manifest_path.is_file():
        validate_corpus(
            corpus,
            manifest_path,
            expected_archive=archive_name,
            expected_archive_sha256=args.archive_sha256,
            expected_url=source_url,
            expected_samples=args.num_samples,
            expected_seed=args.seed,
        )
        write_result(
            args.result_json,
            corpus=corpus,
            manifest=manifest_path,
            source_archive=archive_name,
            source_url=source_url,
            reused=True,
        )
        return 0

    archive: pathlib.Path | None = args.archive.resolve() if args.archive else None
    if archive is not None and archive.name != archive_name:
        raise ValueError(f"local archive {archive.name} does not match resolved {archive_name}")
    if archive is None:
        for cache_dir in args.archive_cache_dir:
            candidate = cache_dir.resolve() / archive_name
            if candidate.is_file():
                archive = candidate
                break
    if archive is None:
        archive = args.work_dir.resolve() / "downloads" / archive_name
        if not archive.is_file():
            print(f"[accuracy-corpus] downloading {source_url}", flush=True)
            download(source_url, archive)
    if args.archive_sha256 is not None:
        actual_archive_sha256 = sha256_file(archive)
        if actual_archive_sha256 != args.archive_sha256:
            raise ValueError(
                f"training archive SHA-256 mismatch: {actual_archive_sha256} "
                f"!= {args.archive_sha256}"
            )

    work_dir = args.work_dir.resolve()
    extracted = work_dir / "extracted" / archive_name.removesuffix(".tgz")
    if extracted.exists():
        shutil.rmtree(extracted)
    print(f"[accuracy-corpus] extracting {archive}", flush=True)
    safe_extract_training_archive(archive, extracted)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python.resolve()),
        str((args.repo.resolve() / "python/sample_accuracy_corpus.py")),
        "--input-dir", str(extracted),
        "--source-archive", str(archive),
        "--num-samples", str(args.num_samples),
        "--seed", str(args.seed),
        "--output-npz", str(corpus),
        "--manifest-json", str(manifest_path),
    ]
    print("[accuracy-corpus] sampling exactly 8192 full-board rows", flush=True)
    subprocess.run(command, check=True)
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "source_archive_url": source_url,
        "source_archive_index_url": args.index_url,
        "source_resolution": "frozen-release-source",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_corpus(
        corpus,
        manifest_path,
        expected_archive=archive_name,
        expected_archive_sha256=args.archive_sha256,
        expected_url=source_url,
        expected_samples=args.num_samples,
        expected_seed=args.seed,
    )
    if not args.keep_extracted:
        shutil.rmtree(extracted)
    write_result(
        args.result_json,
        corpus=corpus,
        manifest=manifest_path,
        source_archive=archive_name,
        source_url=source_url,
        reused=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
