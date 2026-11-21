"""Per-version packing manifests and sealed box contents (offline archives).

A :dfn:`packing manifest` is the archive's attestation for one *frozen*
``(source_version, seed)`` pair.  It says, for every physical box of that
version:

* how many rows of each group the box holds;
* the full *exact* lottery sort key of the box's best candidate per group
  (``row_id`` included, so exact priority ties are still totally ordered);
* a content summary digest covering exactly those claims.

The manifest deliberately carries no bodies.  Bodies arrive later, one box at
a time, as :dfn:`seals`.  A seal lists every row of one box with its version
row number (``rev``), weight and real body, and is verified purely against the
manifest -- no connection to the centre is needed.  Boxes therefore arrive
independently, out of order and repeatedly (verification is idempotent).

Two safety properties run through the whole design:

* **Version identity is part of every document.**  A box name is only unique
  inside one frozen version.  A seal built for another version -- even a
  same-named box whose row count and best key did not change -- is rejected,
  because its identity tuple does not match the manifest.
* **All boundary reasoning uses the exact lottery contract from
  :mod:`wrsource.sampling`.**  The manifest stores exact integer keys, never
  float display values, and every comparison re-enters the exact comparator.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import cmp_to_key
from typing import Dict, List, Mapping, Optional, Sequence

from .sampling import Candidate, compare_rank, log_u, lottery_a

PACKING_FORMAT = "wrpack/1"
SEAL_FORMAT = "wrseal/1"
_JSON_SEP = (",", ":")


class PackingError(Exception):
    """Raised when a manifest or seal cannot be built or verified."""


def canonical_json(obj) -> bytes:
    """Deterministic byte form used for every digest in this module."""
    return json.dumps(obj, sort_keys=True, separators=_JSON_SEP,
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Exact keys
# ---------------------------------------------------------------------------

def exact_key(row_id: str, weight: int, seed: bytes) -> Dict[str, object]:
    """The full precise sort key of a row (tie-break row_id included)."""
    a = lottery_a(seed, row_id)
    return {
        "row_id": row_id,
        "weight": int(weight),
        "hash_hex": format(a - 1, "064x"),
        "a": str(a),
    }


def key_to_candidate(key: Mapping[str, object]) -> Candidate:
    """Rebuild an exact :class:`Candidate` from a manifest sort key.

    The reconstructed candidate feeds the same exact comparator as a live
    candidate.  Its ``lkey`` is a locally recomputed screen, never a value
    trusted from the wire.
    """
    a = int(str(key["a"]))
    weight = int(key["weight"])
    if format(a - 1, "064x") != key["hash_hex"]:
        raise PackingError(f"hash_hex inconsistent with a for {key['row_id']!r}")
    return Candidate(str(key["row_id"]), "", weight, "", a, log_u(a, weight))


def key_to_candidate(key: Mapping[str, object]) -> Candidate:
    """Rebuild an exact :class:`Candidate` from a manifest sort key.

    The reconstructed candidate feeds the same exact comparator as a live
    candidate.  Its ``lkey`` is a locally recomputed screen, never a value
    trusted from the wire.
    """
    a = int(str(key["a"]))
    weight = int(key["weight"])
    if format(a - 1, "064x") != key["hash_hex"]:
        raise PackingError(f"hash_hex inconsistent with a for {key['row_id']!r}")
    return Candidate(str(key["row_id"]), "", weight, "", a, log_u(a, weight))


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def _box_summary(box_id: str, row_count: int,
                 groups: Sequence[Mapping[str, object]]) -> str:
    payload = {
        "box_id": box_id,
        "row_count": row_count,
        "groups": sorted((dict(g) for g in groups),
                         key=lambda g: str(g["group"])),
    }
    return sha256_hex(canonical_json(payload))


def build_packing_manifest(service, version: int, seed: bytes,
                           box_map: Optional[Mapping[str, str]] = None,
                           note: Optional[str] = None) -> dict:
    """Attest the box packing of a frozen source version.

    ``box_map`` explicitly assigns ``row_id -> box_id``.  When omitted, the
    current physical shard of each row is its box, matching the source rule
    that re-sharding only changes physical location.  Every active row of the
    version must belong to exactly one non-empty box.
    """
    service.get_version(version)  # raises VersionError on unknown versions
    meta = service.all_rows_meta(version)
    if len(seed) != 32:
        raise PackingError("seed must be exactly 32 bytes")
    source_id = service.source_id()

    boxes: Dict[str, List[dict]] = {}
    group_row_count: Dict[str, int] = {}
    for row in meta:
        rid = row["row_id"]
        box_id = box_map[rid] if box_map is not None else row["shard"]
        if not isinstance(box_id, str) or not box_id:
            raise PackingError(f"bad box assignment for {rid!r}: {box_id!r}")
        boxes.setdefault(box_id, []).append(row)
        group_row_count[row["group"]] = group_row_count.get(row["group"], 0) + 1

    if box_map is not None:
        assigned = set(box_map)
        active = {r["row_id"] for r in meta}
        unknown = assigned - active
        if unknown:
            raise PackingError(
                f"box_map assigns rows absent from version {version}: "
                f"{sorted(unknown)[:5]}")

    manifest_boxes = []
    for box_id in sorted(boxes):
        rows = boxes[box_id]
        per_group: Dict[str, List[dict]] = {}
        for row in rows:
            per_group.setdefault(row["group"], []).append(row)
        group_entries = []
        for group in sorted(per_group):
            grows = per_group[group]
            cands = [
                Candidate(r["row_id"], group, r["weight"], r["body_sha256"],
                          lottery_a(seed, r["row_id"]),
                          log_u(lottery_a(seed, r["row_id"]), r["weight"]))
                for r in grows
            ]
            best = sorted(cands, key=cmp_to_key(compare_rank))[0]
            group_entries.append({
                "group": group,
                "rows": len(grows),
                "best": exact_key(best.row_id, best.weight, seed),
            })
        entry = {
            "box_id": box_id,
            "row_count": len(rows),
            "groups": group_entries,
        }
        entry["summary"] = _box_summary(box_id, len(rows), group_entries)
        manifest_boxes.append(entry)

    manifest = {
        "format": PACKING_FORMAT,
        "source_id": source_id,
        "source_version": int(version),
        "seed_hex": seed.hex(),
        "created_at": time.time(),
        "note": note,
        "row_count": len(meta),
        "group_row_count": dict(sorted(group_row_count.items())),
        "boxes": manifest_boxes,
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return manifest


def manifest_digest(manifest: Mapping[str, object]) -> str:
    payload = {k: v for k, v in manifest.items()
               if k not in ("manifest_digest",)}
    return sha256_hex(canonical_json(payload))


def box_entry(manifest: Mapping[str, object], box_id: str
              ) -> Optional[dict]:
    for entry in manifest["boxes"]:  # type: ignore[index]
        if entry["box_id"] == box_id:
            return entry
    return None


def boxes_for_group(manifest: Mapping[str, object], group: str) -> List[dict]:
    out = []
    for entry in manifest["boxes"]:  # type: ignore[index]
        for ge in entry["groups"]:
            if ge["group"] == group:
                out.append(entry)
                break
    return out


# ---------------------------------------------------------------------------
# Seal construction (source side) and pure verification (reader side)
# ---------------------------------------------------------------------------

def _seal_digest(envelope: Mapping[str, object]) -> str:
    payload = {k: v for k, v in envelope.items() if k != "seal_digest"}
    return sha256_hex(canonical_json(payload))


def make_seal(service, version: int, box_id: str,
              manifest: Mapping[str, object]) -> dict:
    """Build the seal of one box from the source API.

    The seal is cross-checked against the manifest entry before it leaves the
    source side, so a caller cannot accidentally ship a seal that already
    contradicts the attestation.
    """
    if int(manifest["source_version"]) != int(version):
        raise PackingError("manifest is bound to a different source version")
    seed = bytes.fromhex(str(manifest["seed_hex"]))
    if service.source_id() != manifest.get("source_id"):
        raise PackingError("manifest belongs to a different source archive")
    entry = box_entry(manifest, box_id)
    if entry is None:
        raise PackingError(f"box {box_id!r} is not in the manifest")

    shard_map = service.shard_map(version)
    row_ids = sorted(rid for rid, shard in shard_map.items() if shard == box_id)
    if len(row_ids) != entry["row_count"]:
        raise PackingError(
            f"physical rows in box {box_id!r} ({len(row_ids)}) disagree with "
            f"manifest row_count {entry['row_count']}")
    rows = service.get_rows(version, row_ids)
    if set(rows) != set(row_ids):
        raise PackingError(f"box {box_id!r}: source lost rows of the version")

    seal_rows = []
    for rid in row_ids:
        row = rows[rid]
        body = row["body"]
        digest = sha256_hex(body.encode("utf-8"))
        if digest != row["body_sha256"]:
            raise PackingError(f"body digest mismatch at source for {rid!r}")
        seal_rows.append({
            "row_id": rid,
            "group": row["group"],
            "weight": int(row["weight"]),
            "rev": int(row["rev"]),
            "body_sha256": digest,
            "body": body,
        })
    envelope = {
        "format": SEAL_FORMAT,
        "source_id": str(manifest["source_id"]),
        "source_version": int(version),
        "seed_hex": seed.hex(),
        "box_id": box_id,
        "rows": seal_rows,
    }
    envelope["seal_digest"] = _seal_digest(envelope)
    report = verify_seal(manifest, envelope)
    if not report["ok"]:
        raise PackingError(
            f"fresh seal for {box_id!r} fails its own manifest: {report['errors']}")
    return envelope


def seal_identity(seal: Mapping[str, object]) -> dict:
    """The version-scoped identity of a seal (box names are not global)."""
    return {
        "source_id": str(seal.get("source_id", "")),
        "source_version": int(seal["source_version"]),  # type: ignore[arg-type]
        "seed_hex": str(seal["seed_hex"]),
        "box_id": str(seal["box_id"]),
    }


def verify_seal(manifest: Mapping[str, object], seal: Mapping[str, object]
                ) -> dict:
    """Pure verification of a seal against a manifest (no I/O, no centre).

    Returns ``{"ok", "box_id", "identity", "rows", "errors"}``.  Every failure
    is collected rather than raised, so an acceptor can retain the concrete
    anomaly while continuing to process other boxes.
    """
    box_id = str(seal.get("box_id", ""))
    identity = seal_identity(seal) if "seed_hex" in seal else {
        "source_version": seal.get("source_version"),
        "seed_hex": seal.get("seed_hex"), "box_id": box_id}
    errors: List[str] = []

    if seal.get("format") != SEAL_FORMAT:
        errors.append(f"unknown seal format: {seal.get('format')!r}")
    # Archive- and version-scoped identity: a same-named box in another
    # archive or version is a foreign document even when counts and best keys
    # happen to be identical.
    if str(seal.get("source_id")) != str(manifest.get("source_id")):
        errors.append(
            f"seal source_id {seal.get('source_id')!r} != manifest "
            f"{manifest.get('source_id')!r}")
    if int(seal.get("source_version", -1)) != int(manifest["source_version"]):
        errors.append(
            f"seal version {seal.get('source_version')} != manifest version "
            f"{manifest['source_version']}")
    if str(seal.get("seed_hex")) != str(manifest["seed_hex"]):
        errors.append("seal seed does not match manifest seed")

    want_digest = seal.get("seal_digest")
    if want_digest != _seal_digest(seal):
        errors.append("seal_digest does not match seal contents")

    entry = box_entry(manifest, box_id)
    if entry is None:
        errors.append(f"box {box_id!r} is not attested by the manifest")
        return {"ok": False, "box_id": box_id, "identity": identity,
                "rows": 0, "errors": errors}

    seal_rows = seal.get("rows")
    if not isinstance(seal_rows, list):
        errors.append("seal rows must be a list")
        return {"ok": False, "box_id": box_id, "identity": identity,
                "rows": 0, "errors": errors}

    if len(seal_rows) != entry["row_count"]:
        errors.append(
            f"row count {len(seal_rows)} != manifest {entry['row_count']}")

    seen = set()
    got_counts: Dict[str, int] = {}
    got_best: Dict[str, Candidate] = {}
    seed = bytes.fromhex(str(manifest["seed_hex"]))
    for i, row in enumerate(seal_rows):
        rid = row.get("row_id")
        if not isinstance(rid, str) or not rid:
            errors.append(f"row {i}: bad row_id")
            continue
        if rid in seen:
            errors.append(f"duplicate row in seal: {rid!r}")
        seen.add(rid)
        group = row.get("group")
        weight = row.get("weight")
        got_counts[group] = got_counts.get(group, 0) + 1

        body = row.get("body", "")
        if not isinstance(body, str):
            errors.append(f"{rid}: body must be a string")
            body = ""
        digest = sha256_hex(body.encode("utf-8"))
        if digest != row.get("body_sha256"):
            errors.append(f"{rid}: body_sha256 does not match body bytes")

        # Re-key from the lottery contract instead of trusting seal numbers:
        # the exact key below is rebuilt locally from (seed, row_id, weight).
        if not isinstance(weight, int) or isinstance(weight, bool) \
                or not 1 <= weight <= 100:
            errors.append(f"{rid}: weight out of range: {weight!r}")
            continue
        a = lottery_a(seed, rid)
        cand = Candidate(rid, group, weight, digest, a, log_u(a, weight))
        prev = got_best.get(group)
        if prev is None or compare_rank(cand, prev) < 0:
            got_best[group] = cand

    want_counts = {g["group"]: g["rows"] for g in entry["groups"]}
    if got_counts != want_counts:
        errors.append(
            f"per-group row counts {got_counts} != manifest {want_counts}")

    for ge in entry["groups"]:
        want_best = ge["best"]
        got = got_best.get(ge["group"])
        if got is None:
            errors.append(f"group {ge['group']!r}: no rows present in seal")
            continue
        got_key = exact_key(got.row_id, got.weight, seed)
        if got_key != want_best:
            errors.append(
                f"group {ge['group']!r}: best key {got_key['row_id']} != "
                f"manifest best {want_best['row_id']}")

    if _box_summary(box_id, len(seal_rows), [
        {"group": g, "rows": got_counts.get(g, 0),
         "best": (exact_key(got_best[g].row_id, got_best[g].weight, seed)
                  if g in got_best else None)}
        for g in sorted(set(got_counts) | set(want_counts))
    ]) != entry["summary"]:
        errors.append("box content summary digest does not match manifest")

    return {"ok": not errors, "box_id": box_id, "identity": identity,
            "rows": len(seal_rows), "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    import argparse
    from .source import SourceService

    parser = argparse.ArgumentParser(
        prog="python -m wrsource.packing",
        description="Per-version packing manifests and box seals.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest", help="build a packing manifest")
    p.add_argument("--db", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--seed-hex", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--note")

    p = sub.add_parser("seal", help="seal one box from the source API")
    p.add_argument("--db", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--box", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("verify", help="purely verify a seal against a manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--seal", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "manifest":
        seed = bytes.fromhex(args.seed_hex)
        if len(seed) != 32:
            raise SystemExit("seed must be 32 bytes (64 hex characters)")
        svc = SourceService(args.db)
        manifest = build_packing_manifest(svc, args.version, seed,
                                          note=args.note)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, sort_keys=True,
                      indent=2)
        print(json.dumps({"source_id": manifest["source_id"],
                          "version": manifest["source_version"],
                          "boxes": len(manifest["boxes"]),
                          "rows": manifest["row_count"],
                          "manifest_digest": manifest["manifest_digest"],
                          "out": args.out}, indent=2))
    elif args.cmd == "seal":
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
        svc = SourceService(args.db)
        envelope = make_seal(svc, int(manifest["source_version"]),
                             args.box, manifest)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(envelope, fh, ensure_ascii=False, sort_keys=True,
                      indent=2)
        print(json.dumps({"box_id": args.box, "rows": len(envelope["rows"]),
                          "seal_digest": envelope["seal_digest"],
                          "out": args.out}, indent=2))
    else:
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
        with open(args.seal, encoding="utf-8") as fh:
            envelope = json.load(fh)
        report = verify_seal(manifest, envelope)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
