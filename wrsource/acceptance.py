"""Staged acceptance: verify boxes as they arrive and ship certificates early.

An :class:`Acceptor` trusts exactly one root document: a packing manifest for a
frozen ``(source_version, seed)``.  Box seals then arrive independently, in any
order, possibly repeated.  Each seal is checked by the pure verifier in
:mod:`wrsource.packing`; valid seals become *verified boxes*, invalid ones are
kept as anomalies and never enter any selection.

Two shipping permits are supported per group:

* **fast** (快速单): ship as soon as every selected row has a verified body and
  the manifest's exact best key of every still-missing box provably cannot
  overtake the weakest selected row.  When the group has fewer active rows than
  the quota, all of its non-empty boxes must have arrived, so the manifest's
  group row count proves no active row is missing.
* **full** (全收单): ship only after every non-empty box of the group has been
  verified.

A quota of zero is issued immediately without waiting for any body.  All
ranking goes through the exact lottery comparator; float display values are
never consulted.  An issued certificate embeds the bodies it ships, the exact
boundary evidence and the seal digests it was built on, and is frozen: later
anomalous boxes cannot alter it, and :func:`recheck_certificate` re-adjudicates
it from the certificate alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from functools import cmp_to_key
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .packing import (canonical_json, exact_key, key_to_candidate,
                      manifest_digest, sha256_hex, verify_seal)
from .sampling import Candidate, compare_rank, log_u, lottery_a

CERT_FORMAT = "wraccept/1"
FAST = "fast"
FULL = "full"


class AcceptanceError(Exception):
    pass


@dataclass
class _VerifiedBox:
    box_id: str
    seal_digest: str
    rows: List[dict]                 # seal rows, row_id -> details
    row_to_box: Dict[str, str]
    # group -> exact candidates derived from the verified seal
    by_group: Dict[str, List[Candidate]]


@dataclass
class Anomaly:
    box_id: str
    identity: dict
    seal_digest: Optional[str]
    errors: List[str]
    at: float

    def as_dict(self) -> dict:
        return {"box_id": self.box_id, "identity": self.identity,
                "seal_digest": self.seal_digest, "errors": list(self.errors),
                "at": self.at}


def _ranked(cands: Sequence[Candidate]) -> List[Candidate]:
    return sorted(cands, key=cmp_to_key(compare_rank))


class Acceptor:
    """Stateful, per-``(version, seed)`` staged acceptance machine."""

    def __init__(self, manifest: Mapping[str, Any]):
        want = manifest_digest(manifest)
        if want != manifest.get("manifest_digest"):
            raise AcceptanceError("manifest_digest does not match manifest")
        self.manifest = dict(manifest)
        self.version = int(manifest["source_version"])
        self.seed = bytes.fromhex(str(manifest["seed_hex"]))
        if len(self.seed) != 32:
            raise AcceptanceError("manifest seed is not 32 bytes")
        self.boxes: Dict[str, _VerifiedBox] = {}
        self.anomalies: List[Anomaly] = []
        self._anomaly_seen: set = set()
        self.ledger: List[dict] = []

    # ------------------------------------------------------------------
    # Seal intake: independent, out of order, idempotent
    # ------------------------------------------------------------------
    def ingest(self, seal: Mapping[str, Any]) -> dict:
        report = verify_seal(self.manifest, seal)
        box_id = report["box_id"]
        digest = seal.get("seal_digest")
        if not report["ok"]:
            key = (box_id, digest)
            if key not in self._anomaly_seen:
                self._anomaly_seen.add(key)
                self.anomalies.append(Anomaly(
                    box_id, report["identity"], digest,
                    list(report["errors"]), time.time()))
            return {**report, "idempotent": False, "stored": False}

        existing = self.boxes.get(box_id)
        if existing is not None:
            if existing.seal_digest == digest:
                return {**report, "idempotent": True, "stored": True}
            # A different seal for an already verified box is a conflict: keep
            # the original evidence and retain the new one as an anomaly.
            err = ("conflicting seal for already verified box "
                   f"{box_id!r}; retaining seal {existing.seal_digest[:12]}")
            key = (box_id, digest)
            if key not in self._anomaly_seen:
                self._anomaly_seen.add(key)
                self.anomalies.append(Anomaly(
                    box_id, report["identity"], digest, [err], time.time()))
            return {**report, "idempotent": False, "stored": False,
                    "errors": report["errors"] + [err]}

        rows = list(seal["rows"])
        row_to_box = {r["row_id"]: box_id for r in rows}
        by_group: Dict[str, List[Candidate]] = {}
        for r in rows:
            a = lottery_a(self.seed, r["row_id"])
            cand = Candidate(r["row_id"], r["group"], int(r["weight"]),
                             r["body_sha256"], a, log_u(a, int(r["weight"])))
            by_group.setdefault(r["group"], []).append(cand)
        self.boxes[box_id] = _VerifiedBox(
            box_id, str(digest), rows, row_to_box, by_group)
        return {**report, "idempotent": False, "stored": True}

    def verified_box_ids(self) -> List[str]:
        return sorted(self.boxes)

    def pending_boxes(self, group: Optional[str] = None) -> List[dict]:
        out = []
        for entry in self.manifest["boxes"]:
            if entry["box_id"] in self.boxes:
                continue
            if group is None or any(ge["group"] == group
                                    for ge in entry["groups"]):
                out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Group evaluation
    # ------------------------------------------------------------------
    def _group_pool(self, group: str) -> List[Candidate]:
        pool: List[Candidate] = []
        for box in self.boxes.values():
            pool.extend(box.by_group.get(group, ()))
        return _ranked(pool)

    def _row_body(self, row_id: str) -> dict:
        for box in self.boxes.values():
            if row_id in box.row_to_box:
                return next(r for r in box.rows if r["row_id"] == row_id)
        raise KeyError(row_id)  # pragma: no cover - pool rows always exist

    def evaluate(self, kind: str, group: str, quota: int) -> dict:
        """Return a structured verdict without mutating the ledger."""
        if kind not in (FAST, FULL):
            raise AcceptanceError(f"unknown permit kind: {kind!r}")
        if quota < 0:
            raise AcceptanceError(f"negative quota for group {group!r}")
        manifest_entries = [
            ge for entry in self.manifest["boxes"] for ge in entry["groups"]
            if ge["group"] == group]
        total_rows = sum(ge["rows"] for ge in manifest_entries)
        all_boxes = sorted({entry["box_id"] for entry in self.manifest["boxes"]
                            if any(ge["group"] == group for ge in entry["groups"])})
        pending = [b for b in all_boxes if b not in self.boxes]
        verified = [b for b in all_boxes if b in self.boxes]

        if quota == 0:
            return {"shippable": True, "kind": kind, "group": group,
                    "quota": 0, "selected": [], "pending_boxes": pending,
                    "boundary": None,
                    "reason": "quota zero: no body required"}
        if total_rows == 0:
            return {"shippable": True, "kind": kind, "group": group,
                    "quota": quota, "selected": [], "pending_boxes": [],
                    "boundary": None,
                    "reason": "group has no active rows in the manifest"}

        if kind == FULL and pending:
            return {"shippable": False, "kind": kind, "group": group,
                    "quota": quota, "selected": [], "pending_boxes": pending,
                    "boundary": None,
                    "reason": ("full permit waits for "
                               f"{len(pending)} box(es): {pending}")}

        pool = self._group_pool(group)
        selected = pool[:quota]

        if not pending:
            return {"shippable": True, "kind": kind, "group": group,
                    "quota": quota, "selected": selected,
                    "pending_boxes": [], "boundary": None,
                    "reason": ("all non-empty boxes verified" if selected
                               else "no active rows")}

        # Fast permit with missing boxes.
        if len(selected) < quota:
            return {"shippable": False, "kind": kind, "group": group,
                    "quota": quota, "selected": selected,
                    "pending_boxes": pending, "boundary": None,
                    "reason": ("fewer verified rows than quota; cannot prove "
                               "no active row is missing")}

        weakest = selected[-1]
        boundary_checks = []
        blockers = []
        for entry in self.manifest["boxes"]:
            if entry["box_id"] not in pending:
                continue
            ge = next((g for g in entry["groups"] if g["group"] == group), None)
            if ge is None:
                continue
            best = key_to_candidate(ge["best"])
            # cmp < 0 would put the missing box's best ahead of the weakest
            # selected row; cmp == 0 cannot happen for distinct row ids
            # because compare_rank is a total order including the tie-break.
            cmp = compare_rank(best, weakest)
            boundary_checks.append({
                "box_id": entry["box_id"],
                "missing_best": exact_key(best.row_id, best.weight, self.seed),
                "rank_cmp_vs_last_selected": cmp,
                "cannot_overtake": cmp > 0,
            })
            if cmp <= 0:
                blockers.append(entry["box_id"])
        if blockers:
            return {"shippable": False, "kind": kind, "group": group,
                    "quota": quota, "selected": selected,
                    "pending_boxes": pending,
                    "boundary": {"checks": boundary_checks},
                    "reason": ("missing box(es) still hold a candidate that "
                               f"can make the list: {blockers}")}
        return {"shippable": True, "kind": kind, "group": group,
                "quota": quota, "selected": selected,
                "pending_boxes": pending,
                "boundary": {
                    "rule": "every missing box's exact best key ranks strictly "
                            "behind the last selected key",
                    "last_selected": exact_key(weakest.row_id, weakest.weight,
                                               self.seed),
                    "checks": boundary_checks,
                },
                "reason": "boundary evidence excludes every missing box"}

    # ------------------------------------------------------------------
    # Issuance: freeze verified boxes and boundary evidence
    # ------------------------------------------------------------------
    def issue(self, kind: str, group: str, quota: int,
              embed_bodies: bool = True) -> dict:
        verdict = self.evaluate(kind, group, quota)
        if not verdict["shippable"]:
            return {"issued": False, **verdict}
        cert = self._build_certificate(verdict, embed_bodies)
        self.ledger.append({
            "certificate_id": cert["certificate_id"],
            "group": group, "quota": quota, "kind": kind,
            "issued_at": cert["issued_at"],
            "frozen_boxes": cert["frozen_boxes"],
            "boundary": cert["boundary"],
        })
        return {"issued": True, **verdict, "certificate": cert}

    def issue_many(self, kind: str, quotas: Mapping[str, int],
                   embed_bodies: bool = True) -> dict:
        results = {}
        for group in sorted(quotas):
            results[group] = self.issue(kind, group, int(quotas[group]),
                                        embed_bodies)
        return {
            "issued": [g for g, r in results.items() if r["issued"]],
            "blocked": {g: r["reason"] for g, r in results.items()
                        if not r["issued"]},
            "results": results,
        }

    def _build_certificate(self, verdict: dict, embed_bodies: bool) -> dict:
        group = verdict["group"]
        selected_rows = []
        frozen_box_ids = set()
        for rank, cand in enumerate(verdict["selected"], start=1):
            row = self._row_body(cand.row_id)
            box_id = self._box_of(cand.row_id)
            frozen_box_ids.add(box_id)
            selected_rows.append({
                "rank": rank,
                "row_id": cand.row_id,
                "box_id": box_id,
                "weight": cand.weight,
                "hash_hex": cand.key_hex,
                "a": str(cand.a),
                "body_sha256": row["body_sha256"],
                "rev": row["rev"],
                "body": row["body"] if embed_bodies else None,
            })
        # Freeze every verified box that participates in this group, not only
        # the boxes of the winners: their seals were part of the evaluation.
        for entry in self.manifest["boxes"]:
            if entry["box_id"] in self.boxes and any(
                    ge["group"] == group for ge in entry["groups"]):
                frozen_box_ids.add(entry["box_id"])
        frozen_boxes = [{
            "box_id": bid,
            "seal_digest": self.boxes[bid].seal_digest,
            "row_count": len(self.boxes[bid].rows),
        } for bid in sorted(frozen_box_ids)]

        all_boxes = [entry["box_id"] for entry in self.manifest["boxes"]
                     if any(ge["group"] == group for ge in entry["groups"])]
        cert = {
            "format": CERT_FORMAT,
            "kind": verdict["kind"],
            "source_id": self.manifest.get("source_id"),
            "source_version": self.version,
            "seed_hex": self.seed.hex(),
            "manifest_digest": self.manifest["manifest_digest"],
            "issued_at": time.time(),
            "group": group,
            "quota": verdict["quota"],
            "selected": selected_rows,
            "boundary": verdict["boundary"],
            "completeness": {
                "manifest_group_rows": int(
                    self.manifest["group_row_count"].get(group, 0)),
                "nonempty_boxes": sorted(all_boxes),
                "verified_boxes": [b["box_id"] for b in frozen_boxes],
                "pending_boxes": list(verdict["pending_boxes"]),
            },
            "frozen_boxes": frozen_boxes,
        }
        cert["certificate_id"] = sha256_hex(canonical_json(cert))
        return cert

    def _box_of(self, row_id: str) -> str:
        for box in self.boxes.values():
            if row_id in box.row_to_box:
                return box.box_id
        raise KeyError(row_id)  # pragma: no cover

    def anomalies_as_dicts(self) -> List[dict]:
        return [a.as_dict() for a in self.anomalies]


# ---------------------------------------------------------------------------
# Independent re-adjudication from the certificate alone
# ---------------------------------------------------------------------------

def recheck_certificate(cert: Mapping[str, Any]) -> dict:
    """Verify a shipped certificate from its own embedded evidence.

    No centre connection and no live manifest are required: the certificate
    carries the exact keys, bodies, boundary checks and seal digests.  This is
    what a reader runs after carrying the certificate away.
    """
    errors: List[str] = []
    if cert.get("format") != CERT_FORMAT:
        errors.append(f"unknown certificate format: {cert.get('format')!r}")
    seed = bytes.fromhex(str(cert["seed_hex"]))
    group = cert["group"]

    selected = list(cert["selected"])
    cands: List[Candidate] = []
    for rec in selected:
        a = lottery_a(seed, rec["row_id"])
        if format(a - 1, "064x") != rec["hash_hex"]:
            errors.append(f"{rec['row_id']}: hash_hex not derivable from seed")
        if str(a) != str(rec["a"]):
            errors.append(f"{rec['row_id']}: a inconsistent with hash_hex")
        if rec.get("body") is not None:
            if sha256_hex(rec["body"].encode("utf-8")) != rec["body_sha256"]:
                errors.append(f"{rec['row_id']}: body digest mismatch")
        cands.append(Candidate(rec["row_id"], group, int(rec["weight"]),
                               rec["body_sha256"], a,
                               log_u(a, int(rec["weight"]))))

    ranks = [r["rank"] for r in selected]
    if ranks != list(range(1, len(selected) + 1)):
        errors.append("selected ranks are not 1..n")
    ordered = _ranked(cands)
    if [c.row_id for c in ordered] != [r["row_id"] for r in selected]:
        errors.append("selected rows are not in exact lottery order")
    if len(selected) > int(cert["quota"]):
        errors.append("selected exceeds quota")

    boundary = cert.get("boundary")
    if boundary:
        last = ordered[-1]
        for check in boundary["checks"]:
            best = key_to_candidate(check["missing_best"])
            cmp = compare_rank(best, last)
            if cmp != check["rank_cmp_vs_last_selected"]:
                errors.append(
                    f"boundary check for box {check['box_id']} records a "
                    "comparison that exact arithmetic contradicts")
            if cmp <= 0:
                errors.append(
                    f"box {check['box_id']} can still overtake the selection")
            if not check["cannot_overtake"]:
                errors.append(f"box {check['box_id']} flagged as overtake risk")

    frozen = {f["box_id"]: f["seal_digest"] for f in cert["frozen_boxes"]}
    for rec in selected:
        if rec["box_id"] not in frozen:
            errors.append(f"winner box {rec['box_id']} is not in frozen_boxes")
    if cert["kind"] == FULL and cert["completeness"]["pending_boxes"]:
        errors.append("full certificate still lists pending boxes")
    if not selected and int(cert["quota"]) > 0:
        total = cert["completeness"]["manifest_group_rows"]
        if total != 0:
            errors.append("empty selection with positive quota needs proof of "
                          "zero active rows")

    recomputed = {k: v for k, v in cert.items() if k != "certificate_id"}
    if sha256_hex(canonical_json(recomputed)) != cert.get("certificate_id"):
        errors.append("certificate_id does not match certificate contents")
    return {"ok": not errors, "certificate_id": cert.get("certificate_id"),
            "group": group, "selected": [r["row_id"] for r in selected],
            "errors": errors}


def recheck_certificate_against_manifest(cert: Mapping[str, Any],
                                         manifest: Mapping[str, Any]) -> dict:
    """Stricter recheck that also pins the certificate to a live manifest."""
    base = recheck_certificate(cert)
    errors = list(base["errors"])
    if cert.get("manifest_digest") != manifest.get("manifest_digest"):
        errors.append("certificate is bound to a different manifest")
    if cert.get("source_id") != manifest.get("source_id"):
        errors.append("certificate source_id differs from manifest")
    if int(cert["source_version"]) != int(manifest["source_version"]):
        errors.append("certificate source version differs from manifest")
    # Every pending box's best key in the cert must equal the manifest claim.
    pending = set(cert["completeness"]["pending_boxes"])
    for entry in manifest["boxes"]:
        if entry["box_id"] not in pending:
            continue
        ge = next((g for g in entry["groups"]
                   if g["group"] == cert["group"]), None)
        if ge is None:
            continue
        match = any(check["box_id"] == entry["box_id"]
                    and check["missing_best"] == ge["best"]
                    for check in (cert["boundary"] or {}).get("checks", []))
        if not match:
            errors.append(
                f"pending box {entry['box_id']} boundary claim not backed by "
                "the manifest's exact best key")
    return {**base, "ok": not errors, "errors": errors}
