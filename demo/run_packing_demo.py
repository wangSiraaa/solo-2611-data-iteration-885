#!/usr/bin/env python3
"""End-to-end demo of per-version packing, staged acceptance and dispatch.

Run from the repository root:

    python3 demo/run_packing_demo.py

It builds the exact narratives from the task on real source rows:

1. Frozen version 1: zero seed, uniform weight, E>B>A.  Box jia holds B,
   box yi holds A (both bodies intentionally identical).  With jia first,
   the fast permit ships B while the full permit waits for yi.
2. A second frozen source version adds E into yi.  Same-named boxes, same
   bodies, but the version identity differs: the fast permit can no longer
   ship on jia alone, and after yi arrives it ships E.  A seal from the wrong
   version is rejected even though box jia is byte-identical.
3. One truck, three groups whose champions live in X/Y/Z: the 7/4/5 then
   7/4/9 ticket trade-offs, early-arrival replanning, and a physical-only
   rebox that changes the load without publishing a new source version.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from wrsource.acceptance import FAST, FULL, Acceptor, recheck_certificate
from wrsource.dispatch import Ticket, execute_plan, plan_dispatch
from wrsource.packing import build_packing_manifest, make_seal, verify_seal
from wrsource.source import SourceService

SEED = bytes(32)
SHARED_BODY = "same-payload"


def banner(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def make_source(path, with_e):
    svc = SourceService(path)
    svc.init()
    ops = [
        {"op": "add", "row_id": "B", "group": "G1", "weight": 1,
         "body": SHARED_BODY, "shard": "box-jia"},
        {"op": "add", "row_id": "A", "group": "G1", "weight": 1,
         "body": SHARED_BODY, "shard": "box-yi"},
    ]
    if with_e:
        ops.append({"op": "add", "row_id": "E", "group": "G1", "weight": 1,
                    "body": SHARED_BODY, "shard": "box-yi"})
    version = svc.apply_batch(ops, note=f"frozen version (E={with_e})").version
    return svc, version


def demo_acceptance(work):
    banner("1. Version 1: jia=B, yi=A; jia arrives first")
    svc1, v1 = make_source(os.path.join(work, "v1.sqlite"), with_e=False)
    m1 = build_packing_manifest(svc1, v1, SEED, note="v1 packing")
    print("source_id:", m1["source_id"], "| version:", v1)
    print("boxes:", [(b["box_id"], b["row_count"],
                      [(g["group"], g["best"]["row_id"]) for g in b["groups"]])
                     for b in m1["boxes"]])
    acc = Acceptor(m1)
    r = acc.ingest(make_seal(svc1, v1, "box-jia", m1))
    print("ingest box-jia:", "ok" if r["ok"] else r["errors"],
          "| idempotent repeat:", acc.ingest(
              make_seal(svc1, v1, "box-jia", m1))["idempotent"])

    fast = acc.issue(FAST, "G1", 1)
    print("fast permit on jia alone:",
          "SHIP", [c.row_id for c in fast["selected"]]
          if fast["issued"] else "BLOCKED - " + fast["reason"])
    boundary = fast["certificate"]["boundary"]
    print("boundary evidence:",
          [(c["box_id"], c["missing_best"]["row_id"],
            "cannot overtake" if c["cannot_overtake"] else "can overtake")
           for c in boundary["checks"]])
    print("offline recheck of certificate:",
          recheck_certificate(fast["certificate"])["ok"])

    full = acc.issue(FULL, "G1", 1)
    print("full permit on jia alone:",
          "BLOCKED - " + full["reason"] if not full["issued"] else "SHIP")
    acc.ingest(make_seal(svc1, v1, "box-yi", m1))
    full2 = acc.issue(FULL, "G1", 1)
    print("full permit after yi:",
          "SHIP", [r["row_id"] for r in full2["certificate"]["selected"]])

    banner("2. Version 2: E added to yi; same box names, new version identity")
    svc2, v2 = make_source(os.path.join(work, "v2.sqlite"), with_e=True)
    m2 = build_packing_manifest(svc2, v2, SEED, note="v2 packing")
    print("source_id:", m2["source_id"], "| version:", v2)
    acc2 = Acceptor(m2)
    acc2.ingest(make_seal(svc2, v2, "box-jia", m2))
    fast2 = acc2.issue(FAST, "G1", 1)
    print("fast permit on jia alone:",
          "BLOCKED - " + fast2["reason"] if not fast2["issued"] else "SHIP")
    print("missing-box best key:",
          [(c["box_id"], c["missing_best"]["row_id"])
           for c in fast2["boundary"]["checks"]])
    acc2.ingest(make_seal(svc2, v2, "box-yi", m2))
    shipped = acc2.issue(FAST, "G1", 1)
    winner = shipped["certificate"]["selected"][0]
    print("fast permit after yi: SHIP", winner["row_id"],
          "| body identical to B's:", winner["body"] == SHARED_BODY)

    # Cross-version foreign seal: box-jia content is byte-identical.
    foreign = make_seal(svc2, v2, "box-jia", m2)
    cross = verify_seal(m1, foreign)
    print("v2 box-jia seal checked against v1 manifest: ok =", cross["ok"],
          "|", cross["errors"])
    return svc1, v1, m1


def demo_dispatch(svc, version, manifest, work):
    banner("3. One truck: champions of Gx/Gy/Gz in boxes X/Y/Z, volume 1 each")
    rows = []
    champion_box = {}
    for group, home in (("Gx", "X"), ("Gy", "Y"), ("Gz", "Z")):
        from functools import cmp_to_key
        from wrsource.sampling import compare_rank, make_candidate
        cands = [make_candidate(f"{group}-a", group, 1, "h", SEED),
                 make_candidate(f"{group}-b", group, 1, "h", SEED)]
        ranked = sorted(cands, key=cmp_to_key(compare_rank))
        champion_box[group] = home
        rows.append({"op": "add", "row_id": ranked[0].row_id, "group": group,
                     "weight": 1, "body": f"body-{group}-champ",
                     "shard": home})
        rows.append({"op": "add", "row_id": ranked[1].row_id, "group": group,
                     "weight": 1, "body": f"body-{group}-second",
                     "shard": "Q"})
    v = svc.apply_batch(rows, note="dispatch groups").version
    m = build_packing_manifest(svc, v, SEED)
    volumes = {"X": 1, "Y": 1, "Z": 1, "Q": 1}

    def show(title, tickets, capacity, arrived=()):
        plan = plan_dispatch(m, tickets, volumes, capacity,
                             arrived_boxes=list(arrived))
        print(f"\n{title}")
        print("  optimum value:", plan["optimum"]["value"],
              "| tickets:", plan["optimum"]["ticket_ids"],
              "| pickup:", plan["pickup_boxes"],
              "| spare capacity:", plan["spare_capacity"])
        if plan["optimum"]["tied"]:
            print("  tied alternatives:",
                  [a["ticket_ids"] for a in plan["optimum"]["alternatives"]])
        return plan

    t_745 = [Ticket.create("jia", ["Gx", "Gy"], 7),
             Ticket.create("yi", ["Gx"], 4),
             Ticket.create("bing", ["Gz"], 5)]
    show("case A: jia=7(X,Y), yi=4(X), bing=5(Z), capacity 2", t_745, 2)
    t_749 = [Ticket.create("jia", ["Gx", "Gy"], 7),
             Ticket.create("yi", ["Gx"], 4),
             Ticket.create("bing", ["Gz"], 9)]
    plan_b = show("case B: bing raised to 9, capacity 2", t_749, 2)
    plan_c = show("case C: X already at the archive, capacity 2", t_749, 2,
                  arrived=["X"])

    seals = {b["box_id"]: make_seal(svc, v, b["box_id"], m)
             for b in plan_c["load"]}
    result = execute_plan(plan_c, seals, m,
                          {"Gx": 1, "Gy": 1, "Gz": 1})
    print("\nexecution after loading:",
          "earned", result["earned_value"],
          "via", [e["ticket_id"] for e in result["earned_tickets"]],
          "| anomalies:", len(result["anomalies"]))

    banner("4. Physical-only rebox: move Gx champion X -> Z (no new version)")
    gx_champ = next(c["key"]["row_id"] for c in plan_b["champions"]
                    if c["group"] == "Gx")
    svc.apply_batch([{"op": "set_shard", "row_id": gx_champ, "shard": "Z"}])
    print("latest source version still:", svc.latest_version(), "(unchanged)")
    m_rebox = build_packing_manifest(svc, v, SEED)
    plan_d = plan_dispatch(m_rebox, t_749, volumes, 2)
    print("after rebox, optimum value:", plan_d["optimum"]["value"],
          "| pickup:", plan_d["pickup_boxes"],
          "(Gx and Gz champions now share Z, counted once)")


def main():
    work = tempfile.mkdtemp(prefix="packing-demo-")
    try:
        svc, version, manifest = demo_acceptance(work)
        demo_dispatch(svc, version, manifest, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("\ndemo work directory removed; all evidence above is self-contained.")


if __name__ == "__main__":
    main()
