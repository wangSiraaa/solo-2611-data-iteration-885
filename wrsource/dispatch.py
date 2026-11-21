"""One truck per day: maximise rush-ticket value over un-arrived boxes.

Each rush ticket is placed against one frozen ``(source_version, seed)`` and
names one or more *non-empty* groups; it pays its full value only if the
champion (rank-1 row) of every named group can be delivered.  A ticket is
therefore an all-or-nothing bundle.  The manifest gives, for every group, the
champion's exact key and the box holding it, so the whole plan is computed
from the manifest plus box volumes -- bodies are not needed until the boxes
physically arrive.

Constraints and guarantees:

* one box occupies truck capacity once, however many tickets need it;
* boxes already at the archive occupy no capacity on this trip;
* the truck must fit (capacity is a hard constraint) but need not be full;
* the optimum is found by exact enumeration over ticket subsets, which is
  trivially fast for the usual ~10 tickets even with hundreds of pending
  boxes; tied optima are returned as alternatives, never hidden;
* the output is a guaranteed ticket list plus a load list, computed *before*
  departure; if some boxes arrive early, replanning against the new arrived
  set changes only the load, never the frozen manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .acceptance import FAST, Acceptor
from .packing import key_to_candidate
from .sampling import compare_rank

DISPATCH_FORMAT = "wrdispatch/1"


class DispatchError(Exception):
    pass


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    groups: frozenset
    value: int

    @staticmethod
    def create(ticket_id: str, groups: Sequence[str], value: int) -> "Ticket":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DispatchError(f"ticket {ticket_id!r}: value must be >= 0")
        gs = frozenset(groups)
        if not gs:
            raise DispatchError(f"ticket {ticket_id!r}: needs >= 1 group")
        return Ticket(ticket_id, gs, int(value))


def group_champions(manifest: Mapping[str, Any]) -> Dict[str, dict]:
    """Exact rank-1 key and home box for every non-empty group."""
    best: Dict[str, dict] = {}
    for entry in manifest["boxes"]:
        for ge in entry["groups"]:
            cand = key_to_candidate(ge["best"])
            if ge["group"] not in best or compare_rank(
                    cand, key_to_candidate(best[ge["group"]]["key"])) < 0:
                best[ge["group"]] = {
                    "group": ge["group"],
                    "box_id": entry["box_id"],
                    "key": ge["best"],
                }
    return best


def plan_dispatch(manifest: Mapping[str, Any], tickets: Sequence[Ticket],
                  volumes: Mapping[str, int], capacity: int,
                  arrived_boxes: Optional[Sequence[str]] = None) -> dict:
    """Return the value-maximising feasible ticket bundle and its load."""
    champions = group_champions(manifest)
    arrived = set(arrived_boxes or ())
    # Only boxes backing a group named by some ticket can ever be loaded;
    # boxes of unrelated groups never enter a plan and need no volume.
    ticket_groups = set()
    for t in tickets:
        ticket_groups.update(t.groups)
    relevant_boxes = {champions[g]["box_id"] for g in ticket_groups
                      if g in champions}
    unknown_vols = sorted(b for b in relevant_boxes
                          if b not in volumes and b not in arrived)
    if unknown_vols:
        raise DispatchError(
            f"missing volumes for boxes: {unknown_vols}")

    # Pre-resolve each ticket to its required boxes; tickets naming an empty
    # group or requiring an over-capacity box can never earn.
    resolved: List[dict] = []
    for t in tickets:
        missing_groups = sorted(t.groups - champions.keys())
        if missing_groups:
            resolved.append({"ticket": t, "feasible": False,
                             "reason": f"empty/unknown groups {missing_groups}",
                             "boxes": frozenset()})
            continue
        boxes = frozenset(champions[g]["box_id"] for g in sorted(t.groups))
        pickup = boxes - arrived
        needed_vol = sum(int(volumes[b]) for b in pickup)
        ok = needed_vol <= capacity
        resolved.append({"ticket": t, "feasible": ok,
                         "reason": None if ok else
                         f"pickup volume {needed_vol} > capacity {capacity}",
                         "boxes": boxes})

    eligible = [r for r in resolved if r["feasible"]]
    n = len(eligible)

    def bundle_value(indices) -> int:
        return sum(eligible[i]["ticket"].value for i in indices)

    def bundle_boxes(indices) -> frozenset:
        out: frozenset = frozenset()
        for i in indices:
            out |= eligible[i]["boxes"]
        return out

    best_value = -1
    best_indices: List[tuple] = []
    # Exact subset enumeration.  At ~10 tickets this is 2**10 ~= 1k bundles;
    # every ticket count in the problem is small by construction, so enumerate
    # all feasible subsets and keep every strict optimum (ties included).
    for size in range(n, -1, -1):
        for combo in combinations(range(n), size):
            boxes = bundle_boxes(combo)
            pickup = boxes - arrived
            if sum(int(volumes[b]) for b in pickup) > capacity:
                continue
            value = bundle_value(combo)
            if value > best_value:
                best_value = value
                best_indices = [combo]
            elif value == best_value:
                best_indices.append(combo)

    alternatives = []
    chosen = best_indices[0] if best_indices else tuple()
    for combo in best_indices:
        box_set = bundle_boxes(combo)
        alternatives.append({
            "ticket_ids": sorted(eligible[i]["ticket"].ticket_id
                                 for i in combo),
            "value": bundle_value(combo),
            "load_boxes": sorted(box_set),
        })

    load_boxes = bundle_boxes(chosen)
    pickup_boxes = sorted(load_boxes - arrived)
    load = []
    for bid in sorted(load_boxes):
        load.append({
            "box_id": bid,
            "volume": int(volumes.get(bid, 0)),
            "state": "arrived" if bid in arrived else "pickup",
        })
    used = sum(item["volume"] for item in load if item["state"] == "pickup")

    plan = {
        "format": DISPATCH_FORMAT,
        "source_id": manifest.get("source_id"),
        "source_version": int(manifest["source_version"]),
        "seed_hex": str(manifest["seed_hex"]),
        "capacity": int(capacity),
        "arrived_boxes": sorted(arrived),
        "champions": sorted(
            ({"group": g, **champions[g]} for g in champions),
            key=lambda c: c["group"]),
        "tickets": [
            {"ticket_id": r["ticket"].ticket_id,
             "groups": sorted(r["ticket"].groups),
             "value": r["ticket"].value,
             "required_boxes": sorted(r["boxes"]),
             "feasible": r["feasible"], "reason": r["reason"]}
            for r in resolved],
        "optimum": {
            "value": max(best_value, 0),
            "ticket_ids": alternatives[0]["ticket_ids"] if alternatives else [],
            "alternatives": alternatives,
            "tied": len(alternatives) > 1,
        },
        "load": load,
        "pickup_boxes": pickup_boxes,
        "used_capacity": used,
        "spare_capacity": int(capacity) - used,
    }
    return plan


def execute_plan(plan: Mapping[str, Any], seals: Mapping[str, Mapping[str, Any]],
                 manifest: Mapping[str, Any], quotas: Mapping[str, int]
                 ) -> dict:
    """After loading: verify each box and issue tickets against champion boxes.

    ``seals`` maps box_id -> seal envelope for every box the load touched
    (arrived boxes included).  A rush ticket promises one champion row per
    named group, so each group is issued a **fast** permit after its champion's
    home box has been verified with a real body; the manifest's boundary
    evidence still excludes every box left behind.  A ticket earns only if all
    of its groups certify.
    """
    acceptor = Acceptor(manifest)
    intake = []
    for item in plan["load"]:
        bid = item["box_id"]
        seal = seals.get(bid)
        if seal is None:
            intake.append({"box_id": bid, "ok": False,
                           "errors": ["seal missing at execution"]})
            continue
        report = acceptor.ingest(seal)
        intake.append({"box_id": bid, "ok": report["ok"],
                       "errors": report["errors"]})

    earned, failed, certificates = [], [], {}
    wanted_groups = set()
    for t in plan["tickets"]:
        if t["ticket_id"] in plan["optimum"]["ticket_ids"]:
            wanted_groups.update(t["groups"])
    for group in sorted(wanted_groups):
        quota = int(quotas.get(group, 0))
        result = acceptor.issue(FAST, group, quota)
        if result["issued"]:
            certificates[group] = result["certificate"]
        else:
            failed.append({"group": group, "reason": result["reason"]})

    for t in plan["tickets"]:
        if t["ticket_id"] not in plan["optimum"]["ticket_ids"]:
            continue
        if all(g in certificates for g in t["groups"]):
            earned.append({"ticket_id": t["ticket_id"], "value": t["value"]})
        else:
            failed.append({"ticket_id": t["ticket_id"],
                           "reason": "one or more groups not fully certified"})

    return {
        "plan_value": plan["optimum"]["value"],
        "earned_tickets": earned,
        "earned_value": sum(e["value"] for e in earned),
        "failures": failed,
        "box_intake": intake,
        "anomalies": acceptor.anomalies_as_dicts(),
        "certificates": certificates,
    }
