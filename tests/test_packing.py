import os
import shutil
import tempfile
import unittest

from wrsource.acceptance import (FAST, FULL, Acceptor,
                                 recheck_certificate,
                                 recheck_certificate_against_manifest)
from wrsource.dispatch import Ticket, execute_plan, group_champions, plan_dispatch
from wrsource.packing import (build_packing_manifest, make_seal,
                              manifest_digest, verify_seal)
from wrsource.source import SourceService

ZERO_SEED = bytes(32)


def add(row_id, group, box, body=None, weight=1):
    return {"op": "add", "row_id": row_id, "group": group, "weight": weight,
            "body": body if body is not None else f"body-of-{row_id}",
            "shard": box}


class PackingBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.svc = SourceService(os.path.join(self.dir, "src.sqlite"))
        self.svc.init()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def seal(self, manifest, box_id, svc=None):
        return make_seal(svc or self.svc, manifest["source_version"],
                         box_id, manifest)

    def _second_source(self, with_e):
        """A second frozen source version, same box names as the first."""
        svc2 = SourceService(os.path.join(self.dir, "src2.sqlite"))
        svc2.init()
        shared = "same-payload"
        ops = [add("B", "G1", "box-jia", body=shared),
               add("A", "G1", "box-yi", body=shared)]
        if with_e:
            ops.append(add("E", "G1", "box-yi", body=shared))
        return svc2, svc2.apply_batch(ops).version


class ManifestAndSealTest(PackingBase):
    def _two_box_version(self, with_e=False):
        # Zero seed, uniform weight: E > B > A.
        shared_body = "same-payload"
        ops = [
            add("B", "G1", "box-jia", body=shared_body),
            add("A", "G1", "box-yi", body=shared_body),
        ]
        if with_e:
            ops.append(add("E", "G1", "box-yi", body=shared_body))
        return self.svc.apply_batch(ops).version

    def test_manifest_attests_counts_and_exact_best_key(self):
        v = self._two_box_version()
        m = build_packing_manifest(self.svc, v, ZERO_SEED, note="n")
        self.assertEqual(m["format"], "wrpack/1")
        self.assertEqual(m["manifest_digest"], manifest_digest(m))
        self.assertEqual(m["row_count"], 2)
        jia = next(b for b in m["boxes"] if b["box_id"] == "box-jia")
        self.assertEqual(jia["row_count"], 1)
        self.assertEqual(jia["groups"][0]["group"], "G1")
        self.assertEqual(jia["groups"][0]["rows"], 1)
        self.assertEqual(jia["groups"][0]["best"]["row_id"], "B")
        yi = next(b for b in m["boxes"] if b["box_id"] == "box-yi")
        self.assertEqual(yi["groups"][0]["best"]["row_id"], "A")

    def test_seal_roundtrip_and_idempotent_verification(self):
        v = self._two_box_version()
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        seal = self.seal(m, "box-jia")
        report = verify_seal(m, seal)
        self.assertTrue(report["ok"], report["errors"])
        # Pure verification is deterministic and idempotent.
        self.assertTrue(verify_seal(m, seal)["ok"])
        self.assertEqual(seal["rows"][0]["row_id"], "B")
        self.assertEqual(seal["rows"][0]["rev"], 1)

    def test_tampered_body_and_summary_mismatch_are_detected(self):
        v = self._two_box_version()
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        seal = self.seal(m, "box-jia")
        bad = json_deepcopy(seal)
        bad["rows"][0]["body"] = "tampered"
        bad["seal_digest"] = "x"  # force digest recomputation path too
        report = verify_seal(m, bad)
        self.assertFalse(report["ok"])
        self.assertTrue(any("seal_digest" in e for e in report["errors"]))
        self.assertTrue(any("body_sha256" in e for e in report["errors"]))

        # Recompute digest so only the content-summary claim fails.
        from wrsource.packing import _seal_digest
        bad2 = json_deepcopy(seal)
        bad2["rows"][0]["body"] = "tampered"
        bad2["seal_digest"] = _seal_digest(bad2)
        report2 = verify_seal(m, bad2)
        self.assertFalse(report2["ok"])
        self.assertTrue(any("body_sha256" in e for e in report2["errors"]))

    def test_same_named_box_from_another_version_is_foreign(self):
        v1 = self._two_box_version(with_e=False)
        svc2, v2 = self._second_source(with_e=True)
        m1 = build_packing_manifest(self.svc, v1, ZERO_SEED)
        m2 = build_packing_manifest(svc2, v2, ZERO_SEED)
        # Both manifests contain box-yi; box-jia's best (B) and row count are
        # byte-identical across the two frozen versions, yet the seals carry
        # different version identities.
        seal_jia_v1 = self.seal(m1, "box-jia")
        seal_jia_v2 = make_seal(svc2, v2, "box-jia", m2)
        self.assertTrue(verify_seal(m1, seal_jia_v1)["ok"])
        self.assertTrue(verify_seal(m2, seal_jia_v2)["ok"])
        cross = verify_seal(m1, seal_jia_v2)
        self.assertFalse(cross["ok"])
        self.assertTrue(any("source_id" in e or "version" in e
                            for e in cross["errors"]))

    def test_explicit_box_map_must_cover_exactly_the_active_rows(self):
        v = self._two_box_version()
        meta = self.svc.all_rows_meta(v)
        good = {r["row_id"]: r["shard"] for r in meta}
        m = build_packing_manifest(self.svc, v, ZERO_SEED, box_map=good)
        self.assertEqual(len(m["boxes"]), 2)
        bad = dict(good)
        bad["GHOST"] = "box-jia"
        with self.assertRaises(Exception):
            build_packing_manifest(self.svc, v, ZERO_SEED, box_map=bad)

    def test_reboxing_is_physical_only_and_manifest_tracks_it(self):
        v = self._two_box_version()
        m_before = build_packing_manifest(self.svc, v, ZERO_SEED)
        # Seal box-jia and freeze both documents before the physical move.
        old_seal = make_seal(self.svc, v, "box-jia", m_before)
        frozen_old = json_deepcopy(m_before)
        # Move B into box-yi: no revision, no new source version.
        self.svc.apply_batch(
            [{"op": "set_shard", "row_id": "B", "shard": "box-yi"}])
        self.assertEqual(self.svc.latest_version(), v)
        m_after = build_packing_manifest(self.svc, v, ZERO_SEED)
        ids = {b["box_id"] for b in m_after["boxes"]}
        self.assertEqual(ids, {"box-yi"})
        yi = next(b for b in m_after["boxes"] if b["box_id"] == "box-yi")
        self.assertEqual(yi["row_count"], 2)
        # The frozen old manifest still verifies its frozen seal; the new
        # packing no longer attests a box-jia at all.
        self.assertTrue(verify_seal(frozen_old, old_seal)["ok"])
        self.assertIsNone(
            next((b for b in m_after["boxes"]
                  if b["box_id"] == "box-jia"), None))


def json_deepcopy(obj):
    import json
    return json.loads(json.dumps(obj, ensure_ascii=False))


class AcceptanceScenarioTest(PackingBase):
    def _version(self, with_e):
        shared = "same-payload"
        ops = [add("B", "G1", "box-jia", body=shared),
               add("A", "G1", "box-yi", body=shared)]
        if with_e:
            ops.append(add("E", "G1", "box-yi", body=shared))
        return self.svc.apply_batch(ops).version

    def _acceptor(self, v):
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        return m, Acceptor(m)

    def test_fast_ship_then_full_waits_scenario_v1(self):
        v = self._version(with_e=False)
        m, acc = self._acceptor(v)
        acc.ingest(self.seal(m, "box-jia"))      # B arrives first
        fast = acc.issue(FAST, "G1", 1)
        self.assertTrue(fast["issued"], fast["reason"])
        self.assertEqual([c.row_id for c in fast["selected"]], ["B"])
        cert = fast["certificate"]
        self.assertEqual(cert["selected"][0]["body"], "same-payload")
        # Boundary evidence names the missing box-yi and its exact best A.
        checks = cert["boundary"]["checks"]
        self.assertEqual([c["box_id"] for c in checks], ["box-yi"])
        self.assertEqual(checks[0]["missing_best"]["row_id"], "A")
        self.assertTrue(checks[0]["cannot_overtake"])
        # Full permit must wait.
        full = acc.issue(FULL, "G1", 1)
        self.assertFalse(full["issued"])
        self.assertIn("box-yi", full["reason"])
        # After the second box arrives, full permit ships the same B.
        acc.ingest(self.seal(m, "box-yi"))
        full2 = acc.issue(FULL, "G1", 1)
        self.assertTrue(full2["issued"])
        self.assertEqual(
            [r["row_id"] for r in full2["certificate"]["selected"]], ["B"])

    def test_v2_champion_in_missing_box_blocks_fast_then_ships_e(self):
        svc2, v = self._second_source(with_e=True)
        m = build_packing_manifest(svc2, v, ZERO_SEED)
        acc = Acceptor(m)
        acc.ingest(make_seal(svc2, v, "box-jia", m))   # only B arrived
        fast = acc.issue(FAST, "G1", 1)
        self.assertFalse(fast["issued"])
        self.assertIn("box-yi", fast["reason"])
        # The exact boundary check says box-yi's best E can overtake B.
        check = fast["boundary"]["checks"][0]
        self.assertEqual(check["missing_best"]["row_id"], "E")
        self.assertFalse(check["cannot_overtake"])
        # Full permit is blocked as well.
        self.assertFalse(acc.issue(FULL, "G1", 1)["issued"])
        acc.ingest(make_seal(svc2, v, "box-yi", m))
        shipped = acc.issue(FAST, "G1", 1)
        self.assertTrue(shipped["issued"])
        self.assertEqual([c.row_id for c in shipped["selected"]], ["E"])
        # B and E carry the same body text; the exact keys still distinguish.
        self.assertEqual(shipped["certificate"]["selected"][0]["body"],
                         "same-payload")

    def test_quota_zero_needs_no_bodies(self):
        v = self._version(with_e=False)
        m, acc = self._acceptor(v)
        res = acc.issue(FAST, "G1", 0)
        self.assertTrue(res["issued"])
        self.assertEqual(res["certificate"]["selected"], [])
        self.assertEqual(recheck_certificate(res["certificate"])["ok"], True)

    def test_under_quota_requires_proof_all_boxes_arrived(self):
        # Group G1 has one active row, quota 5: fast must not ship until every
        # non-empty box is verified, then the manifest count proves nothing is
        # missing.
        v = self.svc.apply_batch([add("B", "G1", "box-jia")]).version
        m, acc = self._acceptor(v)
        blocked = acc.issue(FAST, "G1", 5)
        self.assertFalse(blocked["issued"])
        self.assertIn("quota", blocked["reason"])
        acc.ingest(self.seal(m, "box-jia"))
        ok = acc.issue(FAST, "G1", 5)
        self.assertTrue(ok["issued"])
        self.assertEqual(
            [r["row_id"] for r in ok["certificate"]["selected"]], ["B"])
        self.assertEqual(
            ok["certificate"]["completeness"]["manifest_group_rows"], 1)

    def test_idempotent_intake_and_anomaly_retention(self):
        v = self._version(with_e=False)
        m, acc = self._acceptor(v)
        seal = self.seal(m, "box-jia")
        first = acc.ingest(seal)
        second = acc.ingest(seal)
        self.assertFalse(first["idempotent"])
        self.assertTrue(second["idempotent"])
        self.assertEqual(acc.verified_box_ids(), ["box-jia"])

        bad = json_deepcopy(seal)
        bad["rows"][0]["weight"] = 99
        from wrsource.packing import _seal_digest
        bad["seal_digest"] = _seal_digest(bad)
        report = acc.ingest(bad)
        self.assertFalse(report["stored"])
        self.assertEqual(len(acc.anomalies), 1)
        # The already verified box and any certificate survive the anomaly.
        cert = acc.issue(FAST, "G1", 1)
        self.assertTrue(cert["issued"])
        acc.ingest(bad)  # repeated anomaly is not double-counted
        self.assertEqual(len(acc.anomalies), 1)
        self.assertEqual(acc.verified_box_ids(), ["box-jia"])

    def test_certificate_frozen_and_offline_recheckable(self):
        v = self._version(with_e=False)
        m, acc = self._acceptor(v)
        acc.ingest(self.seal(m, "box-jia"))
        cert = acc.issue(FAST, "G1", 1)["certificate"]
        # A later bad arrival cannot invalidate the frozen certificate.
        bad = json_deepcopy(make_seal(self.svc, v, "box-yi", m))
        bad["rows"][0]["body"] = "evil"
        from wrsource.packing import _seal_digest
        bad["seal_digest"] = _seal_digest(bad)
        acc.ingest(bad)
        self.assertTrue(recheck_certificate(cert)["ok"])
        self.assertTrue(recheck_certificate_against_manifest(cert, m)["ok"])
        # Tamper with the certificate itself: the id catches it.
        tampered = json_deepcopy(cert)
        tampered["selected"][0]["body"] = "forged"
        self.assertFalse(recheck_certificate(tampered)["ok"])
        # A certificate built against another frozen version is rejected.
        svc2, v2 = self._second_source(with_e=True)
        m2 = build_packing_manifest(svc2, v2, ZERO_SEED)
        self.assertFalse(
            recheck_certificate_against_manifest(cert, m2)["ok"])

    def test_many_readers_varying_quotas_share_one_manifest(self):
        # Hundreds of boxes, one shared manifest, independent acceptor state.
        ops = []
        for i in range(300):
            ops.append(add(f"r{i:04d}", f"g{i % 6}", f"box{i:03d}",
                           body=f"payload-{i}"))
        v = self.svc.apply_batch(ops).version
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        self.assertEqual(len(m["boxes"]), 300)
        readers = [Acceptor(m), Acceptor(m)]
        # Each reader ingests the same seals in different orders.
        order = [b["box_id"] for b in m["boxes"]]
        for rid, acc in enumerate(readers):
            seq = order if rid == 0 else list(reversed(order))
            for bid in seq[:120]:
                r = acc.ingest(make_seal(self.svc, v, bid, m))
                self.assertTrue(r["ok"], r["errors"])
        q1, q2 = {"g0": 2}, {"g0": 50}
        r1 = readers[0].issue_many(FAST, q1)
        r2 = readers[1].issue_many(FAST, q2)
        # Both stay internally consistent; the quota-50 reader can generally
        # not prove completeness, the quota-2 reader may already ship.
        for res in (r1, r2):
            for g, out in res["results"].items():
                if out["issued"]:
                    self.assertTrue(
                        recheck_certificate(out["certificate"])["ok"])


class DispatchTest(PackingBase):
    def _three_group_source(self):
        # Two rows per group; whichever the exact zero-seed ordering ranks
        # first becomes the champion and is placed in X/Y/Z, the runner-up
        # goes to Q.  The test therefore follows the real lottery contract.
        from functools import cmp_to_key
        from wrsource.sampling import compare_rank, make_candidate
        rows = []
        self.champion_ids = {}
        for group, home in (("Gx", "X"), ("Gy", "Y"), ("Gz", "Z")):
            cands = [make_candidate(f"{group}-a", group, 1, "h", ZERO_SEED),
                     make_candidate(f"{group}-b", group, 1, "h", ZERO_SEED)]
            ranked = sorted(cands, key=cmp_to_key(compare_rank))
            champ, second = ranked[0].row_id, ranked[1].row_id
            self.champion_ids[group] = champ
            rows.append(add(champ, group, home, body=f"body-{group}-champ"))
            rows.append(add(second, group, "Q", body=f"body-{group}-second"))
        return self.svc.apply_batch(rows).version

    def test_textbook_xy_vs_xz_tradeoff(self):
        v = self._three_group_source()
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        champs = group_champions(m)
        self.assertEqual(champs["Gx"]["box_id"], "X")
        self.assertEqual(champs["Gy"]["box_id"], "Y")
        self.assertEqual(champs["Gz"]["box_id"], "Z")
        volumes = {"X": 1, "Y": 1, "Z": 1, "Q": 1}
        tickets = [
            Ticket.create("jia", ["Gx", "Gy"], 7),
            Ticket.create("yi", ["Gx"], 4),
            Ticket.create("bing", ["Gz"], 5),
        ]
        plan = plan_dispatch(m, tickets, volumes, capacity=2)
        self.assertEqual(plan["optimum"]["value"], 11)
        self.assertEqual(plan["optimum"]["ticket_ids"], ["jia", "yi"])
        self.assertEqual(plan["pickup_boxes"], ["X", "Y"])

        # Raise bing to 9: optimum switches to X+Z delivering yi+bing = 13.
        tickets2 = [
            Ticket.create("jia", ["Gx", "Gy"], 7),
            Ticket.create("yi", ["Gx"], 4),
            Ticket.create("bing", ["Gz"], 9),
        ]
        plan2 = plan_dispatch(m, tickets2, volumes, capacity=2)
        self.assertEqual(plan2["optimum"]["value"], 13)
        self.assertEqual(plan2["optimum"]["ticket_ids"], ["bing", "yi"])
        self.assertEqual(plan2["pickup_boxes"], ["X", "Z"])

    def test_shared_box_counts_capacity_once_and_ties_are_returned(self):
        v = self._three_group_source()
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        volumes = {"X": 1, "Y": 1, "Z": 1, "Q": 1}
        tickets = [
            Ticket.create("t1", ["Gx", "Gy"], 5),
            Ticket.create("t2", ["Gz"], 5),
        ]
        # X+Y and Z each fit in capacity 1 only for Z; X+Y needs 2.
        plan = plan_dispatch(m, tickets, volumes, capacity=1)
        self.assertEqual(plan["optimum"]["value"], 5)
        self.assertEqual(plan["optimum"]["ticket_ids"], ["t2"])
        # Equal-value alternatives with room for only one box: both single
        # tickets are strict optima and both must be reported.
        tickets_eq = [Ticket.create("a", ["Gx"], 5),
                      Ticket.create("b", ["Gz"], 5)]
        plan_eq = plan_dispatch(m, tickets_eq, volumes, capacity=1)
        self.assertTrue(plan_eq["optimum"]["tied"])
        alt_ids = sorted(tuple(a["ticket_ids"])
                         for a in plan_eq["optimum"]["alternatives"])
        self.assertEqual(alt_ids, [("a",), ("b",)])
        self.assertEqual(plan_eq["optimum"]["value"], 5)

    def test_arrived_boxes_replan_and_execution_certifies(self):
        v = self._three_group_source()
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        volumes = {"X": 1, "Y": 1, "Z": 1, "Q": 1}
        tickets = [
            Ticket.create("jia", ["Gx", "Gy"], 7),
            Ticket.create("yi", ["Gx"], 4),
            Ticket.create("bing", ["Gz"], 9),
        ]
        # With X already at the archive, capacity 2 can now reach Y and Z.
        plan = plan_dispatch(m, tickets, volumes, capacity=2,
                             arrived_boxes=["X"])
        load_states = {b["box_id"]: b["state"] for b in plan["load"]}
        self.assertEqual(load_states["X"], "arrived")
        self.assertEqual(plan["used_capacity"], 2)  # only Y and Z picked up
        self.assertEqual(plan["optimum"]["value"], 20)

        # Simulate the full trip: seals for every touched box, then fast
        # permits per ticket group (champion box verified, Q left behind but
        # excluded by the manifest boundary evidence).
        seals = {b["box_id"]: make_seal(self.svc, v, b["box_id"], m)
                 for b in plan["load"]}
        exec_result = execute_plan(plan, seals, m,
                                   {"Gx": 1, "Gy": 1, "Gz": 1})
        self.assertEqual(exec_result["earned_value"], 20)
        self.assertEqual(
            sorted(e["ticket_id"] for e in exec_result["earned_tickets"]),
            ["bing", "jia", "yi"])
        for group, cert in exec_result["certificates"].items():
            self.assertTrue(recheck_certificate(cert)["ok"])

        # Dropping one champion seal makes exactly the dependent tickets fail.
        partial = dict(seals)
        del partial["Z"]
        failed_run = execute_plan(plan, partial, m,
                                  {"Gx": 1, "Gy": 1, "Gz": 1})
        self.assertEqual(failed_run["earned_value"], 11)
        failed_groups = {f["group"] for f in failed_run["failures"]
                         if "group" in f}
        self.assertIn("Gz", failed_groups)

    def test_reboxing_champion_box_changes_plan_without_new_version(self):
        v = self._three_group_source()
        m1 = build_packing_manifest(self.svc, v, ZERO_SEED)
        volumes = {"X": 1, "Y": 1, "Z": 1, "Q": 1}
        tickets = [Ticket.create("jia", ["Gx", "Gy"], 7),
                   Ticket.create("bing", ["Gz"], 9)]
        plan1 = plan_dispatch(m1, tickets, volumes, capacity=2)
        # Move Gx's champion physically from X into Z (rebox: no new version).
        champ_x = group_champions(m1)["Gx"]["key"]["row_id"]
        self.svc.apply_batch(
            [{"op": "set_shard", "row_id": champ_x, "shard": "Z"}])
        self.assertEqual(self.svc.latest_version(), v)
        m2 = build_packing_manifest(self.svc, v, ZERO_SEED)
        self.assertEqual(group_champions(m2)["Gx"]["box_id"], "Z")
        plan2 = plan_dispatch(m2, tickets, volumes, capacity=2)
        # Gx and Gz champions now share Z; both tickets fit via Y+Z = 16.
        self.assertEqual(plan2["optimum"]["value"], 16)
        self.assertEqual(sorted(plan2["pickup_boxes"]), ["Y", "Z"])
        self.assertNotEqual(plan1["pickup_boxes"], plan2["pickup_boxes"])

    def test_execution_with_missing_seal_fails_ticket_not_silent(self):
        v = self._three_group_source()
        m = build_packing_manifest(self.svc, v, ZERO_SEED)
        volumes = {"X": 1, "Y": 1, "Z": 1, "Q": 1}
        tickets = [Ticket.create("only", ["Gx"], 4)]
        plan = plan_dispatch(m, tickets, volumes, capacity=2)
        seals = {}  # nothing actually loaded
        result = execute_plan(plan, seals, m, {"Gx": 1})
        self.assertEqual(result["earned_value"], 0)
        self.assertTrue(result["failures"])


if __name__ == "__main__":
    unittest.main()
