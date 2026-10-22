"""Assign must not block on a full-queue rematch; unchanged guesses stay on disk."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fastapi import HTTPException

from app.gallery import Gallery
from app.webui import AssignBody, build_app


def _emb():
    return np.ones(512, dtype=np.float32) / np.sqrt(512)


def _gallery(tmp: Path) -> Gallery:
    (tmp / "persons").mkdir(parents=True, exist_ok=True)
    (tmp / "ignored").mkdir(exist_ok=True)
    return Gallery(tmp, top_k=3, max_per_person=40)


def _app(tmp: Path, frigate=..., static_dir=None):
    data = tmp / "data"
    gallery = _gallery(data)
    cfg = {"faceid": {"match_threshold": 0.5, "unknown_threshold": 0.35,
                      "suggest_threshold": 0.4, "cluster_eps": 0.55,
                      "ignore_threshold": 0.5, "dedupe_threshold": 0.65},
           "folder": {"enabled": False}, "frigate": {"url": ""}, "mqtt": {}}
    if frigate is ...:
        frigate = SimpleNamespace(enabled=False)
    proc = SimpleNamespace(_announced=set(), match_thr=0.5, unknown_thr=0.35,
                           ignore_thr=0.5, history=None, logbuffer=None,
                           frigate=frigate,
                           folder_ingest=SimpleNamespace(status=lambda: {},
                                                         max_indexed_files=5000,
                                                         enabled=False),
                           queue=SimpleNamespace(qsize=lambda: 0),
                           events={})
    app = build_app(cfg, None, gallery, proc, data, Path(static_dir or "static"))
    return app, gallery, proc


def _route(app, path: str):
    for r in app.routes:
        if getattr(r, "path", None) == path:
            return r.endpoint
    raise AssertionError(f"route {path} not found")


class SkipUnchangedRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gal = _gallery(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _unknown(self, uid: str):
        # Write files directly: save_unknown() would collapse identical embeddings.
        vec = [round(float(v), 6) for v in _emb()]
        (self.gal.unknown_dir / f"{uid}.json").write_text(
            json.dumps({"ts": 1, "embedding": vec, "camera": "entrance"}))
        (self.gal.unknown_dir / f"{uid}.jpg").write_bytes(b"jpg")
        return uid

    def test_a_second_pass_does_not_rewrite_unchanged_guesses(self):
        for i in range(4):
            self._unknown(f"u{i}")
        self.gal.refresh_guesses()
        before = {}
        for jf in self.gal.unknown_dir.glob("*.json"):
            st = jf.stat()
            before[jf.name] = (st.st_mtime_ns, st.st_size, jf.read_text())
        self.gal.refresh_guesses()
        for jf in self.gal.unknown_dir.glob("*.json"):
            st = jf.stat()
            mtime, size, raw = before[jf.name]
            self.assertEqual(st.st_mtime_ns, mtime, f"{jf.name} was rewritten")
            self.assertEqual(st.st_size, size)
            self.assertEqual(jf.read_text(), raw)

    def test_skip_unchanged_still_does_not_resurrect_a_deleted_unknown(self):
        self._unknown("u1")
        self.gal.refresh_guesses()
        jf = self.gal.unknown_dir / "u1.json"
        jpg = self.gal.unknown_dir / "u1.jpg"
        real_match = self.gal.match

        def match_and_delete(emb):
            jf.unlink(missing_ok=True)
            jpg.unlink(missing_ok=True)
            return real_match(emb)

        self.gal.match = match_and_delete
        try:
            self.gal.refresh_guesses()
        finally:
            self.gal.match = real_match
        self.assertFalse(jf.exists())
        leftovers = [p.name for p in self.gal.unknown_dir.iterdir()
                     if p.suffix == ".tmp" or p.name.startswith(".")]
        self.assertEqual(leftovers, [])


class AssignDoesNotBlockRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app, self.gallery, self.proc = _app(Path(self.tmp.name))
        self.assign = _route(self.app, "/api/unknowns/assign")
        self.auto = _route(self.app, "/api/unknowns/auto_assign")
        self.health = _route(self.app, "/api/health")
        self.ignore = _route(self.app, "/api/unknowns/ignore")
        self.discard = _route(self.app, "/api/unknowns/discard")

    def tearDown(self):
        self.tmp.cleanup()

    def _save_unknown(self):
        crop = np.full((80, 80, 3), 127, dtype=np.uint8)
        return self.gallery.save_unknown(crop, _emb(), {"camera": "entrance"})

    def test_assign_schedules_refresh_instead_of_running_it(self):
        slug = self.gallery.create_person("Juli")
        uid = self._save_unknown()
        called = []
        orig = self.gallery.refresh_guesses
        def wrapped():
            called.append("sync")
            return orig()
        self.gallery.refresh_guesses = wrapped
        r = self.assign(AssignBody(ids=[uid], person=slug))
        self.assertEqual(r["assigned"], 1)
        self.assertEqual(r["ids"], [uid])
        self.assertNotIn("sync", called,
                         "assign must not run refresh_guesses on the handler thread")
        self.assertTrue(r.get("guesses_refreshing") or self.gallery.guesses_refreshing
                        or self.gallery._refresh_deadline is not None
                        or (self.gallery._refresh_thread and self.gallery._refresh_thread.is_alive()))

    def test_auto_assign_does_not_rematch_twice_on_the_handler(self):
        slug = self.gallery.create_person("Juli")
        # A face that will match Juli (same embedding as an enrolled photo).
        crop = np.full((80, 80, 3), 127, dtype=np.uint8)
        self.gallery.add_face(slug, crop, _emb(), source={"camera": "entrance"})
        uid = self._save_unknown()
        called = []
        orig = self.gallery.refresh_guesses
        def wrapped():
            called.append("sync")
            return orig()
        self.gallery.refresh_guesses = wrapped
        r = self.auto()
        self.assertGreaterEqual(r["total"], 1)
        self.assertNotIn("sync", called)
        self.assertIn(uid, r.get("ids") or [])

    def test_ignore_and_discard_do_not_schedule_refresh(self):
        uid = self._save_unknown()
        with patch.object(self.gallery, "request_refresh_guesses") as req:
            self.ignore(AssignBody(ids=[uid], person="-"))
            req.assert_not_called()
        uid2 = self._save_unknown()
        with patch.object(self.gallery, "request_refresh_guesses") as req:
            self.discard(AssignBody(ids=[uid2], person="-"))
            req.assert_not_called()

    def test_health_exposes_guess_refresh_flags(self):
        h = self.health()
        self.assertIn("guesses_epoch", h)
        self.assertIn("guesses_refreshing", h)
        self.assertIsInstance(h["guesses_epoch"], int)
        self.assertIsInstance(h["guesses_refreshing"], bool)


class SubLabelOffHandlerClockTests(unittest.TestCase):
    def test_assign_returns_before_a_slow_frigate_post(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            slow = []
            def set_sub_label(event_id, label, score):
                time.sleep(0.4)
                slow.append((event_id, label, score))
            frigate = SimpleNamespace(enabled=True, set_sub_label=set_sub_label)
            app, gallery, _proc = _app(Path(tmp.name), frigate=frigate)
            assign = _route(app, "/api/unknowns/assign")
            slug = gallery.create_person("Juli")
            crop = np.full((80, 80, 3), 127, dtype=np.uint8)
            uid = gallery.save_unknown(
                crop, _emb(), {"camera": "entrance", "event_id": "evt-1"})
            t0 = time.perf_counter()
            r = assign(AssignBody(ids=[uid], person=slug))
            dt = time.perf_counter() - t0
            self.assertEqual(r["assigned"], 1)
            self.assertLess(dt, 0.25, f"assign waited on Frigate ({dt:.3f}s)")
            deadline = time.time() + 2.0
            while not slow and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(slow[0][0], "evt-1")
        finally:
            tmp.cleanup()


class ReviewFollowUpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app, self.gallery, self.proc = _app(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _wait_idle(self, timeout=2.0):
        deadline = time.time() + timeout
        while self.gallery.guesses_refreshing and time.time() < deadline:
            time.sleep(0.03)
        self.assertFalse(self.gallery.guesses_refreshing)

    def test_failed_background_pass_clears_refreshing_without_stamping_success(self):
        self.gallery._guesses_updated_ts = 0.0
        def boom():
            raise RuntimeError("disk full")
        self.gallery.refresh_guesses = boom
        self.gallery.request_refresh_guesses(delay=0.01, max_delay=0.01)
        self._wait_idle()
        self.assertEqual(self.gallery.guesses_updated_ts, 0.0)

    def test_successful_background_pass_stamps_updated_ts(self):
        self.gallery._guesses_updated_ts = 0.0
        self.gallery.request_refresh_guesses(delay=0.01, max_delay=0.01)
        self._wait_idle()
        self.assertGreater(self.gallery.guesses_updated_ts, 0.0)

    def test_unassign_and_ignore_person_return_guess_flags(self):
        slug = self.gallery.create_person("Juli")
        crop = np.full((80, 80, 3), 127, dtype=np.uint8)
        fname = self.gallery.add_face(slug, crop, _emb(), source={"camera": "entrance"})
        unassign = _route(self.app, "/api/persons/{slug}/faces/{fname}/unassign")
        r = unassign(slug, fname)
        self.assertTrue(r["ok"])
        self.assertIn("guesses_epoch", r)
        self.assertIn("guesses_refreshing", r)
        slug2 = self.gallery.create_person("Quinton")
        self.gallery.add_face(slug2, crop, _emb(), source={"camera": "entrance"})
        ignore = _route(self.app, "/api/persons/{slug}/ignore")
        r2 = ignore(slug2)
        self.assertGreaterEqual(r2["ignored_faces"], 1)
        self.assertIn("guesses_epoch", r2)
        self.assertIn("guesses_refreshing", r2)

    def test_favicon_missing_is_404(self):
        empty = Path(self.tmp.name) / "nostatic"
        empty.mkdir()
        app, _, _ = _app(Path(self.tmp.name) / "other", static_dir=empty)
        for path in ("/favicon.svg", "/favicon.ico"):
            with self.assertRaises(HTTPException) as cm:
                _route(app, path)()
            self.assertEqual(cm.exception.status_code, 404)

    def test_assign_does_not_enqueue_when_frigate_is_none(self):
        app, gallery, _proc = _app(Path(self.tmp.name) / "nonefrigate", frigate=None)
        assign = _route(app, "/api/unknowns/assign")
        slug = gallery.create_person("Juli")
        crop = np.full((80, 80, 3), 127, dtype=np.uint8)
        uid = gallery.save_unknown(
            crop, _emb(), {"camera": "entrance", "event_id": "evt-none"})
        r = assign(AssignBody(ids=[uid], person=slug))
        self.assertEqual(r["assigned"], 1)


if __name__ == "__main__":
    unittest.main()
