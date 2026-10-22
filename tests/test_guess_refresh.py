"""Assign must not block on a full-queue rematch; unchanged guesses stay on disk."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.gallery import Gallery
from app.webui import AssignBody, build_app


def _emb():
    return np.ones(512, dtype=np.float32) / np.sqrt(512)


def _gallery(tmp: Path) -> Gallery:
    (tmp / "persons").mkdir(parents=True, exist_ok=True)
    (tmp / "ignored").mkdir(exist_ok=True)
    return Gallery(tmp, top_k=3, max_per_person=40)


def _app(tmp: Path, frigate=None):
    data = tmp / "data"
    gallery = _gallery(data)
    cfg = {"faceid": {"match_threshold": 0.5, "unknown_threshold": 0.35,
                      "suggest_threshold": 0.4, "cluster_eps": 0.55,
                      "ignore_threshold": 0.5, "dedupe_threshold": 0.65},
           "folder": {"enabled": False}, "frigate": {"url": ""}, "mqtt": {}}
    proc = SimpleNamespace(_announced=set(), match_thr=0.5, unknown_thr=0.35,
                           ignore_thr=0.5, history=None, logbuffer=None,
                           frigate=frigate or SimpleNamespace(enabled=False),
                           folder_ingest=SimpleNamespace(status=lambda: {},
                                                         max_indexed_files=5000,
                                                         enabled=False),
                           queue=SimpleNamespace(qsize=lambda: 0),
                           events={})
    app = build_app(cfg, None, gallery, proc, data, Path("static"))
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


if __name__ == "__main__":
    unittest.main()
