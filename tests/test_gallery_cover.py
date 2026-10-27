import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.gallery import Gallery
from tests.test_assign_guards import _app, _route


def _face(gallery, slug, seed):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    img = (rng.random((112, 112, 3)) * 255).astype(np.uint8)
    gallery.add_face(slug, img, emb)
    return gallery.persons()[slug]["files"][-1]


class CoverTests(unittest.TestCase):
    def test_cover_is_set_kept_across_reload_and_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery = Gallery(Path(tmp))
            slug = gallery.create_person("Mara")
            first = _face(gallery, slug, 1)
            second = _face(gallery, slug, 2)

            self.assertIsNone(gallery.persons()[slug]["cover"])
            self.assertTrue(gallery.set_cover(slug, second))
            self.assertEqual(gallery.persons()[slug]["cover"], second)

            reloaded = Gallery(Path(tmp))
            self.assertEqual(reloaded.persons()[slug]["cover"], second)

            self.assertTrue(reloaded.set_cover(slug, None))
            self.assertIsNone(reloaded.persons()[slug]["cover"])
            self.assertIn(first, reloaded.persons()[slug]["files"])

    def test_unknown_person_or_photo_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery = Gallery(Path(tmp))
            slug = gallery.create_person("Mara")
            _face(gallery, slug, 1)

            self.assertFalse(gallery.set_cover("nobody", None))
            self.assertFalse(gallery.set_cover(slug, "not-a-photo.jpg"))
            self.assertIsNone(gallery.persons()[slug]["cover"])

    def test_cover_falls_away_when_its_photo_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery = Gallery(Path(tmp))
            slug = gallery.create_person("Mara")
            _face(gallery, slug, 1)
            cover = _face(gallery, slug, 2)
            gallery.set_cover(slug, cover)

            gallery.delete_face(slug, cover)

            self.assertIsNone(gallery.persons()[slug]["cover"])


class FullPersonUndoTests(unittest.TestCase):
    """Undo of an assignment to a person at the photo limit."""

    def _full_gallery(self, tmp):
        gallery = Gallery(Path(tmp), max_per_person=2)
        slug = gallery.create_person("Mara")
        rng = np.random.default_rng(7)
        for seed in (1, 2, 3):
            emb = rng.normal(size=512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            img = (rng.random((112, 112, 3)) * 255).astype(np.uint8)
            gallery.add_face(slug, img, emb, source={"camera": "entrance", "event_ts": 1792000000 + seed})
        return gallery, slug

    def test_a_set_aside_photo_goes_back_to_sorting_with_camera_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery, slug = self._full_gallery(tmp)
            trimmed = gallery.persons()[slug]["trimmed"]
            self.assertEqual(len(trimmed), 1)
            fname = trimmed[0]["file"]

            self.assertTrue(gallery.unassign_trimmed(slug, fname))

            self.assertEqual(gallery.persons()[slug]["trimmed"], [])
            self.assertEqual(gallery.persons()[slug]["count"], 2)
            metas = [json.loads(f.read_text()) for f in gallery.unknown_dir.glob("u*.json")]
            self.assertEqual(len(metas), 1)
            self.assertEqual(metas[0]["camera"], "entrance")
            self.assertIn("event_ts", metas[0])
            self.assertEqual(len(metas[0]["embedding"]), 512)
            self.assertFalse(gallery.unassign_trimmed(slug, fname), "a second call finds nothing")

    def test_restoring_keeps_the_source_and_unassign_carries_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery, slug = self._full_gallery(tmp)
            fname = gallery.persons()[slug]["trimmed"][0]["file"]
            self.assertTrue(gallery.restore_trimmed(slug, fname))
            self.assertEqual(gallery.persons()[slug]["count"], 3, "restore does not enforce the limit")

            self.assertTrue(gallery.unassign_face(slug, fname))
            meta = json.loads(next(gallery.unknown_dir.glob("u*.json")).read_text())
            self.assertEqual(meta["camera"], "entrance")
            self.assertIn("event_ts", meta)


class CoverEndpointTests(unittest.TestCase):
    def test_endpoint_sets_and_refuses(self):
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as tmp:
            app, gallery = _app(Path(tmp))
            slug = gallery.create_person("Mara")
            photo = _face(gallery, slug, 1)
            cover = _route(app, "/api/persons/{slug}/cover")
            body = [r for r in app.routes if getattr(r, "path", "") == "/api/persons/{slug}/cover"][0]
            Body = body.endpoint.__annotations__["body"]

            self.assertEqual(cover(slug, Body(file=photo)), {"ok": True, "cover": photo})
            self.assertEqual(gallery.persons()[slug]["cover"], photo)
            with self.assertRaises(HTTPException) as cm:
                cover(slug, Body(file="missing.jpg"))
            self.assertEqual(cm.exception.status_code, 404)

    def test_classic_route_serves_the_previous_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = _app(Path(tmp))
            resp = _route(app, "/classic")()
            self.assertTrue(str(resp.path).endswith("classic.html"))


if __name__ == "__main__":
    unittest.main()
