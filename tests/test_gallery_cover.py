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
