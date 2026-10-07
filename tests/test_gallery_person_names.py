import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from app.gallery import Gallery, slugify


class PersonNameTests(unittest.TestCase):
    def test_non_latin_names_get_distinct_stable_slugs(self):
        self.assertEqual(slugify("Σοφία"), slugify("σοφία"))
        self.assertNotEqual(slugify("Σοφία"), slugify("さくら"))
        self.assertRegex(slugify("Σοφία"), r"^person-[0-9a-f]{10}$")

    def test_people_with_names_from_different_scripts_are_both_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery = Gallery(Path(tmp))
            first = gallery.create_person("Σοφία")
            second = gallery.create_person("さくら")

            self.assertNotEqual(first, second)
            self.assertEqual(
                {person["name"] for person in gallery.persons().values()},
                {"Σοφία", "さくら"},
            )

    def test_ascii_slug_collisions_do_not_reuse_another_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery = Gallery(Path(tmp))
            first = gallery.create_person("Alex")
            second = gallery.create_person("Alex!")

            self.assertEqual(first, "alex")
            self.assertNotEqual(first, second)
            self.assertEqual(len(gallery.persons()), 2)

    def test_same_unicode_name_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            gallery = Gallery(Path(tmp))
            first = gallery.create_person("Σοφία")
            second = gallery.create_person("σοφία")

            self.assertEqual(first, second)
            self.assertEqual(len(gallery.persons()), 1)


if __name__ == "__main__":
    unittest.main()


class RefreshGuessesConcurrencyTests(unittest.TestCase):
    """Zwei gleichzeitige Durchlaeufe duerfen keine halbe Datei veroeffentlichen."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "persons").mkdir()
        (d / "ignored").mkdir()
        self.gal = Gallery(d, top_k=3, max_per_person=40)

    def tearDown(self):
        self.tmp.cleanup()

    def _unknown(self, uid: str):
        emb = [round(float(v), 6) for v in (np.ones(512, dtype=np.float32) / np.sqrt(512))]
        (self.gal.unknown_dir / f"{uid}.json").write_text(
            json.dumps({"ts": 1, "embedding": emb, "camera": "entrance"}))

    def test_parallel_refreshes_never_publish_a_partial_file(self):
        for i in range(12):
            self._unknown(f"u{i}")
        errors = []

        def run():
            try:
                for _ in range(6):
                    self.gal.refresh_guesses()
            except Exception as exc:  # pragma: no cover - nur im Fehlerfall
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        for jf in self.gal.unknown_dir.glob("*.json"):
            m = json.loads(jf.read_text())  # wirft, wenn abgeschnitten
            self.assertEqual(len(m["embedding"]), 512,
                             f"{jf.name} hat sein Embedding verloren")

    def test_no_temp_files_are_left_behind(self):
        self._unknown("u1")
        self.gal.refresh_guesses()
        leftovers = [p.name for p in self.gal.unknown_dir.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])

    def test_a_burst_cannot_defer_the_refresh_forever(self):
        # Rein nachlaufend wuerde jeder Aufruf die Frist neu setzen; die Obergrenze
        # muss die Frist festnageln, sobald max_delay erreicht ist.
        self._unknown("u1")
        start = time.time()
        for _ in range(50):
            self.gal.request_refresh_guesses(delay=0.4, max_delay=0.3)
        with self.gal._refresh_sched_lock:
            deadline = self.gal._refresh_deadline
        self.assertIsNotNone(deadline)
        self.assertLessEqual(deadline - start, 0.35,
                             "die Obergrenze muss den Dauerstrom abschneiden")
