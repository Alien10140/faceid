import unittest
from types import SimpleNamespace

import numpy as np

from app.mqtt_listener import EventProcessor


class FakeFrigate:
    enabled = False

    def __init__(self):
        self.labels = []

    def set_sub_label(self, *args):
        self.labels.append(args)

    def config(self):
        return {"cameras": {}}


class FakeGallery:
    def __init__(self, match=(None, None, 0.0), ignored=0.0):
        self.match_result = match
        self.ignored = ignored
        self.saved = []

    def match(self, embedding):
        return self.match_result

    def match_ignored(self, embedding):
        return self.ignored

    def save_unknown(self, crop, embedding, meta, full_bgr=None):
        self.saved.append((crop, embedding, meta, full_bgr))
        return "u1"

    def add_ignore_anchor(self, crop, embedding):
        return None


def config():
    return {
        "mqtt": {"enabled": False},
        "faceid": {
            "match_threshold": 0.5, "unknown_threshold": 0.35,
            "ignore_threshold": 0.5, "ignore_learning": False,
            "set_sub_label": True, "discovery_cameras": ["front_door"],
        },
    }


def face():
    return SimpleNamespace(
        normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        bbox=np.array([10, 10, 90, 90], dtype=np.float32),
        det_score=0.9,
    )


class LocalEventProcessorTests(unittest.TestCase):
    def test_unknown_is_saved_with_source_path_and_published_to_recent(self):
        gallery = FakeGallery()
        processor = EventProcessor(config(), object(), gallery, FakeFrigate())
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = processor.process_local_face(
            "folder-abc-1", "front_door", 1234.0, image, face(), "/watch/visit.mp4")

        self.assertEqual(result["person"], "unknown")
        self.assertEqual(result["unknown_id"], "u1")
        self.assertEqual(gallery.saved[0][2]["media_path"], "/watch/visit.mp4")
        self.assertIs(gallery.saved[0][3], image)
        self.assertEqual(processor.recent[0]["media_file"], "visit.mp4")

    def test_known_person_does_not_try_to_write_a_frigate_label(self):
        frigate = FakeFrigate()
        gallery = FakeGallery(match=("alice", "Alice", 0.8))
        processor = EventProcessor(config(), object(), gallery, frigate)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = processor.process_local_face(
            "folder-known-1", "front_door", 1234.0, image, face(), "/watch/visit.mp4")

        self.assertEqual(result["person"], "Alice")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(frigate.labels, [])
        self.assertEqual(gallery.saved, [])

    def test_ignored_face_is_neither_saved_nor_reported(self):
        gallery = FakeGallery(ignored=0.8)
        processor = EventProcessor(config(), object(), gallery, FakeFrigate())
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = processor.process_local_face(
            "folder-ignore-1", "front_door", 1234.0, image, face(), "/watch/visit.mp4")

        self.assertEqual(result["person"], "ignored")
        self.assertEqual(gallery.saved, [])
        self.assertEqual(list(processor.recent), [])


if __name__ == "__main__":
    unittest.main()


class ProcessorDefaultsTests(unittest.TestCase):
    def test_folder_ingest_has_a_class_level_default(self):
        # Damit kein Leser den Zeitraum zwischen start() und der Zuweisung abfangen muss.
        from app.mqtt_listener import EventProcessor
        self.assertIsNone(EventProcessor.folder_ingest)


class FolderDiscoveryTests(unittest.TestCase):
    """Im reinen Ordnerbetrieb gibt es keine Kameraliste zum Abfragen.

    Gemeldet als Issue #31: ohne Frigate hat FaceID keine Quelle fuer Kameranamen, und
    der Sensor entstuende erst beim ersten erkannten Gesicht. Bis dahin sieht die
    Installation aus, als tue sie nichts.
    """

    def _processor(self, cfg):
        return EventProcessor(cfg, engine=None, gallery=FakeGallery(), frigate=FakeFrigate())

    def _announced(self, cfg):
        proc = self._processor(cfg)
        seen = []
        proc.client = SimpleNamespace(publish=lambda topic, payload=None, **kw:
                                      seen.append(topic))
        proc._publish_discovery()
        return proc._announced

    def test_the_folder_camera_is_announced_without_frigate(self):
        cfg = config()
        cfg["faceid"]["discovery_cameras"] = []
        cfg["folder"] = {"enabled": True, "camera": "driveway"}
        self.assertIn("driveway", self._announced(cfg))

    def test_the_folder_camera_joins_the_configured_ones(self):
        cfg = config()
        cfg["folder"] = {"enabled": True, "camera": "driveway"}
        announced = self._announced(cfg)
        self.assertIn("driveway", announced)
        self.assertIn("front_door", announced)

    def test_a_disabled_folder_contributes_no_camera(self):
        cfg = config()
        cfg["folder"] = {"enabled": False, "camera": "driveway"}
        self.assertNotIn("driveway", self._announced(cfg))

    def test_an_empty_folder_camera_is_not_announced(self):
        cfg = config()
        cfg["faceid"]["discovery_cameras"] = []
        cfg["folder"] = {"enabled": True, "camera": "  "}
        self.assertEqual(self._announced(cfg), set())

    def test_frigate_is_not_queried_when_the_folder_name_suffices(self):
        # Der Aufruf geht ueber das Netz. Im Ordnerbetrieb steht der Name schon fest.
        cfg = config()
        cfg["faceid"]["discovery_cameras"] = []
        cfg["folder"] = {"enabled": True, "camera": "driveway"}
        proc = self._processor(cfg)
        proc.client = SimpleNamespace(publish=lambda *a, **k: None)
        called = []
        proc.frigate_enabled = False
        proc._frigate_cameras = lambda: called.append(1) or set()
        proc._publish_discovery()
        self.assertEqual(called, [], "ohne Frigate darf nicht gefragt werden")

    def test_frigate_is_still_queried_when_nothing_else_is_known(self):
        cfg = config()
        cfg["faceid"]["discovery_cameras"] = []
        proc = self._processor(cfg)
        proc.frigate_enabled = True
        proc.client = SimpleNamespace(publish=lambda *a, **k: None)
        proc._frigate_cameras = lambda: {"from_frigate"}
        proc._publish_discovery()
        self.assertIn("from_frigate", proc._announced)

    def test_frigate_cameras_still_appear_alongside_a_folder(self):
        # Gemischter Aufbau: beide Quellen muessen beim Start einen Sensor bekommen.
        cfg = config()
        cfg["faceid"]["discovery_cameras"] = []
        cfg["folder"] = {"enabled": True, "camera": "driveway"}
        proc = self._processor(cfg)
        proc.frigate_enabled = True
        proc.client = SimpleNamespace(publish=lambda *a, **k: None)
        proc._frigate_cameras = lambda: {"hof", "tor"}
        proc._publish_discovery()
        self.assertEqual(proc._announced, {"driveway", "hof", "tor"})
