"""Die App-Optionen gegen Schema und run.sh pruefen.

Zwei Fehlerklassen, die beide schon aufgetreten sind und die kein Python-Test bisher
gefangen haette:

* Issue #24 — eine Option stand unter ``options``, fehlte aber im ``schema``. Der
  Supervisor warnte bei jedem Start, und der Wert war wirkungslos.
* Issue #31 — der Ordner-Eingang existierte im Dienst, aber die App bot ihn nicht an:
  keine Optionen, kein ``map``, und ``run.sh`` schrieb keinen ``folder``-Block.
"""
import re
import unittest
from pathlib import Path

import yaml

ADDON = Path(__file__).resolve().parent.parent / "faceid-addon"
CONFIG = yaml.safe_load((ADDON / "config.yaml").read_text(encoding="utf-8"))
RUN_SH = (ADDON / "run.sh").read_text(encoding="utf-8")
# Nur die erzeugte Vorlage, nicht das ganze Skript: davor duerfen Werte roh gelesen
# werden (Vorab-Pruefungen, und die MQTT-Daten vor der Broker-Erkennung). Entscheidend
# ist, was in der Datei landet.
TEMPLATE = RUN_SH.split("cat > /opt/faceid/config.yaml << EOF\n", 1)[1].split("\nEOF", 1)[0]


class SchemaCoverageTests(unittest.TestCase):
    def test_every_option_has_a_schema_entry(self):
        missing = sorted(set(CONFIG["options"]) - set(CONFIG["schema"]))
        self.assertEqual(missing, [], f"ohne Schema, Supervisor warnt beim Start: {missing}")

    def test_every_schema_entry_has_a_default(self):
        missing = sorted(set(CONFIG["schema"]) - set(CONFIG["options"]))
        self.assertEqual(missing, [], f"im Schema, aber ohne Vorgabe: {missing}")

    def test_every_option_reaches_the_generated_config(self):
        # Sonst ist die Option sichtbar, einstellbar und wirkungslos — genau der
        # Zustand, den Issue #24 beschrieb.
        ignored = {
            "mqtt_host", "mqtt_port", "mqtt_user", "mqtt_password",  # via jstr nach der Broker-Erkennung
        }
        # Zwei Schreibweisen: Skalare ueber cfg/yml ('.x'), Listen ueber jq -c ('.x //').
        missing = [k for k in CONFIG["options"]
                   if k not in ignored
                   and f".{k}'" not in RUN_SH and f".{k} //" not in RUN_SH]
        self.assertEqual(missing, [], f"nicht in run.sh verwendet: {missing}")


class FolderModeTests(unittest.TestCase):
    def test_media_and_share_are_mapped_read_only(self):
        # Ohne die Einhaengung sieht die App den Ordner nicht, egal wie er konfiguriert
        # ist. Lesend genuegt: FaceID veraendert die Aufnahmen nie.
        self.assertEqual(sorted(CONFIG.get("map") or []), ["media:ro", "share:ro"])

    def test_the_generated_config_has_a_folder_block(self):
        self.assertRegex(TEMPLATE, r"(?m)^folder:$")

    def test_frigate_url_is_optional(self):
        # Pflichtfeld hiesse: ohne Frigate laesst sich die App gar nicht speichern.
        self.assertEqual(CONFIG["schema"]["frigate_url"], "url?")

    def test_folder_options_are_complete(self):
        expected = {
            "folder_enabled", "folder_path", "folder_camera", "folder_recursive",
            "folder_process_existing", "folder_extensions", "folder_poll_interval",
            "folder_settle_seconds", "folder_max_frames", "folder_max_people_per_file",
            "folder_max_indexed_files",
        }
        self.assertEqual(expected - set(CONFIG["options"]), set())

    def test_the_folder_block_uses_the_unprefixed_keys(self):
        # run.sh uebersetzt folder_path -> folder.path. Schreibt es versehentlich den
        # App-Namen in die Dienst-Konfiguration, liest FolderIngest die Vorgabe.
        block = TEMPLATE.split("\nfolder:\n", 1)[1]
        keys = re.findall(r"(?m)^\s+([a-z_]+):", block)
        expected = ["enabled", "path", "camera", "recursive", "process_existing",
                    "extensions", "poll_interval", "settle_seconds", "max_frames",
                    "max_people_per_file", "max_indexed_files"]
        self.assertEqual(sorted(keys), sorted(expected))
        # Die jq-Ausdruecke rechts duerfen das App-Praefix tragen (.folder_path), die
        # YAML-Schluessel links nicht — sonst liest FolderIngest seine Vorgaben.
        self.assertEqual([k for k in keys if k.startswith("folder_")], [])


class FreeTextEscapingTests(unittest.TestCase):
    """Freitext muss durch jq, sonst zerlegt ein Anfuehrungszeichen die Datei."""

    def test_string_options_go_through_the_encoder(self):
        raw = {k for k in CONFIG["options"]
               if isinstance(CONFIG["schema"].get(k), str)
               and CONFIG["schema"][k].rstrip("?") in {"str", "url", "password"}
               and f"cfg '.{k}'" in TEMPLATE}
        self.assertEqual(raw, set(), f"roh statt ueber yml(): {sorted(raw)}")
