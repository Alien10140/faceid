import os
import tempfile
import unittest
from pathlib import Path

from app.backup_util import check_backup_dir, prune_backups, write_backup_file


class CheckBackupDirTests(unittest.TestCase):
    def test_a_writable_directory_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(check_backup_dir(tmp))

    def test_a_missing_directory_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "deep" / "faceid"
            self.assertIsNone(check_backup_dir(target))
            self.assertTrue(target.is_dir())

    def test_a_relative_path_is_rejected(self):
        problem = check_backup_dir("backups/faceid")
        self.assertIsNotNone(problem)
        self.assertIn("absolute", problem)

    def test_a_file_in_the_way_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "faceid"
            blocker.write_text("not a directory")
            self.assertIsNotNone(check_backup_dir(blocker))

    @unittest.skipIf(os.geteuid() == 0, "root darf auch in 0o500 schreiben")
    def test_a_read_only_directory_is_reported(self):
        # Das ist der /media-Fall im HA-Addon: der Ordner existiert, ist lesbar,
        # und erst der Schreibversuch zeigt, dass kein Backup entstehen kann.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "ro"
            target.mkdir()
            target.chmod(0o500)
            try:
                problem = check_backup_dir(target)
            finally:
                target.chmod(0o700)
            self.assertIsNotNone(problem, "read-only muss auffallen")
            self.assertIn("cannot write", problem)

    def test_the_probe_leaves_nothing_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            check_backup_dir(tmp)
            self.assertEqual(sorted(os.listdir(tmp)), [])


class PruneBackupsTests(unittest.TestCase):
    def test_only_own_archives_are_deleted(self):
        # Gegenprobe zum Verdacht, die Rotation koenne fremde Dateien loeschen.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ("faceid-backup-20250101-000000.tar.gz",
                         "faceid-backup-20250102-000000.tar.gz",
                         "faceid-backup-20250103-000000.tar.gz",
                         "family-photos.tar.gz", "notes.txt", "movie.mkv"):
                (d / name).write_text("x")
            prune_backups(d, keep=1)
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             ["faceid-backup-20250103-000000.tar.gz",
                              "family-photos.tar.gz", "movie.mkv", "notes.txt"])

    def test_keep_zero_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "faceid-backup-20250101-000000.tar.gz").write_text("x")
            prune_backups(d, keep=0)
            self.assertEqual(len(list(d.iterdir())), 1)


class WriteBackupFileTests(unittest.TestCase):
    def test_a_read_only_target_raises_instead_of_writing_nothing(self):
        if os.geteuid() == 0:
            self.skipTest("root darf auch in 0o500 schreiben")
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            (data / "persons" / "anna").mkdir(parents=True)
            (data / "persons" / "anna" / "1.jpg").write_bytes(b"jpg")
            target = Path(tmp) / "ro"
            target.mkdir()
            target.chmod(0o500)
            try:
                with self.assertRaises(OSError):
                    write_backup_file(data, target)
            finally:
                target.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
