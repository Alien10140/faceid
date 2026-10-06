"""Backup der Galerie (persons + ignored) als tar.gz — geteilt von API und Auto-Scheduler."""
import io
import os
import logging
import tarfile
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger("faceid.backup")

# Nur die unersetzliche Handarbeit sichern — nicht die Unknown-Queue oder Frigate-Vollbilder.
BACKUP_SUBDIRS = ("persons", "ignored")


def build_backup_gz(data_dir: Path) -> bytes:
    """Aktuelle Galerie als gzip-tar-Bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in BACKUP_SUBDIRS:
            d = data_dir / sub
            if d.exists():
                tar.add(d, arcname=sub)
    return buf.getvalue()


def _persistent_roots() -> list[Path]:
    """Die Mounts, die einen Neustart ueberleben — leer heisst: nicht pruefen.

    Im Home-Assistant-Addon setzt run.sh FACEID_PERSISTENT_ROOTS, weil dort alles
    ausserhalb der Mounts im Overlay liegt und beim Update verschwindet. Standalone
    (Docker, LXC, bare metal) gibt es diese Grenze nicht, also wird dort nichts
    eingeschraenkt.
    """
    raw = os.environ.get("FACEID_PERSISTENT_ROOTS", "")
    # Auch die Wurzeln aufloesen: sonst vergleicht man einen aufgeloesten Zielpfad
    # gegen einen Symlink und lehnt ein voellig gueltiges Ziel ab.
    return [Path(r).resolve() for r in raw.split(":") if r.strip()]


def check_backup_dir(backup_dir) -> str | None:
    """Pruefen, ob dort wirklich geschrieben werden kann. Fehlertext oder None.

    Ein Backup-Ziel faellt sonst erst beim Wiederherstellen auf: der Scheduler
    loggt den Fehler und laeuft weiter, und der Nutzer glaubt, er habe Backups.
    Deshalb einmal echt hinschreiben statt os.access zu fragen — im HA-Addon
    ist /media read-only gemountet, und das sieht man nur am Schreibversuch.
    """
    given = Path(str(backup_dir))
    if not given.is_absolute():
        return f"{given} is not an absolute path"
    # Aufloesen, bevor irgendetwas geprueft wird: path.parents ist rein lexikalisch,
    # also kaeme /share/../config/faceid als "unter /share" durch und wuerde dann in
    # /config landen. resolve() nimmt .. heraus und folgt Symlinks im vorhandenen Teil.
    path = given.resolve()
    # Erst die Lage pruefen, dann anlegen: ein Tippfehler wie /shre/faceid ist im
    # Container schreibbar, liegt aber im Overlay und ist nach dem naechsten Update
    # weg. Ein Backup dort ist schlimmer als keins, weil es keins zu sein scheint.
    roots = _persistent_roots()
    if roots and not any(path == r or r in path.parents for r in roots):
        where = ", ".join(str(r) for r in roots)
        shown = f"{given} (resolves to {path})" if path != given else str(given)
        return f"{shown} is not inside a mount that survives a restart ({where})"
    # Merken, was wir selbst anlegen: scheitert die Probe, soll die Pruefung keine
    # leeren Verzeichnisse hinterlassen.
    created = []
    probe_parent = path
    while not probe_parent.exists() and probe_parent != probe_parent.parent:
        created.append(probe_parent)
        probe_parent = probe_parent.parent
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"cannot create {path}: {exc.strerror or exc}"
    problem = None
    if not path.is_dir():
        problem = f"{path} is not a directory"
    else:
        try:
            with tempfile.NamedTemporaryFile(dir=path, prefix=".faceid-write-test-") as probe:
                probe.write(b"faceid")
        except OSError as exc:
            problem = f"cannot write to {path}: {exc.strerror or exc}"
    if problem:
        for d in created:
            try:
                d.rmdir()
            except OSError:
                break  # nicht leer oder nicht unser — stehen lassen
    return problem


def write_backup_file(data_dir: Path, backup_dir: Path) -> Path:
    """Archiv atomar schreiben: erst unter einem Namen, den prune_backups nicht sieht.

    Bricht das Schreiben ab (kein Platz, I/O-Fehler), bliebe sonst ein abgeschnittenes
    faceid-backup-*.tar.gz liegen — und weil es das neueste ist, wuerde die Rotation ein
    gueltiges aelteres dafuer wegwerfen. Ein gescheitertes Backup darf kein gutes kosten.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = backup_dir / f"faceid-backup-{ts}.tar.gz"
    tmp = backup_dir / f".faceid-backup-{ts}.part"
    try:
        tmp.write_bytes(build_backup_gz(data_dir))
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return path


def prune_backups(backup_dir: Path, keep: int):
    if keep <= 0:
        return
    files = sorted(backup_dir.glob("faceid-backup-*.tar.gz"), reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)


def start_auto_backup(cfg_faceid: dict, data_dir: Path):
    """Täglicher Backup-Thread, wenn faceid.backup_enabled gesetzt ist.
    Liest die Config bei jedem Tick neu (Settings-Tab wirkt live)."""
    def loop():
        last_day = None
        while True:
            try:
                if cfg_faceid.get("backup_enabled"):
                    hour = int(cfg_faceid.get("backup_hour", 3))
                    now = time.localtime()
                    day = time.strftime("%Y-%m-%d", now)
                    if now.tm_hour >= hour and day != last_day:
                        backup_dir = Path(cfg_faceid.get("backup_dir") or (data_dir / "backups"))
                        p = write_backup_file(data_dir, backup_dir)
                        prune_backups(backup_dir, int(cfg_faceid.get("backup_keep", 7)))
                        last_day = day
                        log.info("auto backup written: %s", p)
            except Exception:
                log.exception("auto backup failed")
            time.sleep(300)  # alle 5 Min prüfen

    threading.Thread(target=loop, daemon=True, name="faceid-autobackup").start()
