"""SQLite progress store. Segments themselves live on the persistent volume;
the DB only tracks job state and per-segment progress."""

import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'recording',   -- recording | finalizing | done | failed
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  first_seq INTEGER,          -- media sequence of the first segment of the first playlist
  last_seq INTEGER,           -- highest sequence seen in any received playlist
  stop_seq INTEGER,           -- pinned tail of the last received playlist once stop accepted
  target_duration REAL,
  error TEXT,
  zip_path TEXT
);
CREATE TABLE IF NOT EXISTS segments (
  job_id INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  url TEXT,                   -- NULL for segments the window skipped before discovery
  duration REAL,
  title TEXT,
  status TEXT NOT NULL DEFAULT 'pending',    -- pending | done | missing
  path TEXT,
  bytes INTEGER,
  error TEXT,
  PRIMARY KEY (job_id, seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_segments_job_url
  ON segments(job_id, url) WHERE url IS NOT NULL;
CREATE TABLE IF NOT EXISTS missing_ranges (
  job_id INTEGER NOT NULL,
  start_seq INTEGER NOT NULL,   -- closed interval [start_seq, end_seq], both ends missing
  end_seq INTEGER NOT NULL,
  PRIMARY KEY (job_id, start_seq)
);
"""

# A skipped span up to this many sequence numbers is still stored as one row
# per number (the original per-sequence interface lists them individually).
# Longer spans are stored as a single closed-interval row in missing_ranges,
# so a huge jump (e.g. a source reconnect skipping hundreds of thousands of
# numbers) never floods the DB row-by-row nor stalls the other jobs.
MISSING_ROW_MAX = 1000


class DB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    # ---- jobs ----

    def insert_job(self, url: str) -> int:
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO jobs(url, state, created_at, updated_at) VALUES (?,?,?,?)",
                (url, "recording", now, now),
            )
            return cur.lastrowid

    def get_job(self, job_id: int):
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_jobs(self):
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
            return [dict(r) for r in rows]

    def unfinished_jobs(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state IN ('recording','finalizing') ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    def _touch(self, job_id: int):
        self._conn.execute("UPDATE jobs SET updated_at=? WHERE id=?", (time.time(), job_id))

    def init_window(self, job_id: int, first_seq: int, last_seq: int, target: float):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET first_seq=?, last_seq=?, target_duration=? WHERE id=?",
                (first_seq, last_seq, target, job_id),
            )
            self._touch(job_id)

    def update_last_seq(self, job_id: int, last_seq: int):
        with self._lock, self._conn:
            self._conn.execute("UPDATE jobs SET last_seq=? WHERE id=?", (last_seq, job_id))
            self._touch(job_id)

    def set_stopping(self, job_id: int, stop_seq):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET state='finalizing', stop_seq=? WHERE id=? AND state='recording'",
                (stop_seq, job_id),
            )
            self._touch(job_id)

    def job_done(self, job_id: int, zip_path: str):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET state='done', zip_path=? WHERE id=?", (zip_path, job_id)
            )
            self._touch(job_id)

    def job_failed(self, job_id: int, error: str):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET state='failed', error=? WHERE id=? AND state NOT IN ('done','failed')",
                (error, job_id),
            )
            self._touch(job_id)

    # ---- segments ----

    def insert_segment(self, job_id: int, seq: int, url: str, duration: float, title: str):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO segments(job_id, seq, url, duration, title, status) VALUES (?,?,?,?,?,'pending')",
                (job_id, seq, url, duration, title),
            )

    def record_missing_span(self, job_id: int, start_seq: int, end_seq: int):
        """Persist the closed interval [start_seq, end_seq] the window skipped.

        Small spans keep per-sequence rows so the original interface can list
        every missing number; large spans become one interval row — O(1)
        writes regardless of span length, so the DB lock is held only briefly
        and other recordings/stops/queries keep responding.
        """
        count = end_seq - start_seq + 1
        with self._lock, self._conn:
            if count <= MISSING_ROW_MAX:
                for s in range(start_seq, end_seq + 1):
                    self._conn.execute(
                        "INSERT OR IGNORE INTO segments(job_id, seq, url, status, error) VALUES (?,?,NULL,'missing','窗口越过，未发现该序号')",
                        (job_id, s),
                    )
            else:
                self._conn.execute(
                    "INSERT OR IGNORE INTO missing_ranges(job_id, start_seq, end_seq) VALUES (?,?,?)",
                    (job_id, start_seq, end_seq),
                )
            self._touch(job_id)

    def missing_ranges(self, job_id: int):
        with self._lock:
            rows = self._conn.execute(
                "SELECT start_seq, end_seq FROM missing_ranges WHERE job_id=? ORDER BY start_seq",
                (job_id,),
            ).fetchall()
            return [[r["start_seq"], r["end_seq"]] for r in rows]

    def get_segment(self, job_id: int, seq: int):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM segments WHERE job_id=? AND seq=?", (job_id, seq)
            ).fetchone()
            return dict(row) if row else None

    def get_segment_by_url(self, job_id: int, url: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM segments WHERE job_id=? AND url=?", (job_id, url)
            ).fetchone()
            return dict(row) if row else None

    def next_pending(self, job_id: int):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM segments WHERE job_id=? AND status='pending' ORDER BY seq LIMIT 1",
                (job_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_segments(self, job_id: int):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE job_id=? ORDER BY seq", (job_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def segment_done(self, job_id: int, seq: int, path: str, nbytes: int):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE segments SET status='done', path=?, bytes=?, error=NULL WHERE job_id=? AND seq=?",
                (path, nbytes, job_id, seq),
            )

    def segment_pending(self, job_id: int, seq: int):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE segments SET status='pending', path=NULL, bytes=NULL WHERE job_id=? AND seq=?",
                (job_id, seq),
            )

    def segment_missing(self, job_id: int, seq: int, error: str):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE segments SET status='missing', error=? WHERE job_id=? AND seq=?",
                (error, job_id, seq),
            )

    def counts(self, job_id: int):
        with self._lock:
            row = self._conn.execute(
                "SELECT "
                "COUNT(*) AS total, "
                "SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done, "
                "SUM(CASE WHEN status='missing' THEN 1 ELSE 0 END) AS missing "
                "FROM segments WHERE job_id=?",
                (job_id,),
            ).fetchone()
            # Large skipped spans live in missing_ranges, not in segments;
            # their length still counts towards expected/missing so the
            # progress numbers stay accurate.
            skipped = self._conn.execute(
                "SELECT COALESCE(SUM(end_seq - start_seq + 1), 0) AS n "
                "FROM missing_ranges WHERE job_id=?",
                (job_id,),
            ).fetchone()["n"] or 0
            return {
                "total": (row["total"] or 0) + skipped,
                "done": row["done"] or 0,
                "missing": (row["missing"] or 0) + skipped,
            }

    def missing_seqs(self, job_id: int):
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq FROM segments WHERE job_id=? AND status='missing' ORDER BY seq",
                (job_id,),
            ).fetchall()
            return [r["seq"] for r in rows]
