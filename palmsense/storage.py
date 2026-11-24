"""
storage.py - SQLite database for users, protected templates and the audit log.

Privacy rules implemented here:
  * Only a pseudonymous user ID is stored (no names, no raw palm images).
  * Templates are encrypted with Fernet (AES) before being saved.
  * The audit log stores numbers only: attempt id, user id, time, score, outcome, duration.
  * Deleting a user removes the user row AND the templates (consent withdrawal).
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    hand_side    TEXT NOT NULL CHECK (hand_side IN ('left','right')),
    thumb_side   TEXT,
    n_samples    INTEGER NOT NULL,
    consent_at   TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
    user_id   TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    method    TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    data      BLOB NOT NULL,
    PRIMARY KEY (user_id, method)
);
CREATE TABLE IF NOT EXISTS auth_log (
    attempt_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    method       TEXT,
    score        REAL,
    threshold    REAL,
    score_lines  REAL,
    score_lbp    REAL,
    outcome      TEXT NOT NULL,
    reason       TEXT,
    duration_ms  REAL
);
"""


def _key() -> bytes:
    """Encryption key from .env (PALMSENSE_TEMPLATE_KEY) or a local, git-ignored key file."""
    if config.TEMPLATE_KEY:
        return config.TEMPLATE_KEY.encode()
    key_file = config.DATA_DIR / "template.key"
    if not key_file.exists():
        key_file.write_bytes(Fernet.generate_key())
    return key_file.read_bytes().strip()


def _fernet() -> Fernet:
    return Fernet(_key())


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------- users ------------------------------------- #
def user_exists(user_id: str) -> bool:
    with connect() as con:
        return con.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone() is not None


def get_user(user_id: str):
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row) if row else None


def list_users() -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM users ORDER BY created_at DESC")]


def save_enrollment(user_id: str, hand_side: str, thumb_side: str, n_samples: int, template: dict) -> None:
    f = _fernet()
    ts = now()
    with connect() as con:
        con.execute("INSERT INTO users VALUES (?,?,?,?,?,?)",
                    (user_id, hand_side, thumb_side, n_samples, ts, ts))
        for method, vec in template.items():
            blob = f.encrypt(np.asarray(vec, np.float32).tobytes())
            con.execute("INSERT INTO templates VALUES (?,?,?,?)", (user_id, method, len(vec), blob))


def load_template(user_id: str) -> dict | None:
    f = _fernet()
    with connect() as con:
        rows = con.execute("SELECT method, data FROM templates WHERE user_id=?", (user_id,)).fetchall()
    if not rows:
        return None
    try:
        return {r["method"]: np.frombuffer(f.decrypt(r["data"]), np.float32) for r in rows}
    except InvalidToken:
        raise RuntimeError("Template could not be decrypted (encryption key changed).")


def delete_user(user_id: str) -> bool:
    with connect() as con:
        cur = con.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        con.execute("DELETE FROM templates WHERE user_id=?", (user_id,))
        return cur.rowcount > 0


# ------------------------------- audit log --------------------------------- #
def log_attempt(user_id, outcome, reason="", method=None, score=None, threshold=None,
                score_lines=None, score_lbp=None, duration_ms=None) -> str:
    attempt_id = "A-" + uuid.uuid4().hex[:8].upper()
    with connect() as con:
        con.execute("INSERT INTO auth_log VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (attempt_id, user_id, now(), method, score, threshold,
                     score_lines, score_lbp, outcome, reason, duration_ms))
    return attempt_id


def read_log(limit: int = 500) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM auth_log ORDER BY timestamp DESC LIMIT ?", (limit,))]


def clear_log() -> None:
    with connect() as con:
        con.execute("DELETE FROM auth_log")
