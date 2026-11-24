-- PalmSense SQLite schema (created automatically by palmsense/storage.py)
-- No raw palm images are stored. Templates are Fernet-encrypted float32 vectors.

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
