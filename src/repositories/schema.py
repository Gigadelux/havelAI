"""The analytics schema, shipped with the package. Must equal the repo's SCHEMA.sql (tested)."""

SCHEMA_SQL = """-- havelAI analytics schema. Last updated: 2026-11-26
--
-- One row per Havel.detect() call in `detections`, one row per layer in `layer_scores`,
-- the vector layer's top matches in `vector_hits`. `learned_vectors` mirrors what the vector
-- layer stored (the vectors themselves live in Milvus Lite or Qdrant).
-- Written by src/repositories/sqlite.py (local) or by the server behind
-- src/repositories/remote_http.py (hosted). Valid for SQLite 3.35+ and PostgreSQL 13+.
-- The prompt text is never stored. vector_hits.text and learned_vectors.text come from the
-- user's own vector store.

CREATE TABLE IF NOT EXISTS detections (
    id                      TEXT PRIMARY KEY,          -- uuid4 hex
    created_at              TEXT NOT NULL,             -- ISO 8601, UTC
    mode                    TEXT NOT NULL,             -- 'local' | 'hosted'
    score                   REAL NOT NULL,             -- final weighted score, 0..1
    is_injection            INTEGER NOT NULL,          -- 0 | 1
    attack_type             TEXT NOT NULL,             -- src/data/taxonomy.py label
    detected_by             TEXT NOT NULL,             -- comma-separated layers that fired
    detect_threshold        REAL NOT NULL,
    catch_threshold         REAL,                      -- vector similarity threshold, NULL if vector off
    learn_threshold         REAL,                      -- vector learning threshold, NULL if vector off
    detection_ms            REAL NOT NULL,
    prompt_tokens           INTEGER,
    chunk_size              INTEGER,                   -- target chunk size for this prompt
    chunk_count             INTEGER,
    detecting_chunk_tokens  INTEGER,
    detecting_chunk_index   INTEGER,
    vectors_in_store        INTEGER,                   -- at detection time
    max_similarity          REAL,
    learned_chunks          INTEGER NOT NULL DEFAULT 0,
    db_type                 TEXT,                      -- 'milvus_lite' | 'qdrant' | NULL
    embedder_model          TEXT,
    embedder_dimensions     INTEGER,
    library_version         TEXT NOT NULL,
    local_model_version     TEXT,                      -- NULL when hosted
    hosted_model            TEXT,                      -- NULL when local
    hosted_endpoint         TEXT                       -- NULL when local
);

CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections (created_at);
CREATE INDEX IF NOT EXISTS idx_detections_attack_type ON detections (attack_type);

CREATE TABLE IF NOT EXISTS layer_scores (
    detection_id    TEXT NOT NULL REFERENCES detections (id) ON DELETE CASCADE,
    layer           TEXT NOT NULL,                     -- 'regex' | 'model' | 'vector'
    enabled         INTEGER NOT NULL,
    score           REAL NOT NULL,                     -- the layer's own score, 0..1
    weight          REAL NOT NULL,
    contribution    REAL NOT NULL,                     -- weight * score
    fired           INTEGER NOT NULL,
    attack_type     TEXT,
    seconds         REAL NOT NULL,
    PRIMARY KEY (detection_id, layer)
);

CREATE TABLE IF NOT EXISTS vector_hits (
    detection_id    TEXT NOT NULL REFERENCES detections (id) ON DELETE CASCADE,
    rank            INTEGER NOT NULL,                  -- 1 = most similar
    text            TEXT NOT NULL,
    similarity      REAL NOT NULL,                     -- cosine, -1..1
    source          TEXT NOT NULL,                     -- 'import' | 'learned'
    attack_type     TEXT,
    PRIMARY KEY (detection_id, rank)
);

CREATE TABLE IF NOT EXISTS learned_vectors (
    id              TEXT PRIMARY KEY,                  -- sha256(text)[:32], same id as the vector store
    text            TEXT NOT NULL,
    source          TEXT NOT NULL,                     -- 'import' | 'learned'
    attack_type     TEXT,
    score           REAL,                              -- regex+model score when learned
    embedder        TEXT,
    dimensions      INTEGER,
    created_at      TEXT NOT NULL
);
"""
