-- ══════════════════════════════════════════════════════════
-- COGNITIVE MEMORY OPERATING SYSTEM
-- PostgreSQL Schema — v1.0
--
-- Tables:
--   source_documents      raw ingested files + OCR output
--   memory_objects        extracted, structured memories
--   life_patterns         detected behavioral loops
--   pattern_instances     linking table: memory → pattern
--   beliefs               versioned belief records
--   decisions             documented decision objects
--   people                person profiles (aliased for privacy)
--   person_mentions       memory ↔ person linking
--   life_phases           defined life periods
--   contradictions        detected contradiction pairs
--   values_espoused       stated values with strength
--   values_enacted        behaviorally evidenced values
--   identity_snapshots    identity dimension evolution
--   emotional_triggers    detected trigger patterns
--   behavioral_scripts    situation → behavior mappings
--   reflections           long-term reflection records
--   wisdom_principles     crystallized personal wisdom
--   query_log             audit trail of all queries
--   user_corrections      user-provided memory corrections
--   consolidation_log     weekly/monthly consolidation records
-- ══════════════════════════════════════════════════════════

-- ── EXTENSIONS ──────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- fuzzy text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- GIN index support

-- ── ENUMERATIONS ────────────────────────────────────────

CREATE TYPE memory_type_enum AS ENUM (
    'episode', 'insight', 'pattern', 'belief',
    'tension', 'decision', 'value', 'goal', 'fear', 'relation'
);

CREATE TYPE source_type_enum AS ENUM (
    'handwritten_diary', 'typed_note', 'voice_transcript',
    'screenshot_note', 'photo_with_text', 'structured_export',
    'reflection_session'
);

CREATE TYPE emotion_enum AS ENUM (
    'joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust',
    'anger', 'anticipation', 'grief', 'burnout', 'shame', 'pride',
    'loneliness', 'love', 'excitement', 'disillusionment', 'relief',
    'regret', 'gratitude', 'unknown'
);

CREATE TYPE domain_enum AS ENUM (
    'work', 'relationships', 'health', 'creativity', 'finances',
    'spirituality', 'family', 'friendship', 'identity', 'learning',
    'travel', 'general'
);

CREATE TYPE pattern_status_enum AS ENUM (
    'active', 'resolved', 'dormant', 'emerging'
);

CREATE TYPE processing_status_enum AS ENUM (
    'pending', 'processing', 'complete', 'failed', 'review_needed'
);

-- ── SOURCE DOCUMENTS ────────────────────────────────────

CREATE TABLE source_documents (
    id                  VARCHAR(64) PRIMARY KEY,
    original_filename   TEXT,
    source_type         source_type_enum NOT NULL,
    date_occurred       DATE,
    date_confidence     FLOAT DEFAULT 0.5,
    date_is_inferred    BOOLEAN DEFAULT FALSE,
    date_inference_method TEXT,
    life_phase          TEXT,
    content_raw         TEXT,
    content_corrected   TEXT,
    content_passages    TEXT[],           -- semantic passage array
    visual_annotations  TEXT[],           -- underlines, stars, etc.
    processing_status   processing_status_enum DEFAULT 'pending',
    ocr_confidence_avg  FLOAT,
    needs_human_review  BOOLEAN DEFAULT FALSE,
    file_hash           VARCHAR(64) UNIQUE,
    memory_object_ids   VARCHAR(64)[],
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_source_docs_date ON source_documents (date_occurred);
CREATE INDEX idx_source_docs_status ON source_documents (processing_status);
CREATE INDEX idx_source_docs_phase ON source_documents (life_phase);

-- ── MEMORY OBJECTS ──────────────────────────────────────

CREATE TABLE memory_objects (
    id                      VARCHAR(64) PRIMARY KEY,
    version                 INTEGER DEFAULT 1,
    memory_type             memory_type_enum NOT NULL,
    title                   TEXT NOT NULL,
    content                 TEXT NOT NULL,
    original_language       TEXT,          -- verbatim quote from source

    -- Temporal
    date_occurred           DATE,
    date_documented         DATE,
    date_confidence         FLOAT DEFAULT 0.5,
    date_is_inferred        BOOLEAN DEFAULT FALSE,
    date_inference_method   TEXT,
    life_phase              TEXT,
    approximate_period      TEXT,          -- "early 2021", "post-burnout"

    -- Emotional
    emotion_primary         emotion_enum DEFAULT 'unknown',
    emotion_intensity       FLOAT DEFAULT 0.0 CHECK (emotion_intensity BETWEEN 0 AND 1),
    valence                 FLOAT DEFAULT 0.0 CHECK (valence BETWEEN -1 AND 1),
    arousal                 FLOAT DEFAULT 0.0 CHECK (arousal BETWEEN 0 AND 1),
    emotion_secondary       TEXT[],
    source_language         TEXT[],        -- exact words that signal emotion
    somatic_markers         TEXT[],        -- physical sensations mentioned

    -- Classification
    domains                 domain_enum[],
    pattern_tags            TEXT[],
    has_lesson              BOOLEAN DEFAULT FALSE,
    lesson_content          TEXT,
    has_decision            BOOLEAN DEFAULT FALSE,

    -- Quality
    salience_score          FLOAT DEFAULT 0.5 CHECK (salience_score BETWEEN 0 AND 1),
    is_sensitive            BOOLEAN DEFAULT FALSE,
    sensitivity_reason      TEXT,

    -- Type-specific payloads (JSONB for flexibility)
    episode_payload         JSONB,
    decision_payload        JSONB,
    pattern_payload         JSONB,
    belief_payload          JSONB,
    relation_payload        JSONB,

    -- Relationships to other memories
    reinforces              VARCHAR(64)[],
    contradicts             VARCHAR(64)[],
    derived_from            VARCHAR(64)[],
    caused_by               VARCHAR(64)[],
    led_to                  VARCHAR(64)[],

    -- Embedding IDs (references to Qdrant points)
    embedding_ids           JSONB DEFAULT '{}',

    -- Provenance
    source_doc_id           VARCHAR(64) REFERENCES source_documents(id) ON DELETE SET NULL,
    source_passage_ref      TEXT,

    -- Usage tracking
    reinforcement_count     INTEGER DEFAULT 0,
    last_retrieved          TIMESTAMPTZ,

    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Critical indexes for retrieval performance
CREATE INDEX idx_mem_date ON memory_objects (date_occurred);
CREATE INDEX idx_mem_type ON memory_objects (memory_type);
CREATE INDEX idx_mem_emotion ON memory_objects (emotion_primary, emotion_intensity);
CREATE INDEX idx_mem_salience ON memory_objects (salience_score DESC);
CREATE INDEX idx_mem_phase ON memory_objects (life_phase);
CREATE INDEX idx_mem_sensitive ON memory_objects (is_sensitive);
CREATE INDEX idx_mem_domains ON memory_objects USING GIN (domains);
CREATE INDEX idx_mem_pattern_tags ON memory_objects USING GIN (pattern_tags);
CREATE INDEX idx_mem_content_trgm ON memory_objects USING GIN (content gin_trgm_ops);
CREATE INDEX idx_mem_source_doc ON memory_objects (source_doc_id);

-- Full-text search index
CREATE INDEX idx_mem_fts ON memory_objects
    USING GIN (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,'') || ' ' || coalesce(original_language,'')));

-- ── LIFE PATTERNS ───────────────────────────────────────

CREATE TABLE life_patterns (
    id                      VARCHAR(64) PRIMARY KEY,
    name                    TEXT NOT NULL,
    description             TEXT NOT NULL,
    loop_description        JSONB,         -- {trigger, behavior, outcome, interpretation}
    trigger_description     TEXT,
    behavior_description    TEXT,
    outcome_description     TEXT,
    interpretation_description TEXT,
    instance_count          INTEGER DEFAULT 0,
    first_occurrence        VARCHAR(20),   -- "2018-Q4"
    most_recent             VARCHAR(20),
    status                  pattern_status_enum DEFAULT 'active',
    significance_score      FLOAT DEFAULT 0.5,
    is_adaptive             BOOLEAN,
    domains                 domain_enum[],
    tags                    TEXT[],
    resolution_notes        TEXT,
    supporting_memory_ids   VARCHAR(64)[],
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    last_updated            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_patterns_status ON life_patterns (status);
CREATE INDEX idx_patterns_significance ON life_patterns (significance_score DESC);
CREATE INDEX idx_patterns_domains ON life_patterns USING GIN (domains);

-- ── PATTERN INSTANCES (linking table) ───────────────────

CREATE TABLE pattern_instances (
    id              SERIAL PRIMARY KEY,
    pattern_id      VARCHAR(64) REFERENCES life_patterns(id) ON DELETE CASCADE,
    memory_id       VARCHAR(64) REFERENCES memory_objects(id) ON DELETE CASCADE,
    confidence      FLOAT DEFAULT 0.8,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(pattern_id, memory_id)
);

-- ── BELIEFS (versioned) ──────────────────────────────────

CREATE TABLE beliefs (
    id              VARCHAR(64) PRIMARY KEY,
    domain          domain_enum,
    evolution_summary TEXT,
    trend           TEXT DEFAULT 'stable',
    current_version_id VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE belief_versions (
    id              VARCHAR(64) PRIMARY KEY,
    belief_id       VARCHAR(64) REFERENCES beliefs(id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    statement       TEXT NOT NULL,
    memory_id       VARCHAR(64) REFERENCES memory_objects(id),
    date_formed     DATE,
    life_phase      TEXT,
    confidence      FLOAT DEFAULT 0.7,
    is_explicit     BOOLEAN DEFAULT TRUE,
    original_language TEXT,
    evolved_from_id VARCHAR(64) REFERENCES belief_versions(id),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_belief_versions_belief ON belief_versions (belief_id, version_number);

-- ── CONTRADICTIONS ──────────────────────────────────────

CREATE TABLE contradictions (
    id                  VARCHAR(64) PRIMARY KEY,
    type                TEXT NOT NULL,   -- value_action, lesson_repetition, etc.
    memory_a_id         VARCHAR(64) REFERENCES memory_objects(id),
    memory_b_id         VARCHAR(64) REFERENCES memory_objects(id),
    description         TEXT NOT NULL,
    contradiction_score FLOAT DEFAULT 0.0 CHECK (contradiction_score BETWEEN 0 AND 1),
    temporal_gap_days   INTEGER,
    same_domain         BOOLEAN DEFAULT FALSE,
    is_acknowledged     BOOLEAN DEFAULT FALSE,
    resolution          TEXT,
    detected_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_contradictions_score ON contradictions (contradiction_score DESC);
CREATE INDEX idx_contradictions_type ON contradictions (type);

-- ── PEOPLE (privacy-preserving) ─────────────────────────

CREATE TABLE people (
    id              VARCHAR(64) PRIMARY KEY,
    alias           TEXT NOT NULL UNIQUE,  -- "cofounder_A", "partner_2019"
    role            TEXT,
    relationship_type TEXT,
    period_from     DATE,
    period_to       DATE,
    mention_count   INTEGER DEFAULT 0,
    emotional_trajectory JSONB,
    lessons         TEXT[],
    pattern_ref     VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Real names stored separately and encrypted (never in main tables)
CREATE TABLE people_name_resolution (
    alias           TEXT PRIMARY KEY REFERENCES people(alias),
    encrypted_name  BYTEA,                 -- AES-256 encrypted
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── VALUES SYSTEM ───────────────────────────────────────

CREATE TABLE values_espoused (
    id              SERIAL PRIMARY KEY,
    value_name      TEXT NOT NULL,
    strength        FLOAT DEFAULT 0.5,
    first_mentioned DATE,
    mention_count   INTEGER DEFAULT 0,
    memory_ids      VARCHAR(64)[],
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(value_name)
);

CREATE TABLE values_enacted (
    id              SERIAL PRIMARY KEY,
    value_name      TEXT NOT NULL,
    strength        FLOAT DEFAULT 0.5,
    evidence_count  INTEGER DEFAULT 0,
    memory_ids      VARCHAR(64)[],
    avg_salience    FLOAT DEFAULT 0.5,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(value_name)
);

-- ── IDENTITY SNAPSHOTS ──────────────────────────────────

CREATE TABLE identity_snapshots (
    id              VARCHAR(64) PRIMARY KEY,
    dimension       TEXT NOT NULL,
    period          TEXT NOT NULL,
    description     TEXT NOT NULL,
    evidence_strength FLOAT DEFAULT 0.5,
    supporting_memory_ids VARCHAR(64)[],
    evolved_from_id VARCHAR(64) REFERENCES identity_snapshots(id),
    evolved_to_id   VARCHAR(64) REFERENCES identity_snapshots(id),
    trajectory      TEXT DEFAULT 'stable',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_identity_dimension ON identity_snapshots (dimension);

-- ── BEHAVIORAL SCRIPTS ──────────────────────────────────

CREATE TABLE behavioral_scripts (
    id              VARCHAR(64) PRIMARY KEY,
    situation_type  TEXT NOT NULL UNIQUE,
    trigger_conditions TEXT[],
    behavioral_script TEXT NOT NULL,
    emotional_program TEXT,
    typical_outcome TEXT,
    awareness_level TEXT DEFAULT 'partial',
    adaptive_rating TEXT DEFAULT 'mixed',
    evidence_strength FLOAT DEFAULT 0.5,
    exceptions      TEXT,
    growth_trajectory TEXT DEFAULT 'stable',
    supporting_memory_ids VARCHAR(64)[],
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── REFLECTIONS ─────────────────────────────────────────

CREATE TABLE reflections (
    id              VARCHAR(64) PRIMARY KEY,
    type            TEXT NOT NULL,   -- 'weekly', 'monthly', 'annual', 'proactive'
    period          TEXT,
    arc             TEXT,
    growth          TEXT,
    persistent_loops TEXT,
    hidden_costs    TEXT,
    emerging_strengths TEXT,
    unresolved      TEXT,
    the_question    TEXT,
    full_content    TEXT,            -- full reflection text
    confidence      FLOAT DEFAULT 0.5,
    memories_covered INTEGER DEFAULT 0,
    is_read         BOOLEAN DEFAULT FALSE,
    generated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── WISDOM PRINCIPLES ───────────────────────────────────

CREATE TABLE wisdom_principles (
    id              VARCHAR(64) PRIMARY KEY,
    principle       TEXT NOT NULL,
    won_through     TEXT,
    evidence_strength FLOAT DEFAULT 0.5,
    domain          TEXT,
    caveat          TEXT,
    document_version INTEGER DEFAULT 1,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── PSYCHOLOGICAL MODEL ─────────────────────────────────

CREATE TABLE psychological_model (
    id              SERIAL PRIMARY KEY,
    model_version   TEXT NOT NULL,
    emotional_defaults JSONB DEFAULT '{}',
    core_fears      TEXT[],
    core_motivators TEXT[],
    coping_adaptive TEXT[],
    coping_maladaptive TEXT[],
    decision_style  JSONB DEFAULT '{}',
    self_concept_beliefs TEXT[],
    espoused_values JSONB DEFAULT '{}',
    enacted_values  JSONB DEFAULT '{}',
    evidence_entry_count INTEGER DEFAULT 0,
    confidence_overall FLOAT DEFAULT 0.5,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── QUERY LOG (audit + feedback) ────────────────────────

CREATE TABLE query_log (
    id              SERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    query_intent    TEXT,
    memory_ids_retrieved VARCHAR(64)[],
    pattern_ids_used VARCHAR(64)[],
    response_text   TEXT,
    confidence_level TEXT,
    latency_ms      FLOAT,
    was_helpful     BOOLEAN,         -- user feedback
    user_correction TEXT,            -- if user corrected the response
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── USER CORRECTIONS ────────────────────────────────────

CREATE TABLE user_corrections (
    id              SERIAL PRIMARY KEY,
    memory_id       VARCHAR(64) REFERENCES memory_objects(id),
    field_corrected TEXT,
    original_value  TEXT,
    corrected_value TEXT,
    correction_note TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── CONSOLIDATION LOG ───────────────────────────────────

CREATE TABLE consolidation_log (
    id              SERIAL PRIMARY KEY,
    type            TEXT NOT NULL,   -- 'weekly', 'monthly', 'annual'
    period          TEXT,
    memories_processed INTEGER DEFAULT 0,
    patterns_created INTEGER DEFAULT 0,
    patterns_updated INTEGER DEFAULT 0,
    contradictions_found INTEGER DEFAULT 0,
    insights_generated INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'complete',
    run_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── TRIGGERS: Auto-update updated_at ────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_memory_updated_at
    BEFORE UPDATE ON memory_objects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_source_doc_updated_at
    BEFORE UPDATE ON source_documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_pattern_updated_at
    BEFORE UPDATE ON life_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── VIEWS FOR COMMON QUERIES ─────────────────────────────

-- High-salience memories with source context
CREATE VIEW v_salient_memories AS
SELECT
    m.*,
    s.original_filename,
    s.source_type as source_file_type
FROM memory_objects m
LEFT JOIN source_documents s ON m.source_doc_id = s.id
WHERE m.salience_score > 0.6
  AND m.is_sensitive = FALSE
ORDER BY m.salience_score DESC, m.date_occurred DESC;

-- Active patterns with instance counts
CREATE VIEW v_active_patterns AS
SELECT
    p.*,
    COUNT(pi.memory_id) as confirmed_instances
FROM life_patterns p
LEFT JOIN pattern_instances pi ON p.id = pi.pattern_id
WHERE p.status = 'active'
GROUP BY p.id
ORDER BY p.significance_score DESC;

-- Value gap analysis
CREATE VIEW v_value_gaps AS
SELECT
    COALESCE(e.value_name, a.value_name) AS value_name,
    COALESCE(e.strength, 0) AS espoused_strength,
    COALESCE(a.strength, 0) AS enacted_strength,
    COALESCE(e.strength, 0) - COALESCE(a.strength, 0) AS gap
FROM values_espoused e
FULL OUTER JOIN values_enacted a ON e.value_name = a.value_name
ORDER BY ABS(COALESCE(e.strength, 0) - COALESCE(a.strength, 0)) DESC;

-- ── SEED DATA: Life phase templates ─────────────────────

CREATE TABLE life_phase_definitions (
    phase_key       TEXT PRIMARY KEY,
    display_name    TEXT,
    typical_themes  TEXT[],
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO life_phase_definitions (phase_key, display_name, typical_themes) VALUES
    ('university', 'University Years', ARRAY['identity formation', 'independence', 'academic pressure']),
    ('early_career', 'Early Career', ARRAY['proving myself', 'learning fast', 'financial anxiety']),
    ('startup_phase_1', 'Startup — Launch', ARRAY['excitement', 'uncertainty', 'team building']),
    ('startup_phase_2', 'Startup — Scale', ARRAY['pressure', 'identity through work', 'founder stress']),
    ('burnout_recovery', 'Burnout Recovery', ARRAY['depletion', 'identity questioning', 'rebuilding']),
    ('post_exit', 'Post-Exit Transition', ARRAY['freedom', 'identity void', 'what next']),
    ('relationship_start', 'New Relationship', ARRAY['vulnerability', 'hope', 'attachment patterns']),
    ('relationship_end', 'Relationship Ending', ARRAY['grief', 'identity reconstruction', 'lessons']),
    ('grief_period', 'Grief & Loss', ARRAY['processing loss', 'support systems', 'meaning making']),
    ('creative_phase', 'Creative Phase', ARRAY['flow', 'self-expression', 'vulnerability of creating']),
    ('transition_period', 'Life Transition', ARRAY['uncertainty', 'possibility', 'letting go']),
    ('spiritual_seeking', 'Spiritual Seeking', ARRAY['meaning', 'transcendence', 'inner work']),
    ('peak_performance', 'Peak Performance Phase', ARRAY['flow', 'achievement', 'sustainability questions']);
