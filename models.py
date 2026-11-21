"""
COGNITIVE MEMORY OPERATING SYSTEM
Core Data Models — The atomic units of autobiographical memory

These are not generic document schemas.
They are psychological + temporal + emotional data structures
designed for personal cognition, behavioral reasoning, and identity modeling.
"""

from __future__ import annotations
from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────

class MemoryType(str, Enum):
    EPISODE   = "episode"       # discrete life event
    INSIGHT   = "insight"       # realized lesson / realization
    PATTERN   = "pattern"       # recurring behavioral loop
    BELIEF    = "belief"        # stated or implicit conviction
    TENSION   = "tension"       # unresolved internal conflict
    DECISION  = "decision"      # a documented choice
    VALUE     = "value"         # expressed life value
    GOAL      = "goal"          # stated aspiration
    FEAR      = "fear"          # expressed fear
    RELATION  = "relation"      # relationship snapshot


class EmotionLabel(str, Enum):
    # Plutchik's 8 primary + key derivatives
    JOY          = "joy"
    TRUST        = "trust"
    FEAR         = "fear"
    SURPRISE     = "surprise"
    SADNESS      = "sadness"
    DISGUST      = "disgust"
    ANGER        = "anger"
    ANTICIPATION = "anticipation"
    # Derivatives — common in personal writing
    GRIEF        = "grief"
    BURNOUT      = "burnout"
    SHAME        = "shame"
    PRIDE        = "pride"
    LONELINESS   = "loneliness"
    LOVE         = "love"
    EXCITEMENT   = "excitement"
    DISILLUSION  = "disillusionment"
    RELIEF       = "relief"
    REGRET       = "regret"
    GRATITUDE    = "gratitude"
    UNKNOWN      = "unknown"


class LifeDomain(str, Enum):
    WORK         = "work"
    RELATIONSHIPS= "relationships"
    HEALTH       = "health"
    CREATIVITY   = "creativity"
    FINANCES     = "finances"
    SPIRITUALITY = "spirituality"
    FAMILY       = "family"
    FRIENDSHIP   = "friendship"
    IDENTITY     = "identity"
    LEARNING     = "learning"
    TRAVEL       = "travel"
    GENERAL      = "general"


class SourceType(str, Enum):
    HANDWRITTEN_DIARY  = "handwritten_diary"
    TYPED_NOTE         = "typed_note"
    VOICE_TRANSCRIPT   = "voice_transcript"
    SCREENSHOT         = "screenshot_note"
    PHOTO_WITH_TEXT    = "photo_with_text"
    STRUCTURED_EXPORT  = "structured_export"   # Notion, Day One, etc.
    REFLECTION_SESSION = "reflection_session"  # user typed directly into system


class DateConfidence(float, Enum):
    EXACT    = 1.0   # from EXIF or explicit date string
    HIGH     = 0.9   # date in entry text, unambiguous
    MEDIUM   = 0.7   # inferred from context + surrounding entries
    LOW      = 0.5   # rough period only ("sometime in early 2021")
    UNKNOWN  = 0.1   # cannot determine, guessed


class PatternStatus(str, Enum):
    ACTIVE   = "active"     # still occurring
    RESOLVED = "resolved"   # broken / overcome
    DORMANT  = "dormant"    # not recently active, may resurface
    EMERGING = "emerging"   # only 2-3 instances, not confirmed


class BeliefTrend(str, Enum):
    STRENGTHENING = "strengthening"
    WEAKENING     = "weakening"
    STABLE        = "stable"
    CONTRADICTED  = "contradicted"
    EVOLVED       = "evolved_into_new"


# ─────────────────────────────────────────────
# SUB-MODELS
# ─────────────────────────────────────────────

class EmotionalProfile(BaseModel):
    """Rich emotional representation of a memory moment."""
    primary: EmotionLabel = EmotionLabel.UNKNOWN
    intensity: float = Field(0.0, ge=0.0, le=1.0, description="0=neutral, 1=overwhelming")
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="-1=negative, +1=positive")
    arousal: float = Field(0.0, ge=0.0, le=1.0, description="0=calm, 1=activated")
    secondary: list[EmotionLabel] = Field(default_factory=list)
    source_language: list[str] = Field(
        default_factory=list,
        description="Exact words/phrases from original text that signal this emotion"
    )
    somatic_markers: list[str] = Field(
        default_factory=list,
        description="Physical sensations mentioned: 'chest tightness', 'couldn't sleep'"
    )
    context_note: Optional[str] = None


class DateInfo(BaseModel):
    """Temporal metadata with confidence — critical for chronological integrity."""
    occurred: Optional[date] = None
    documented: Optional[date] = None
    confidence: float = Field(DateConfidence.UNKNOWN, ge=0.0, le=1.0)
    is_inferred: bool = False
    inference_method: Optional[str] = None   # "exif", "text_explicit", "contextual", "approximate"
    period_label: Optional[str] = None       # "early 2021", "post-breakup period"
    life_phase: Optional[str] = None         # "startup_phase_2", "post-exit-transition"
    approximate_only: bool = False


class PersonRef(BaseModel):
    """Privacy-preserving person reference."""
    alias: str          # "cofounder_A", "therapist_B", "partner_2019"
    role: str           # "business partner", "romantic partner", "mentor"
    relationship_type: Optional[str] = None
    emotional_valence_at_time: Optional[float] = None  # how you felt about them at this memory


class SourceProvenance(BaseModel):
    """Full traceability back to the original source document."""
    source_doc_id: str
    source_type: SourceType
    file_path: Optional[str] = None     # original file location
    page_reference: Optional[str] = None   # "page 4, lines 3-22"
    ocr_confidence: Optional[float] = None
    was_corrected: bool = False
    correction_method: Optional[str] = None   # "llm_vision_pass", "human_review"


# ─────────────────────────────────────────────
# CORE MEMORY OBJECT
# ─────────────────────────────────────────────

class MemoryObject(BaseModel):
    """
    The atomic unit of the Cognitive Memory Operating System.

    Not a document chunk. Not a database row.
    A structured representation of one unit of personal experience —
    with emotional depth, temporal precision, behavioral context,
    and full provenance.
    """
    id: str = Field(default_factory=lambda: f"mem_{uuid4().hex[:12]}")
    version: int = 1
    memory_type: MemoryType

    # Content
    title: str = Field(..., description="Concise, specific label for this memory")
    content: str = Field(..., description="The core content — event description, insight text, pattern description")
    original_language: Optional[str] = Field(
        None,
        description="Verbatim quote or close paraphrase from source. Preserves the user's voice."
    )

    # Temporal
    date_info: DateInfo = Field(default_factory=DateInfo)

    # Emotional
    emotion: EmotionalProfile = Field(default_factory=EmotionalProfile)

    # Classification
    domains: list[LifeDomain] = Field(default_factory=list)
    pattern_tags: list[str] = Field(
        default_factory=list,
        description="Personal taxonomy tags: 'burnout-cycle', 'fear-then-leap', 'overcommitment'"
    )
    people_involved: list[PersonRef] = Field(default_factory=list)

    # Quality + Importance
    salience_score: float = Field(0.5, ge=0.0, le=1.0)
    is_sensitive: bool = False
    sensitivity_reason: Optional[str] = None

    # Relationships to other memories
    reinforces: list[str] = Field(default_factory=list, description="memory IDs this strengthens")
    contradicts: list[str] = Field(default_factory=list, description="memory IDs this conflicts with")
    derived_from: list[str] = Field(default_factory=list, description="source episode IDs (for insights/patterns)")
    caused_by: list[str] = Field(default_factory=list)
    led_to: list[str] = Field(default_factory=list)

    # Payloads per type (optional, type-specific)
    lesson_content: Optional[str] = None          # for INSIGHT type
    decision_payload: Optional[dict] = None       # for DECISION type
    pattern_payload: Optional[dict] = None        # for PATTERN type
    belief_payload: Optional[dict] = None         # for BELIEF type
    relation_payload: Optional[dict] = None       # for RELATION type

    # Provenance
    provenance: Optional[SourceProvenance] = None

    # System metadata
    reinforcement_count: int = 0   # how many times this memory was retrieved and used
    last_retrieved: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Embedding IDs (references to vectors stored in Qdrant)
    embedding_ids: dict[str, str] = Field(
        default_factory=dict,
        description="{'semantic': 'qdrant_point_id', 'emotional': 'qdrant_point_id', ...}"
    )

    @field_validator("salience_score")
    @classmethod
    def clamp_salience(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def compute_salience(
        self,
        days_since_created: int = 0,
        user_importance_flag: bool = False
    ) -> float:
        """
        Salience Score Formula:

        score = 0.30 × emotional_intensity
              + 0.25 × recency_weight(days)
              + 0.20 × reinforcement_normalized
              + 0.15 × outcome_significance
              + 0.10 × user_importance_flag

        recency_weight uses half-life = 365 days
        """
        import math

        emotional = self.emotion.intensity
        recency = math.exp(-0.693 * days_since_created / 365)
        reinforcement = min(1.0, self.reinforcement_count / 20.0)
        outcome_sig = 1.0 if self.memory_type in (
            MemoryType.DECISION, MemoryType.INSIGHT, MemoryType.PATTERN
        ) else 0.5
        importance = 1.0 if user_importance_flag else 0.0

        score = (
            0.30 * emotional
            + 0.25 * recency
            + 0.20 * reinforcement
            + 0.15 * outcome_sig
            + 0.10 * importance
        )
        self.salience_score = round(score, 4)
        return self.salience_score

    class Config:
        use_enum_values = True


# ─────────────────────────────────────────────
# SPECIALIZED MEMORY PAYLOADS
# ─────────────────────────────────────────────

class EpisodePayload(BaseModel):
    """Payload for EPISODE type memories — discrete life events."""
    event_type: str                  # "career_exit", "relationship_start", "confrontation"
    trigger: Optional[str] = None
    consequence: Optional[str] = None
    emotional_arc: dict = Field(
        default_factory=dict,
        description="{'before': 'dread', 'during': 'clarity', 'after': 'relief'}"
    )
    duration: Optional[str] = None   # "single_event", "2_weeks", "3_months"
    outcome_at_6mo: Optional[str] = None
    outcome_at_24mo: Optional[str] = None


class DecisionPayload(BaseModel):
    """Payload for DECISION type memories."""
    decision_type: str     # "exit", "commit", "delay", "reject", "accept"
    stakes: str            # "high", "medium", "low"
    time_to_decide: Optional[str] = None
    consulted: list[str] = Field(default_factory=list)   # roles, not names
    primary_driver: Optional[str] = None    # "emotional exhaustion", "opportunity"
    stated_reasoning: Optional[str] = None
    actual_reasoning: Optional[str] = None  # what behavior reveals, vs what was stated
    emotional_state_at_decision: Optional[str] = None
    outcome_short: Optional[str] = None
    outcome_long: Optional[str] = None
    lesson: Optional[str] = None
    regretted: Optional[bool] = None
    would_repeat: Optional[bool] = None
    bias_flags: list[str] = Field(
        default_factory=list,
        description="Detected cognitive biases: 'loss_aversion', 'overconfidence'"
    )


class PatternPayload(BaseModel):
    """Payload for PATTERN type memories — behavioral loops across time."""
    trigger_description: str
    behavior_description: str
    outcome_description: str
    interpretation_description: str  # what you told yourself about it
    instance_count: int = 0
    first_occurrence: Optional[str] = None   # "2018-Q4"
    most_recent: Optional[str] = None
    status: PatternStatus = PatternStatus.ACTIVE
    significance_score: float = 0.5
    is_adaptive: Optional[bool] = None
    resolution_notes: Optional[str] = None
    domain_spread: list[LifeDomain] = Field(
        default_factory=list,
        description="Does this pattern appear across multiple life domains?"
    )


class BeliefPayload(BaseModel):
    """Payload for BELIEF type memories — convictions about self/world."""
    belief_statement: str
    belief_about: str       # "self", "others", "work", "relationships", "world"
    is_explicit: bool       # stated directly vs inferred from behavior
    confidence_in_belief: float = 0.7   # how strongly was this held
    supporting_behaviors: list[str] = Field(default_factory=list)
    contradicting_behaviors: list[str] = Field(default_factory=list)
    trend: BeliefTrend = BeliefTrend.STABLE
    evolved_from: Optional[str] = None     # previous version belief_id
    evolved_to: Optional[str] = None       # next version belief_id
    is_limiting: Optional[bool] = None     # is this a limiting belief?


class RelationshipPayload(BaseModel):
    """Payload for RELATION type memories — relationship snapshots."""
    person_alias: str
    relationship_type: str
    dynamic_description: str
    emotional_valence: float   # -1 to +1
    trust_level: float         # 0 to 1
    power_dynamic: Optional[str] = None    # "equal", "deferential", "leading"
    recurring_conflict_themes: list[str] = Field(default_factory=list)
    growth_moments: list[str] = Field(default_factory=list)
    period: Optional[str] = None


# ─────────────────────────────────────────────
# SOURCE DOCUMENT
# ─────────────────────────────────────────────

class SourceDocument(BaseModel):
    """Raw ingested document — the immutable ground truth."""
    id: str = Field(default_factory=lambda: f"src_{uuid4().hex[:12]}")
    original_filename: Optional[str] = None
    source_type: SourceType
    date_info: DateInfo = Field(default_factory=DateInfo)
    content_raw: Optional[str] = None        # raw OCR / transcript
    content_corrected: Optional[str] = None  # LLM-corrected
    content_passages: list[str] = Field(
        default_factory=list,
        description="Semantically chunked passages for retrieval"
    )
    visual_annotations: list[str] = Field(
        default_factory=list,
        description="Detected visual elements: 'underline_paragraph_3', 'star_margin', 'crossed_out_line_7'"
    )
    processing_status: str = "pending"   # pending, processing, complete, failed, review_needed
    ocr_confidence_avg: Optional[float] = None
    needs_human_review: bool = False
    memory_object_ids: list[str] = Field(default_factory=list)
    file_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# LIFE PATTERN (Top-level, aggregates memory objects)
# ─────────────────────────────────────────────

class LifePattern(BaseModel):
    """
    A detected recurring behavioral/psychological pattern.
    Built from multiple MemoryObject instances.
    The most valuable output of the pattern detection system.
    """
    id: str = Field(default_factory=lambda: f"ptn_{uuid4().hex[:10]}")
    name: str
    description: str
    loop_description: str = Field(
        ...,
        description="trigger → emotion → behavior → outcome → interpretation"
    )
    payload: PatternPayload
    supporting_memory_ids: list[str] = Field(default_factory=list)
    significance_score: float = 0.5
    domains: list[LifeDomain] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# CONTRADICTION
# ─────────────────────────────────────────────

class Contradiction(BaseModel):
    """A detected tension between two memories — belief/action/statement."""
    id: str = Field(default_factory=lambda: f"con_{uuid4().hex[:10]}")
    type: str   # "value_action", "lesson_repetition", "self_concept", "temporal_belief"
    memory_a_id: str
    memory_b_id: str
    description: str
    contradiction_score: float = Field(0.0, ge=0.0, le=1.0)
    temporal_gap_days: Optional[int] = None
    same_domain: bool = False
    is_acknowledged: bool = False
    resolution: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# PSYCHOLOGICAL MODEL
# ─────────────────────────────────────────────

class PsychologicalModel(BaseModel):
    """
    The living psychological model of the user.
    Updated continuously as new data is ingested.
    This is the 'who you are' layer — not just 'what happened'.
    """
    user_id: str = "self"
    model_version: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m"))

    emotional_defaults: dict[str, list[str]] = Field(
        default_factory=dict,
        description="{'work': ['anxious-achievement', 'driven'], 'relationships': ['cautious-warmth']}"
    )
    core_fears: list[str] = Field(default_factory=list)
    core_motivators: list[str] = Field(default_factory=list)
    coping_strategies_adaptive: list[str] = Field(default_factory=list)
    coping_strategies_maladaptive: list[str] = Field(default_factory=list)

    decision_style: dict = Field(
        default_factory=dict,
        description="{'speed': 'slow_deliberate', 'primary_driver': 'emotional_clarity', 'bias_flags': [...]}"
    )

    attachment_patterns: dict = Field(default_factory=dict)
    self_concept_beliefs: list[str] = Field(default_factory=list)

    # Value system
    espoused_values: dict[str, float] = Field(
        default_factory=dict,
        description="{'creative_freedom': 0.9, 'family': 0.8}"
    )
    enacted_values: dict[str, float] = Field(
        default_factory=dict,
        description="Derived from behavioral evidence — may differ from espoused"
    )

    # Evidence tracking
    evidence_entry_count: int = 0
    confidence_overall: float = 0.5
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def value_gap(self, value: str) -> float:
        """Espoused vs enacted gap — a tension indicator."""
        espoused = self.espoused_values.get(value, 0.0)
        enacted = self.enacted_values.get(value, 0.0)
        return round(espoused - enacted, 3)


# ─────────────────────────────────────────────
# IDENTITY SNAPSHOT
# ─────────────────────────────────────────────

class IdentitySnapshot(BaseModel):
    """
    A snapshot of a specific identity dimension at a point in time.
    Multiple snapshots form the identity evolution timeline.
    """
    id: str = Field(default_factory=lambda: f"idt_{uuid4().hex[:10]}")
    dimension: str      # "relationship_with_work", "self_worth_sources", "risk_tolerance"
    period: str         # "2019-2020", "post-exit-2021"
    description: str
    evidence_strength: float = Field(0.5, ge=0.0, le=1.0)
    supporting_memory_ids: list[str] = Field(default_factory=list)
    evolved_from_id: Optional[str] = None
    evolved_to_id: Optional[str] = None
    trajectory: str = "stable"    # "positive", "negative", "stable", "regressing"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# RETRIEVAL RESULT
# ─────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """A ranked, scored retrieval result from the memory store."""
    memory: MemoryObject
    retrieval_score: float       # final combined score
    semantic_score: float = 0.0
    emotional_score: float = 0.0
    salience_contribution: float = 0.0
    temporal_score: float = 0.0
    retrieval_path: str = "vector"   # "vector", "graph", "metadata_filter", "keyword"
    graph_hops: int = 0              # 0 = direct match, 1+ = via graph traversal


class ContextPackage(BaseModel):
    """
    The assembled context passed to the reasoning agent.
    Contains ranked memories, psychological context, and metadata
    for generating a grounded, cited response.
    """
    query: str
    query_intent: str           # "pattern_retrieval", "specific_recall", "decision_support"
    query_domains: list[LifeDomain] = Field(default_factory=list)
    time_filter: Optional[tuple[date, date]] = None

    retrieved_memories: list[RetrievalResult] = Field(default_factory=list)
    active_patterns: list[LifePattern] = Field(default_factory=list)
    relevant_contradictions: list[Contradiction] = Field(default_factory=list)
    psych_model_snapshot: Optional[dict] = None

    total_memories_searched: int = 0
    retrieval_latency_ms: float = 0.0
    assembled_at: datetime = Field(default_factory=datetime.utcnow)
