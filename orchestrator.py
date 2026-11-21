"""
SYSTEM ORCHESTRATOR
Cognitive Memory OS

The master orchestrator that wires all 5 phases into a single coherent system.

Architecture:
  - IngestOrchestrator     — handles all new data entry (Phase 1)
  - IntelligenceScheduler  — runs background analysis jobs (Phase 2)
  - QueryOrchestrator      — handles user queries end-to-end (Phase 3)
  - IdentityRefresher      — updates identity model on new data (Phase 4)
  - WisdomScheduler        — runs long-term synthesis jobs (Phase 5)
  - CMOSApplication        — the unified public interface

This is the single entry point for all interactions with the system.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline import OCROrchestrator, OCRConfig
from extractor import MemoryExtractor, MetadataEnricher
from memory_repo import MultiSpaceEmbeddingEngine, QdrantMemoryStore, MemoryRetrievalOrchestrator
from detector import LifePatternDetector, ContradictionEngine, EmotionalPatternAnalyzer, TemporalPatternAnalyzer
from agent import MasterReasoningAgent, AgentResponse
from engine import (
    LifeNarrativeEngine, StrategicInsightEngine,
    PredictiveBehaviorEngine, LongTermReflectionEngine, WisdomCrystallizer
)
from models import (
    SourceDocument, SourceType, MemoryObject, PsychologicalModel
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SYSTEM CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class CMOSConfig:
    # API Keys
    gemini_api_key: str
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Models
    reasoning_model: str = "gemini-3.1-flash"
    extraction_model: str = "gemini-3.1-flash"
    analysis_model: str = "gemini-3.1-flash"
    fast_model: str = "gemini-3.1-flash"

    # Storage
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    postgres_url: str = "postgresql://cmos:cmos@localhost:5432/cmos"
    neo4j_url: str = "bolt://localhost:7687"

    # Processing
    ocr_confidence_threshold: float = 0.75
    pattern_min_instances: int = 3
    contradiction_score_threshold: float = 0.55
    salience_hot_storage_threshold: float = 0.4

    # Privacy
    anonymize_people_names: bool = True
    local_only_mode: bool = False


# ─────────────────────────────────────────────
# INGEST ORCHESTRATOR (Phase 1 pipeline)
# ─────────────────────────────────────────────

class IngestOrchestrator:
    """
    Handles the complete ingestion pipeline for a new source document.
    Phase 1 end-to-end: raw file → indexed memory objects.
    """

    def __init__(self, config: CMOSConfig, memory_repo, vector_store):
        self.config = config
        self.memory_repo = memory_repo
        self.vector_store = vector_store

        self.ocr = OCROrchestrator(
            config=OCRConfig(
                min_confidence_for_llm_correction=config.ocr_confidence_threshold
            ),
            api_key=config.gemini_api_key
        )
        self.extractor = MemoryExtractor(
            api_key=config.gemini_api_key,
            model=config.extraction_model
        )
        self.enricher = MetadataEnricher(
            api_key=config.gemini_api_key,
            model=config.fast_model
        )
        self.embedder = MultiSpaceEmbeddingEngine(
            gemini_api_key=config.gemini_api_key
        )

    async def ingest_image(
        self,
        image_bytes: bytes,
        filename: str,
        hint_date: Optional[str] = None,
        hint_life_phase: Optional[str] = None
    ) -> IngestResult:
        """Ingest a diary image through the full pipeline."""
        logger.info(f"Ingesting image: {filename}")

        # Step 1: OCR
        ocr_result = await self.ocr.process_image(
            image_bytes=image_bytes,
            source_filename=filename,
            hint_date=hint_date
        )

        if not ocr_result.corrected_text:
            return IngestResult(
                success=False,
                error="OCR produced no text",
                source_doc_id=None,
                memories_created=0,
            )

        # Step 2: Create source document record
        doc_date = ocr_result.date_extracted or hint_date
        source_doc = SourceDocument(
            original_filename=filename,
            source_type=SourceType.HANDWRITTEN_DIARY,
            content_raw=ocr_result.raw_ocr_text,
            content_corrected=ocr_result.corrected_text,
            content_passages=ocr_result.semantic_passages,
            visual_annotations=ocr_result.visual_annotations,
            processing_status="processing",
            ocr_confidence_avg=ocr_result.ocr_confidence,
            needs_human_review=ocr_result.needs_human_review,
        )
        await self.memory_repo.save_source_doc(source_doc)

        # Step 3: Infer life phase if not provided
        life_phase = hint_life_phase
        if not life_phase and doc_date:
            life_phase, _ = await self.enricher.infer_life_phase(
                ocr_result.corrected_text[:500], doc_date
            )

        # Step 4: Extract memory objects from all passages
        memories = await self.extractor.extract_from_document(
            passages=ocr_result.semantic_passages,
            source_doc_id=source_doc.id,
            source_type=SourceType.HANDWRITTEN_DIARY,
            date_str=doc_date,
            life_phase=life_phase
        )

        # Step 5: Enrich with metadata and pattern tags
        for memory in memories:
            pattern_tags = self.enricher.suggest_pattern_tags(memory)
            memory.pattern_tags.extend(
                tag for tag in pattern_tags if tag not in memory.pattern_tags
            )
            memory.date_info.life_phase = life_phase

        # Step 6: Embed and index all memories
        for memory in memories:
            try:
                embeddings = await self.embedder.embed_memory(memory)
                payload = self._memory_to_payload(memory)
                self.vector_store.upsert_memory(
                    memory_id=memory.id,
                    embeddings=embeddings,
                    payload=payload
                )
                memory.embedding_ids = {
                    space: memory.id for space in embeddings.keys()
                }
            except Exception as e:
                logger.error(f"Embedding error for {memory.id}: {e}")

        # Step 7: Save to database
        for memory in memories:
            await self.memory_repo.save_memory(memory)

        # Step 8: Update source doc status
        source_doc.memory_object_ids = [m.id for m in memories]
        source_doc.processing_status = "complete"
        await self.memory_repo.update_source_doc(source_doc)

        logger.info(
            f"Ingestion complete: {filename} → {len(memories)} memories, "
            f"review_needed={ocr_result.needs_human_review}"
        )

        return IngestResult(
            success=True,
            source_doc_id=source_doc.id,
            memories_created=len(memories),
            needs_review=ocr_result.needs_human_review,
            date_extracted=doc_date,
            life_phase=life_phase,
            ocr_confidence=ocr_result.ocr_confidence,
        )

    async def ingest_text(
        self,
        text: str,
        source_type: SourceType,
        date_str: Optional[str] = None,
        hint_life_phase: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> IngestResult:
        """Ingest plain text (typed note, voice transcript, etc.)."""
        from pipeline import SemanticChunker

        chunker = SemanticChunker(api_key=self.config.gemini_api_key)
        passages = await chunker.chunk(text)

        life_phase = hint_life_phase
        if not life_phase and date_str:
            life_phase, _ = await self.enricher.infer_life_phase(text[:500], date_str)

        source_doc = SourceDocument(
            original_filename=filename or "text_input",
            source_type=source_type,
            content_raw=text,
            content_corrected=text,
            content_passages=passages,
            processing_status="processing",
        )
        await self.memory_repo.save_source_doc(source_doc)

        memories = await self.extractor.extract_from_document(
            passages=passages,
            source_doc_id=source_doc.id,
            source_type=source_type,
            date_str=date_str,
            life_phase=life_phase
        )

        for memory in memories:
            pattern_tags = self.enricher.suggest_pattern_tags(memory)
            memory.pattern_tags.extend(
                t for t in pattern_tags if t not in memory.pattern_tags
            )
            memory.date_info.life_phase = life_phase

        for memory in memories:
            try:
                embeddings = await self.embedder.embed_memory(memory)
                self.vector_store.upsert_memory(
                    memory_id=memory.id,
                    embeddings=embeddings,
                    payload=self._memory_to_payload(memory)
                )
            except Exception as e:
                logger.error(f"Embedding error for {memory.id}: {e}")

        for memory in memories:
            await self.memory_repo.save_memory(memory)

        source_doc.memory_object_ids = [m.id for m in memories]
        source_doc.processing_status = "complete"
        await self.memory_repo.update_source_doc(source_doc)

        return IngestResult(
            success=True,
            source_doc_id=source_doc.id,
            memories_created=len(memories),
            date_extracted=date_str,
            life_phase=life_phase,
        )

    def _memory_to_payload(self, memory: MemoryObject) -> dict:
        """Convert MemoryObject to a Qdrant-compatible payload dict."""
        return {
            "memory_id": memory.id,
            "memory_type": memory.memory_type if isinstance(memory.memory_type, str) else memory.memory_type.value,
            "title": memory.title,
            "date": str(memory.date_info.occurred) if memory.date_info.occurred else None,
            "life_phase": memory.date_info.life_phase,
            "emotion_primary": str(memory.emotion.primary),
            "emotion_intensity": memory.emotion.intensity,
            "valence": memory.emotion.valence,
            "domains": [d if isinstance(d, str) else d.value for d in memory.domains],
            "salience_score": memory.salience_score,
            "is_sensitive": memory.is_sensitive,
            "pattern_tags": memory.pattern_tags,
            "has_lesson": memory.lesson_content is not None,
        }


@dataclass
class IngestResult:
    success: bool
    source_doc_id: Optional[str] = None
    memories_created: int = 0
    needs_review: bool = False
    date_extracted: Optional[str] = None
    life_phase: Optional[str] = None
    ocr_confidence: Optional[float] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# INTELLIGENCE SCHEDULER (Phase 2 background jobs)
# ─────────────────────────────────────────────

class IntelligenceScheduler:
    """
    Runs background intelligence jobs:
    - Weekly pattern detection
    - Contradiction scanning
    - Emotional pattern analysis
    - Seasonal pattern detection
    """

    def __init__(self, config: CMOSConfig, memory_repo, pattern_repo, contradiction_repo):
        self.config = config
        self.memory_repo = memory_repo
        self.pattern_repo = pattern_repo
        self.contradiction_repo = contradiction_repo

        self.pattern_detector = LifePatternDetector(
            api_key=config.gemini_api_key,
            model=config.analysis_model
        )
        self.contradiction_engine = ContradictionEngine(
            api_key=config.gemini_api_key,
            model=config.analysis_model
        )
        self.emotional_analyzer = EmotionalPatternAnalyzer()
        self.temporal_analyzer = TemporalPatternAnalyzer()

    async def run_weekly_analysis(self) -> WeeklyAnalysisResult:
        """Run full weekly analysis pipeline."""
        logger.info("Starting weekly intelligence analysis")

        memories = await self.memory_repo.get_all(limit=5000)
        if not memories:
            return WeeklyAnalysisResult(patterns_found=0, contradictions_found=0)

        # Pattern detection
        new_patterns = await self.pattern_detector.detect_from_all_memories(
            memories=memories,
            embedding_store=None  # Pass actual embedding store in production
        )
        for p in new_patterns:
            await self.pattern_repo.upsert(p)
        logger.info(f"Pattern detection: {len(new_patterns)} patterns found/updated")

        # Contradiction scanning
        all_patterns = await self.pattern_repo.get_all()
        lesson_contradictions = self.contradiction_engine.find_lesson_repetition_contradictions(memories)
        for c in lesson_contradictions:
            if c.contradiction_score > self.config.contradiction_score_threshold:
                await self.contradiction_repo.save(c)

        # Emotional analysis
        triggers = self.emotional_analyzer.analyze_emotional_triggers(memories)
        logger.info(f"Emotional triggers found: {len(triggers)}")

        return WeeklyAnalysisResult(
            patterns_found=len(new_patterns),
            contradictions_found=len(lesson_contradictions),
            triggers_found=len(triggers),
            run_at=datetime.utcnow(),
        )


@dataclass
class WeeklyAnalysisResult:
    patterns_found: int
    contradictions_found: int
    triggers_found: int = 0
    run_at: datetime = None

    def __post_init__(self):
        if self.run_at is None:
            self.run_at = datetime.utcnow()


# ─────────────────────────────────────────────
# QUERY ORCHESTRATOR (Phase 3 end-to-end query handling)
# ─────────────────────────────────────────────

class QueryOrchestrator:
    """
    Handles a complete user query from input to response.
    Wires retrieval + reasoning agent together.
    """

    def __init__(
        self,
        config: CMOSConfig,
        retrieval_orchestrator: MemoryRetrievalOrchestrator,
        reasoning_agent: MasterReasoningAgent,
        psych_model_store,
    ):
        self.config = config
        self.retrieval = retrieval_orchestrator
        self.agent = reasoning_agent
        self.psych_model_store = psych_model_store
        self.conversation_history: list[dict] = []

    async def query(
        self,
        user_input: str,
        stream: bool = False,
    ) -> AgentResponse:
        """Process a user query end-to-end."""
        logger.info(f"Query: '{user_input[:80]}'")

        # Retrieve relevant context
        context_package = await self.retrieval.retrieve(
            query=user_input,
            top_k=12,
            include_patterns=True,
            include_contradictions=True,
        )

        # Inject psychological model into context
        psych_model = await self.psych_model_store.get_current()
        if psych_model:
            context_package.psych_model_snapshot = {
                "current_life_phase": getattr(psych_model, "current_life_phase", None),
                "core_fears": psych_model.core_fears[:3],
                "core_motivators": psych_model.core_motivators[:3],
                "dominant_themes": list(psych_model.espoused_values.keys())[:3],
            }

        # Generate response
        response = await self.agent.respond(
            query=user_input,
            context_package=context_package,
            conversation_history=self.conversation_history,
        )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response.response_text})

        # Keep history bounded
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response


# ─────────────────────────────────────────────
# THE UNIFIED APPLICATION
# ─────────────────────────────────────────────

class CMOSApplication:
    """
    The Cognitive Memory Operating System — unified public interface.

    All external code should interact with the system through this class.
    It provides a clean API over the internal complexity of all 5 phases.
    """

    def __init__(self, config: CMOSConfig):
        self.config = config
        self._initialized = False

        # These are set during initialize()
        self.ingest: Optional[IngestOrchestrator] = None
        self.query: Optional[QueryOrchestrator] = None
        self.intelligence: Optional[IntelligenceScheduler] = None
        self.narrative: Optional[LifeNarrativeEngine] = None
        self.insights: Optional[StrategicInsightEngine] = None
        self.prediction: Optional[PredictiveBehaviorEngine] = None
        self.reflection: Optional[LongTermReflectionEngine] = None
        self.wisdom: Optional[WisdomCrystallizer] = None

    async def initialize(self, memory_repo, pattern_repo, contradiction_repo, psych_model_store):
        """
        Initialize all system components.
        Call this once at application startup.
        """
        logger.info("Initializing Cognitive Memory Operating System...")

        # Storage layer
        vector_store = QdrantMemoryStore(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port
        )
        vector_store.initialize_collections()

        # Phase 1: Ingestion
        self.ingest = IngestOrchestrator(
            config=self.config,
            memory_repo=memory_repo,
            vector_store=vector_store,
        )

        # Phase 2: Intelligence
        self.intelligence = IntelligenceScheduler(
            config=self.config,
            memory_repo=memory_repo,
            pattern_repo=pattern_repo,
            contradiction_repo=contradiction_repo,
        )

        # Phase 3: Retrieval + Reasoning
        embedding_engine = MultiSpaceEmbeddingEngine(
            gemini_api_key=self.config.gemini_api_key
        )
        from memory_repo import GraphMemoryExpander
        graph_expander = GraphMemoryExpander()

        retrieval_orch = MemoryRetrievalOrchestrator(
            embedding_engine=embedding_engine,
            vector_store=vector_store,
            graph_expander=graph_expander,
            memory_repository=memory_repo,
            patterns_store=pattern_repo,
            contradictions_store=contradiction_repo,
        )

        reasoning_agent = MasterReasoningAgent(
            api_key=self.config.gemini_api_key,
            model=self.config.reasoning_model,
        )

        self.query = QueryOrchestrator(
            config=self.config,
            retrieval_orchestrator=retrieval_orch,
            reasoning_agent=reasoning_agent,
            psych_model_store=psych_model_store,
        )

        # Phase 5: Wisdom layer
        self.narrative = LifeNarrativeEngine(
            api_key=self.config.gemini_api_key,
            model=self.config.reasoning_model
        )
        self.insights = StrategicInsightEngine(
            api_key=self.config.gemini_api_key,
            model=self.config.reasoning_model
        )
        self.prediction = PredictiveBehaviorEngine(
            api_key=self.config.gemini_api_key,
            model=self.config.reasoning_model
        )
        self.reflection = LongTermReflectionEngine(
            api_key=self.config.gemini_api_key,
            model=self.config.reasoning_model
        )
        self.wisdom = WisdomCrystallizer(
            api_key=self.config.gemini_api_key,
            model=self.config.reasoning_model
        )

        self._initialized = True
        logger.info("CMOS initialized successfully.")

    async def ask(self, question: str) -> str:
        """
        The primary user interface.
        Ask anything about your life and receive a grounded, cited response.
        """
        self._check_initialized()
        response = await self.query.query(question)
        return response.response_text

    async def ingest_diary_image(
        self,
        image_bytes: bytes,
        filename: str,
        hint_date: Optional[str] = None,
        hint_life_phase: Optional[str] = None,
    ) -> IngestResult:
        """Ingest a scanned diary page."""
        self._check_initialized()
        return await self.ingest.ingest_image(
            image_bytes=image_bytes,
            filename=filename,
            hint_date=hint_date,
            hint_life_phase=hint_life_phase
        )

    async def ingest_note(
        self,
        text: str,
        date_str: Optional[str] = None,
        hint_life_phase: Optional[str] = None,
    ) -> IngestResult:
        """Ingest a typed note or voice transcript."""
        self._check_initialized()
        return await self.ingest.ingest_text(
            text=text,
            source_type=SourceType.TYPED_NOTE,
            date_str=date_str,
            hint_life_phase=hint_life_phase,
        )

    async def reconstruct_story(
        self,
        scope: str,
        theme: Optional[str] = None,
        memories: Optional[list] = None,
        patterns: Optional[list] = None,
    ) -> str:
        """Reconstruct a narrative arc from your life data."""
        self._check_initialized()
        arc = await self.narrative.reconstruct(
            memories=memories or [],
            patterns=patterns or [],
            scope=scope,
            theme=theme,
        )
        return arc.narrative

    async def predict_behavior(
        self,
        scenario: str,
        patterns: Optional[list] = None,
        decisions: Optional[list] = None,
        psych_model: Optional[PsychologicalModel] = None,
    ) -> str:
        """Get a pattern-grounded behavioral prediction for a future scenario."""
        self._check_initialized()
        if psych_model is None:
            psych_model = PsychologicalModel()
        pred = await self.prediction.predict(
            scenario=scenario,
            patterns=patterns or [],
            decisions=decisions or [],
            psych_model=psych_model,
        )
        return (
            f"**Most likely response:** {pred.most_likely_response}\n\n"
            f"**Emotional trajectory:** {pred.emotional_trajectory}\n\n"
            f"**If pattern holds:** {pred.likely_outcome_if_pattern_holds}\n\n"
            f"**What would need to change:** {pred.pattern_override_conditions}\n\n"
            f"**Growth edge this scenario offers:** {pred.growth_edge}\n\n"
            f"*Confidence: {pred.confidence:.0%} | Based on: {', '.join(pred.historical_parallels[:2])}*"
        )

    async def run_background_analysis(self) -> WeeklyAnalysisResult:
        """Run weekly intelligence analysis (typically called by a scheduler)."""
        self._check_initialized()
        return await self.intelligence.run_weekly_analysis()

    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "CMOSApplication not initialized. Call await app.initialize(...) first."
            )
