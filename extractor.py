"""
PHASE 1 — MEMORY EXTRACTION PIPELINE
Cognitive Memory OS

Transforms cleaned diary text into structured MemoryObject instances.

This is not document parsing. This is psychological + behavioral extraction:
- Life events with emotional arcs
- Explicit and implicit insights
- Behavioral patterns (single-entry candidates)
- Beliefs about self and world
- Unresolved tensions
- Decisions with reasoning and outcomes
- Relationship dynamics

Design principle: Extract only what is present. Never infer beyond the text.
Confidence-score everything. Preserve original language.
"""

from __future__ import annotations
import json
import logging
import re
from datetime import datetime
from typing import Optional

import httpx

from models import (
    MemoryObject, MemoryType, EmotionLabel, EmotionalProfile,
    DateInfo, LifeDomain, SourceProvenance, SourceType, PersonRef,
    EpisodePayload, DecisionPayload, BeliefPayload,
    DateConfidence, PatternStatus
)
from windows_compat import call_gemini_openai

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EXTRACTION PROMPTS
# ─────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a Cognitive Memory Extractor — a specialized AI that reads personal diary entries and extracts structured psychological memory objects.

CRITICAL RULES:
1. Extract ONLY what is explicitly stated or clearly and directly implied
2. Never infer emotions not expressed in the text — use the writer's own words
3. Preserve the EXACT language used to describe emotions (do not substitute synonyms)
4. Mark uncertainty with confidence scores (0.0–1.0)
5. If a date is mentioned, extract it precisely
6. Distinguish between what happened vs what the writer felt vs what they concluded
7. Extract minimum 2, maximum 8 memory objects per diary entry
8. For pattern candidates: only flag if the writer uses language like "I always...", "I tend to...", "every time...", "again I..."
9. For decisions: only extract if an actual choice is documented
10. Preserve the writer's voice — use their exact phrases in original_language fields
11. Support code-mixed/transliterated GujLish (Gujarati transliterated into English script). Interpret and translate GujLish words/phrases to English when populating 'title', 'content', 'lesson_content', and 'tension_description'. However, preserve the exact transliterated Gujarati words/phrases in 'original_language' and 'source_language' fields to capture the user's authentic voice.

OUTPUT FORMAT: Return ONLY a valid JSON array of memory objects. No commentary, no preamble, no markdown fences."""

EXTRACTION_USER_PROMPT = """Extract memory objects from this diary entry.

DATE: {date}
LIFE PHASE: {life_phase}
ENTRY TEXT:
---
{text}
---

Return a JSON array. Each memory object must follow this schema:
{{
  "memory_type": "episode|insight|pattern|belief|tension|decision",
  "title": "concise specific label (under 10 words)",
  "content": "full description in 2-4 sentences",
  "original_language": "verbatim quote from text that most captures this memory (under 50 words)",
  "domains": ["work", "relationships", "health", "creativity", "finances", "identity", "family", "friendship", "spirituality", "learning"],
  "emotion_primary": "joy|trust|fear|surprise|sadness|disgust|anger|anticipation|grief|burnout|shame|pride|loneliness|love|excitement|disillusionment|relief|regret|gratitude|unknown",
  "emotion_intensity": 0.0,
  "valence": 0.0,
  "arousal": 0.0,
  "emotion_secondary": [],
  "source_language": ["exact phrases that signal this emotion"],
  "somatic_markers": ["physical sensations mentioned: 'couldn't sleep', 'chest heaviness'"],
  "people_aliases": [{{ "alias": "role-based alias", "role": "role description" }}],
  "has_lesson": false,
  "lesson_content": null,
  "has_decision": false,
  "is_pattern_candidate": false,
  "pattern_trigger": null,
  "has_tension": false,
  "tension_description": null,
  "has_belief": false,
  "belief_statement": null,
  "salience_estimate": 0.5,
  "is_sensitive": false
}}

Guidance per memory_type:
- episode: use for discrete life events (a meeting, an ending, a realization moment)
- insight: use when writer explicitly draws a lesson or realization ("I realized...", "I've learned...")
- pattern: ONLY when writer uses pattern language ("I always...", "again...", "same as last time")
- belief: use for conviction statements ("I believe...", "I know that...", "people always...")
- tension: use for unresolved conflicts ("I still don't know...", "part of me wants X but...")
- decision: use when a specific choice is documented"""


# ─────────────────────────────────────────────
# MEMORY EXTRACTOR
# ─────────────────────────────────────────────

class MemoryExtractor:
    """
    Core extraction engine — transforms diary text into memory objects.
    Uses LLM with a structured prompt that enforces psychological accuracy.
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model

    async def extract_from_passage(
        self,
        text: str,
        source_doc_id: str,
        source_type: SourceType,
        date_str: Optional[str] = None,
        life_phase: Optional[str] = None,
        passage_ref: Optional[str] = None
    ) -> list[MemoryObject]:
        """
        Extract memory objects from a single semantic passage.
        Returns a list of MemoryObject instances, ready for storage.
        """
        if not text or len(text.strip().split()) < 15:
            logger.debug("Passage too short for extraction, skipping")
            return []

        raw_extractions = await self._llm_extract(text, date_str or "unknown", life_phase or "unknown")
        if not raw_extractions:
            return []

        objects = []
        for i, raw in enumerate(raw_extractions):
            try:
                obj = self._build_memory_object(
                    raw=raw,
                    source_doc_id=source_doc_id,
                    source_type=source_type,
                    date_str=date_str,
                    passage_ref=passage_ref or f"passage_{i+1}"
                )
                objects.append(obj)
            except Exception as e:
                logger.warning(f"Failed to build memory object {i}: {e}")
                continue

        logger.info(f"Extracted {len(objects)} memory objects from passage")
        return objects

    async def extract_from_document(
        self,
        passages: list[str],
        source_doc_id: str,
        source_type: SourceType,
        date_str: Optional[str] = None,
        life_phase: Optional[str] = None
    ) -> list[MemoryObject]:
        """Extract from all passages of a source document."""
        all_objects = []
        for i, passage in enumerate(passages):
            objects = await self.extract_from_passage(
                text=passage,
                source_doc_id=source_doc_id,
                source_type=source_type,
                date_str=date_str,
                life_phase=life_phase,
                passage_ref=f"passage_{i+1}_of_{len(passages)}"
            )
            all_objects.extend(objects)

        # Post-processing: cross-passage deduplication
        all_objects = self._deduplicate(all_objects)
        logger.info(f"Total extracted from document: {len(all_objects)} memory objects")
        return all_objects

    async def _llm_extract(
        self, text: str, date_str: str, life_phase: str
    ) -> list[dict]:
        """Call Gemini extraction and parse JSON response."""
        user_prompt = EXTRACTION_USER_PROMPT.format(
            date=date_str,
            life_phase=life_phase,
            text=text
        )

        try:
            raw_text = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                max_tokens=4096
            )

            # Clean and parse JSON
            raw_text = re.sub(r"```[a-z]*\n?", "", raw_text).strip()
            if not raw_text.startswith("["):
                # Try to extract array from response
                match = re.search(r"\[[\s\S]+\]", raw_text)
                if match:
                    raw_text = match.group(0)
                else:
                    logger.warning("LLM did not return a JSON array")
                    return []

            return json.loads(raw_text)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in memory extraction: {e}")
            return []
        except Exception as e:
            logger.error(f"Memory extraction API error: {e}")
            return []

    def _build_memory_object(
        self,
        raw: dict,
        source_doc_id: str,
        source_type: SourceType,
        date_str: Optional[str],
        passage_ref: str
    ) -> MemoryObject:
        """Build a MemoryObject from raw LLM extraction dict."""

        # Parse date
        date_info = DateInfo()
        if date_str:
            try:
                from datetime import date as date_type
                parsed_date = date_type.fromisoformat(date_str)
                date_info.occurred = parsed_date
                date_info.confidence = 0.9
            except ValueError:
                date_info.confidence = DateConfidence.LOW
                date_info.approximate_only = True

        # Build emotional profile
        primary_emotion = self._parse_emotion(raw.get("emotion_primary", "unknown"))
        emotion = EmotionalProfile(
            primary=primary_emotion,
            intensity=float(raw.get("emotion_intensity", 0.5)),
            valence=float(raw.get("valence", 0.0)),
            arousal=float(raw.get("arousal", 0.3)),
            secondary=[
                self._parse_emotion(e)
                for e in raw.get("emotion_secondary", [])
                if self._parse_emotion(e) != EmotionLabel.UNKNOWN
            ],
            source_language=raw.get("source_language", []),
            somatic_markers=raw.get("somatic_markers", [])
        )

        # Build person refs
        people = []
        for p in raw.get("people_aliases", []):
            if isinstance(p, dict):
                people.append(PersonRef(
                    alias=p.get("alias", "person_unknown"),
                    role=p.get("role", "unknown")
                ))

        # Parse domains
        domain_map = {d.value: d for d in LifeDomain}
        domains = [
            domain_map[d] for d in raw.get("domains", [])
            if d in domain_map
        ]

        # Build type-specific payload
        decision_payload = None
        pattern_payload = None
        belief_payload = None

        if raw.get("has_decision") and raw.get("memory_type") == "decision":
            decision_payload = {
                "decision_type": raw.get("decision_type", "unknown"),
                "stated_reasoning": raw.get("lesson_content"),
            }

        if raw.get("is_pattern_candidate"):
            pattern_payload = {
                "trigger_description": raw.get("pattern_trigger", ""),
                "instance_count": 1,
                "status": PatternStatus.EMERGING
            }

        if raw.get("has_belief"):
            belief_payload = {
                "belief_statement": raw.get("belief_statement", ""),
                "is_explicit": True,
                "confidence_in_belief": 0.7
            }

        # Provenance
        provenance = SourceProvenance(
            source_doc_id=source_doc_id,
            source_type=source_type,
            page_reference=passage_ref
        )

        # Compute initial salience
        salience = float(raw.get("salience_estimate", 0.5))
        # Boost high-emotion memories
        if emotion.intensity > 0.7:
            salience = min(1.0, salience + 0.15)
        # Boost memories with explicit lessons
        if raw.get("has_lesson"):
            salience = min(1.0, salience + 0.1)

        memory_type_str = raw.get("memory_type", "episode")
        try:
            mtype = MemoryType(memory_type_str)
        except ValueError:
            mtype = MemoryType.EPISODE

        return MemoryObject(
            memory_type=mtype,
            title=raw.get("title", "Untitled memory"),
            content=raw.get("content", ""),
            original_language=raw.get("original_language"),
            date_info=date_info,
            emotion=emotion,
            domains=domains,
            people_involved=people,
            lesson_content=raw.get("lesson_content") if raw.get("has_lesson") else None,
            decision_payload=decision_payload,
            pattern_payload=pattern_payload,
            belief_payload=belief_payload,
            provenance=provenance,
            salience_score=salience,
            is_sensitive=bool(raw.get("is_sensitive", False))
        )

    def _parse_emotion(self, emotion_str: str) -> EmotionLabel:
        try:
            return EmotionLabel(emotion_str.lower())
        except ValueError:
            return EmotionLabel.UNKNOWN

    def _deduplicate(self, objects: list[MemoryObject]) -> list[MemoryObject]:
        """Remove near-duplicate memory objects from the same document."""
        if len(objects) <= 1:
            return objects

        seen_titles = set()
        unique = []
        for obj in objects:
            # Simple title-based dedup — embeddings-based dedup happens at ingestion
            normalized_title = obj.title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique.append(obj)

        if len(unique) < len(objects):
            logger.debug(f"Deduplication removed {len(objects) - len(unique)} duplicates")
        return unique


# ─────────────────────────────────────────────
# METADATA ENRICHER
# ─────────────────────────────────────────────

LIFE_PHASE_INFERENCE_PROMPT = """Given this diary entry text and date, infer the user's life phase.

Common life phases: university, early_career, startup_phase_1, startup_phase_2, 
burnout_recovery, post_exit, relationship_start, relationship_end, 
grief_period, creative_phase, transition_period, peak_performance, 
family_phase, spiritual_seeking

Date: {date}
Text excerpt (first 300 chars): {excerpt}

Return JSON: {{"life_phase": "phase_name", "confidence": 0.0-1.0, "reasoning": "brief reason"}}
Return only JSON, no other text."""


class MetadataEnricher:
    """
    Enriches memory objects with additional metadata:
    - Life phase inference
    - Domain classification refinement
    - Salience score computation
    - Pattern tag suggestions from known taxonomy
    """

    PERSONAL_TAXONOMY = [
        "burnout-cycle", "overcommitment-spiral", "fear-then-leap",
        "identity-through-achievement", "avoidant-under-criticism",
        "delayed-exit", "overdeference-in-partnership", "control-seeking-under-uncertainty",
        "isolation-coping", "external-validation-seeking", "perfectionism-paralysis",
        "values-misalignment-exit", "care-at-expense-of-self",
        "emotional-withdrawal-under-conflict", "rush-commit-regret-cycle"
    ]

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model  # Use faster/cheaper model for enrichment

    async def infer_life_phase(self, text: str, date_str: str) -> tuple[str, float]:
        """Infer which life phase this entry belongs to."""
        try:
            raw = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": LIFE_PHASE_INFERENCE_PROMPT.format(
                        date=date_str,
                        excerpt=text[:300]
                    )
                }]
            )
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(raw)
            return data.get("life_phase", "unknown"), data.get("confidence", 0.5)
        except Exception as e:
            logger.debug(f"Life phase inference failed: {e}")
            return "unknown", 0.3

    def suggest_pattern_tags(self, memory: MemoryObject) -> list[str]:
        """
        Suggest personal taxonomy tags for a memory object
        based on content, emotion, and behavioral signals.
        """
        suggestions = []
        content_lower = (memory.content + " " + (memory.original_language or "")).lower()

        # Rule-based tag suggestions from taxonomy
        tag_signals = {
            "burnout-cycle": ["burned out", "exhausted", "drained", "empty", "can't continue"],
            "overcommitment-spiral": ["said yes to", "took on too much", "overwhelmed", "spread thin"],
            "fear-then-leap": ["terrified but", "jumped anyway", "despite the fear"],
            "identity-through-achievement": ["worth comes from", "only matter if", "prove myself"],
            "avoidant-under-criticism": ["shut down", "withdrew", "went quiet when"],
            "delayed-exit": ["should have left earlier", "stayed too long", "knew it was wrong but"],
            "control-seeking-under-uncertainty": ["need to control", "made plans obsessively", "couldn't sit with not knowing"],
            "isolation-coping": ["closed off", "didn't tell anyone", "dealt with it alone"],
            "values-misalignment-exit": ["didn't believe in", "mission changed", "couldn't align"],
        }

        for tag, signals in tag_signals.items():
            if any(signal in content_lower for signal in signals):
                suggestions.append(tag)

        return suggestions[:3]  # Max 3 auto-suggested tags

    def compute_salience(self, memory: MemoryObject, days_since_created: int = 0) -> float:
        """Compute and set salience score on memory object."""
        score = memory.compute_salience(
            days_since_created=days_since_created,
            user_importance_flag=False
        )
        return score
