"""
PHASE 2 — INTELLIGENCE LAYER
Cognitive Memory OS

Three core intelligence components:
1. Emotional Pattern Analyzer — finds emotional loops across time
2. Life Pattern Detector — detects behavioral patterns from memory clusters
3. Contradiction Engine — finds inconsistencies between beliefs and actions

These transform raw memory objects into psychological intelligence.
The difference between a diary search tool and a cognitive mirror.
"""

from __future__ import annotations
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import httpx

from models import (
    MemoryObject, MemoryType, EmotionLabel, LifeDomain,
    LifePattern, PatternPayload, PatternStatus,
    Contradiction
)
from windows_compat import call_gemini_openai

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EMOTIONAL PATTERN ANALYZER
# ─────────────────────────────────────────────

class EmotionalPatternAnalyzer:
    """
    Analyzes emotional patterns across the full memory store.

    Key capabilities:
    - Emotional trigger detection (what consistently precedes high-emotion states)
    - Domain-specific emotional defaults (how you feel in different life contexts)
    - Emotional arc detection (how emotions evolve during life events)
    - Seasonal/cyclical emotional patterns
    """

    def analyze_emotional_triggers(
        self, memories: list[MemoryObject]
    ) -> list[EmotionalTrigger]:
        """
        Identify what triggers strong emotional states.
        A trigger = a context/situation that reliably precedes high-intensity emotion.
        """
        # Group memories by high-intensity emotion
        high_intensity = [
            m for m in memories
            if m.emotion.intensity > 0.65
        ]

        if len(high_intensity) < 3:
            return []

        # Find common contextual patterns before high-emotion events
        triggers = self._extract_trigger_patterns(high_intensity, memories)
        return triggers

    def _extract_trigger_patterns(
        self,
        high_emotion_memories: list[MemoryObject],
        all_memories: list[MemoryObject]
    ) -> list[EmotionalTrigger]:
        """Extract what precedes high-emotion events."""
        # Group by primary emotion
        by_emotion: dict[str, list[MemoryObject]] = defaultdict(list)
        for m in high_emotion_memories:
            by_emotion[str(m.emotion.primary)].append(m)

        triggers = []
        for emotion_label, cluster in by_emotion.items():
            if len(cluster) < 2:
                continue

            # Find common pattern_tags across cluster
            tag_counts: dict[str, int] = defaultdict(int)
            domain_counts: dict[str, int] = defaultdict(int)
            person_counts: dict[str, int] = defaultdict(int)

            for m in cluster:
                for tag in m.pattern_tags:
                    tag_counts[tag] += 1
                for domain in m.domains:
                    domain_str = domain if isinstance(domain, str) else domain.value
                    domain_counts[domain_str] += 1
                for person in m.people_involved:
                    person_counts[person.role] += 1

            # Tags appearing in >= 50% of cluster = trigger pattern
            threshold = max(2, len(cluster) * 0.4)
            trigger_tags = [tag for tag, count in tag_counts.items() if count >= threshold]
            trigger_domains = [d for d, count in domain_counts.items() if count >= threshold]

            if trigger_tags or len(cluster) >= 3:
                triggers.append(EmotionalTrigger(
                    emotion=emotion_label,
                    avg_intensity=sum(m.emotion.intensity for m in cluster) / len(cluster),
                    instance_count=len(cluster),
                    context_tags=trigger_tags,
                    context_domains=trigger_domains,
                    description=self._describe_trigger(emotion_label, trigger_tags, trigger_domains, cluster),
                    memory_ids=[m.id for m in cluster]
                ))

        return sorted(triggers, key=lambda t: t.avg_intensity * t.instance_count, reverse=True)

    def _describe_trigger(
        self,
        emotion: str,
        tags: list[str],
        domains: list[str],
        cluster: list[MemoryObject]
    ) -> str:
        """Generate a natural language trigger description."""
        domain_str = f" in {', '.join(domains[:2])}" if domains else ""
        tag_str = f" involving {', '.join(tags[:2]).replace('-', ' ')}" if tags else ""
        return f"You consistently experience {emotion}{domain_str}{tag_str} ({len(cluster)} documented instances)"

    def compute_domain_emotional_defaults(
        self, memories: list[MemoryObject]
    ) -> dict[str, list[str]]:
        """
        Compute your emotional default states per life domain.
        Result: {'work': ['anxious-achievement', 'driven'], 'relationships': ['cautious-warmth']}
        """
        domain_emotions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        domain_counts: dict[str, int] = defaultdict(int)

        for m in memories:
            for domain in m.domains:
                domain_str = domain if isinstance(domain, str) else domain.value
                emotion_str = str(m.emotion.primary)
                domain_emotions[domain_str][emotion_str] += m.emotion.intensity
                domain_counts[domain_str] += 1

        defaults = {}
        for domain, emotions in domain_emotions.items():
            if domain_counts[domain] < 3:
                continue
            # Normalize by count and sort by weighted frequency
            normalized = {
                e: score / domain_counts[domain]
                for e, score in emotions.items()
            }
            top_emotions = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
            defaults[domain] = [e for e, _ in top_emotions if _ > 0.1]

        return defaults


@dataclass
class EmotionalTrigger:
    emotion: str
    avg_intensity: float
    instance_count: int
    context_tags: list[str]
    context_domains: list[str]
    description: str
    memory_ids: list[str]


# ─────────────────────────────────────────────
# LIFE PATTERN DETECTOR
# ─────────────────────────────────────────────

PATTERN_DETECTION_SYSTEM_PROMPT = """You are a behavioral pattern analyst studying a person's life through their personal memories.

You are given a cluster of memories that have been grouped together by semantic similarity. Your task is to identify if they form a genuine recurring behavioral pattern.

A genuine pattern has:
1. A consistent TRIGGER (what initiates it)
2. A consistent BEHAVIOR (what the person does)
3. A consistent OUTCOME (what results)
4. A consistent INTERPRETATION (what the person tells themselves about it)

Rules:
- Speak in second person ("You tend to...")
- Use the person's own language where possible
- Be specific — name the actual pattern, don't generalize
- Only declare a pattern if it appears across at least 3 distinct life events in different time periods
- State your confidence (high/medium/low)
- Note if this pattern appears adaptive or maladaptive
- If the cluster does NOT form a genuine pattern, say so

Output ONLY JSON, no commentary."""

PATTERN_DETECTION_USER_PROMPT = """Analyze these {count} memories from different time periods to detect a behavioral pattern.

MEMORIES:
{memories_text}

Output JSON:
{{
  "is_genuine_pattern": true/false,
  "confidence": "high|medium|low",
  "pattern_name": "short hyphenated name",
  "description": "2-3 sentence description in second person",
  "loop": {{
    "trigger": "what initiates this",
    "emotion": "what you feel",
    "behavior": "what you do",
    "outcome": "what results",
    "interpretation": "what you tell yourself"
  }},
  "is_adaptive": true/false/null,
  "domain_spread": ["work", "relationships", ...],
  "earliest_evidence": "approximate period",
  "significance": 0.0-1.0,
  "reason_for_rejection": null
}}"""


class LifePatternDetector:
    """
    Detects behavioral patterns from clusters of semantically similar memories.

    Pipeline:
    1. Receive pre-clustered memory groups
    2. For each cluster with >= 3 members spanning >= 2 time periods:
       a. Run LLM pattern analysis
       b. If genuine pattern confirmed, create LifePattern object
       c. Link supporting memories
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model

    async def detect_from_cluster(
        self, cluster: list[MemoryObject]
    ) -> Optional[LifePattern]:
        """
        Analyze a cluster of similar memories for a behavioral pattern.
        Returns LifePattern if genuine pattern found, None otherwise.
        """
        if len(cluster) < 3:
            return None

        # Check temporal spread — must span at least 2 different life phases or 180 days
        if not self._has_temporal_spread(cluster):
            logger.debug("Cluster lacks temporal spread — skipping pattern detection")
            return None

        memories_text = self._format_memories_for_analysis(cluster)
        raw_result = await self._llm_detect(memories_text, len(cluster))

        if not raw_result:
            return None

        if not raw_result.get("is_genuine_pattern", False):
            logger.debug(f"Cluster rejected: {raw_result.get('reason_for_rejection', 'no reason')}")
            return None

        return self._build_pattern(raw_result, cluster)

    async def detect_from_all_memories(
        self, memories: list[MemoryObject], embedding_store
    ) -> list[LifePattern]:
        """
        Full pattern detection run across all memories.
        Uses embedding-based clustering to find candidate pattern groups.
        """
        # Cluster memories by semantic similarity
        clusters = await self._cluster_memories(memories, embedding_store)
        logger.info(f"Found {len(clusters)} candidate clusters for pattern detection")

        patterns = []
        for i, cluster in enumerate(clusters):
            logger.debug(f"Analyzing cluster {i+1}/{len(clusters)} ({len(cluster)} memories)")
            pattern = await self.detect_from_cluster(cluster)
            if pattern:
                patterns.append(pattern)
                logger.info(f"Pattern detected: {pattern.name}")

        return patterns

    async def _cluster_memories(
        self, memories: list[MemoryObject], embedding_store
    ) -> list[list[MemoryObject]]:
        """
        Use HDBSCAN to cluster memories by behavioral embedding similarity.
        """
        try:
            import numpy as np
            from hdbscan import HDBSCAN
        except ImportError:
            logger.warning("hdbscan not installed — using simple keyword clustering fallback")
            return self._keyword_cluster_fallback(memories)

        # Get behavioral pattern embeddings from Qdrant
        # In production, fetch from Qdrant; here we demonstrate the approach
        vectors = []
        valid_memories = []
        for m in memories:
            if m.embedding_ids.get("behavioral_pattern"):
                # Would fetch from Qdrant in production
                valid_memories.append(m)

        if len(valid_memories) < 6:
            return self._keyword_cluster_fallback(memories)

        # Simplified: cluster by pattern tags + domain overlap
        return self._tag_cluster(valid_memories)

    def _tag_cluster(self, memories: list[MemoryObject]) -> list[list[MemoryObject]]:
        """Cluster by shared pattern tags."""
        tag_groups: dict[str, list[MemoryObject]] = defaultdict(list)
        unclustered = []

        for m in memories:
            if m.pattern_tags:
                for tag in m.pattern_tags[:2]:  # Use top 2 tags
                    tag_groups[tag].append(m)
            else:
                unclustered.append(m)

        clusters = []
        for tag, group in tag_groups.items():
            if len(group) >= 3:
                clusters.append(group)

        # Also cluster by domain
        domain_groups: dict[str, list[MemoryObject]] = defaultdict(list)
        for m in unclustered:
            for domain in m.domains[:1]:
                domain_str = domain if isinstance(domain, str) else domain.value
                domain_groups[domain_str].append(m)

        for domain, group in domain_groups.items():
            if len(group) >= 4:
                clusters.append(group[:8])  # Cap at 8 memories per cluster for LLM

        return clusters

    def _keyword_cluster_fallback(
        self, memories: list[MemoryObject]
    ) -> list[list[MemoryObject]]:
        """Simple keyword-based clustering when embeddings not available."""
        clusters = defaultdict(list)
        for m in memories:
            if m.pattern_tags:
                key = m.pattern_tags[0]
                clusters[key].append(m)
            else:
                for domain in m.domains[:1]:
                    domain_str = domain if isinstance(domain, str) else domain.value
                    clusters[f"domain_{domain_str}"].append(m)
        return [c for c in clusters.values() if len(c) >= 3]

    def _has_temporal_spread(self, cluster: list[MemoryObject]) -> bool:
        """Check if cluster spans at least 180 days or different life phases."""
        dates = [
            m.date_info.occurred for m in cluster
            if m.date_info.occurred is not None
        ]
        if len(dates) >= 2:
            span = (max(dates) - min(dates)).days
            if span >= 180:
                return True

        phases = {
            m.date_info.life_phase for m in cluster
            if m.date_info.life_phase
        }
        return len(phases) >= 2

    def _format_memories_for_analysis(self, cluster: list[MemoryObject]) -> str:
        """Format memory cluster for LLM analysis."""
        parts = []
        for i, m in enumerate(cluster[:8], 1):  # Cap at 8 memories
            date_str = str(m.date_info.occurred) if m.date_info.occurred else "unknown date"
            phase_str = m.date_info.life_phase or ""
            parts.append(
                f"Memory {i} [{date_str}, {phase_str}]:\n"
                f"  Title: {m.title}\n"
                f"  Content: {m.content}\n"
                f"  Emotion: {m.emotion.primary} (intensity: {m.emotion.intensity:.1f})\n"
                f"  Original language: \"{m.original_language or 'N/A'}\"\n"
                f"  Lesson: {m.lesson_content or 'None stated'}"
            )
        return "\n\n".join(parts)

    async def _llm_detect(self, memories_text: str, count: int) -> Optional[dict]:
        """Call Gemini for pattern detection."""
        prompt = PATTERN_DETECTION_USER_PROMPT.format(
            count=count,
            memories_text=memories_text
        )
        try:
            raw = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                system_prompt=PATTERN_DETECTION_SYSTEM_PROMPT,
                max_tokens=1500
            )
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Pattern detection LLM error: {e}")
            return None

    def _build_pattern(self, raw: dict, cluster: list[MemoryObject]) -> LifePattern:
        """Build LifePattern from LLM result."""
        loop = raw.get("loop", {})
        confidence_map = {"high": 0.85, "medium": 0.6, "low": 0.4}
        significance = raw.get("significance", 0.6)

        dates = sorted([
            m.date_info.occurred for m in cluster
            if m.date_info.occurred
        ])

        payload = PatternPayload(
            trigger_description=loop.get("trigger", ""),
            behavior_description=loop.get("behavior", ""),
            outcome_description=loop.get("outcome", ""),
            interpretation_description=loop.get("interpretation", ""),
            instance_count=len(cluster),
            first_occurrence=raw.get("earliest_evidence", str(dates[0]) if dates else None),
            most_recent=str(dates[-1]) if dates else None,
            status=PatternStatus.ACTIVE,
            significance_score=significance * confidence_map.get(raw.get("confidence", "medium"), 0.6),
            is_adaptive=raw.get("is_adaptive"),
        )

        domains = [LifeDomain(d) for d in raw.get("domain_spread", []) if d in [e.value for e in LifeDomain]]

        return LifePattern(
            name=raw.get("pattern_name", "unnamed-pattern"),
            description=raw.get("description", ""),
            loop_description=json.dumps(loop),
            payload=payload,
            supporting_memory_ids=[m.id for m in cluster],
            significance_score=significance,
            domains=domains,
            tags=[raw.get("pattern_name", "")]
        )


# ─────────────────────────────────────────────
# CONTRADICTION ENGINE
# ─────────────────────────────────────────────

class ContradictionEngine:
    """
    Detects contradictions between memories:
    - Value-action contradictions (you say X matters but do Y)
    - Lesson-repetition contradictions (learned X but did X again)
    - Temporal belief contradictions (believed opposite things at different times)
    - Self-concept contradictions (self-description vs behavioral evidence)

    The most psychologically valuable and most computationally expensive component.
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model

    def score_contradiction(
        self,
        memory_a: MemoryObject,
        memory_b: MemoryObject
    ) -> float:
        """
        Score the contradiction strength between two memories.

        Formula:
        score = semantic_opposition × temporal_proximity_weight
                × domain_overlap × explicitness_factor
        """
        # Semantic opposition: beliefs/values with opposite valence
        valence_diff = abs(memory_a.emotion.valence - memory_b.emotion.valence)
        semantic_opposition = min(1.0, valence_diff)

        # Temporal proximity weight: recent contradictions score higher
        days_apart = 0
        if memory_a.date_info.occurred and memory_b.date_info.occurred:
            days_apart = abs((memory_a.date_info.occurred - memory_b.date_info.occurred).days)
        temporal_weight = max(0.2, 1.0 - (days_apart / 1825))  # 5-year decay

        # Domain overlap: same domain = stronger contradiction
        domains_a = {d if isinstance(d, str) else d.value for d in memory_a.domains}
        domains_b = {d if isinstance(d, str) else d.value for d in memory_b.domains}
        domain_overlap = len(domains_a & domains_b) / max(len(domains_a | domains_b), 1)

        # Explicitness: beliefs and insights score higher than episodes
        explicit_types = {MemoryType.BELIEF, MemoryType.INSIGHT, MemoryType.VALUE}
        explicitness = 1.0 if (
            memory_a.memory_type in explicit_types or
            memory_b.memory_type in explicit_types
        ) else 0.6

        score = (
            semantic_opposition * 0.4
            + temporal_weight * 0.3
            + domain_overlap * 0.2
            + explicitness * 0.1
        )
        return round(score, 4)

    def find_lesson_repetition_contradictions(
        self, memories: list[MemoryObject]
    ) -> list[Contradiction]:
        """
        Find cases where a lesson was stated but the same behavior occurred again.
        The most common and psychologically significant contradiction type.
        """
        contradictions = []

        # Find insight memories with explicit lessons
        lessons = [m for m in memories if m.lesson_content and m.memory_type == MemoryType.INSIGHT]
        episodes = [m for m in memories if m.memory_type == MemoryType.EPISODE]

        for lesson in lessons:
            lesson_date = lesson.date_info.occurred
            if not lesson_date:
                continue

            for episode in episodes:
                episode_date = episode.date_info.occurred
                if not episode_date:
                    continue

                # Episode must be AFTER the lesson
                if episode_date <= lesson_date:
                    continue

                # Check for domain overlap
                lesson_domains = {d if isinstance(d, str) else d.value for d in lesson.domains}
                episode_domains = {d if isinstance(d, str) else d.value for d in episode.domains}

                if not (lesson_domains & episode_domains):
                    continue

                # Check for pattern tag overlap
                shared_tags = set(lesson.pattern_tags) & set(episode.pattern_tags)
                if not shared_tags and not (lesson_domains & episode_domains):
                    continue

                score = self.score_contradiction(lesson, episode)
                if score > 0.55:
                    days_gap = (episode_date - lesson_date).days
                    contradictions.append(Contradiction(
                        type="lesson_repetition",
                        memory_a_id=lesson.id,
                        memory_b_id=episode.id,
                        description=(
                            f"You documented the lesson '{lesson.lesson_content[:100]}...' "
                            f"in {lesson_date.year}, but {days_gap} days later a similar "
                            f"pattern appears in your {', '.join(list(episode_domains)[:2])} entries."
                        ),
                        contradiction_score=score,
                        temporal_gap_days=days_gap,
                        same_domain=bool(lesson_domains & episode_domains)
                    ))

        return sorted(contradictions, key=lambda c: c.contradiction_score, reverse=True)

    def find_value_action_contradictions(
        self,
        memories: list[MemoryObject],
        espoused_values: dict[str, float]
    ) -> list[Contradiction]:
        """
        Find where stated values diverge from behavioral evidence.
        """
        contradictions = []

        value_memories = [m for m in memories if m.memory_type == MemoryType.VALUE]
        behavior_memories = [m for m in memories if m.memory_type in (MemoryType.EPISODE, MemoryType.DECISION)]

        for value_mem in value_memories:
            if not value_mem.belief_payload:
                continue
            belief = value_mem.belief_payload
            if isinstance(belief, dict):
                belief_statement = belief.get("belief_statement", "")
            else:
                belief_statement = str(belief)

            # Look for behavioral memories that contradict this value
            for behavior_mem in behavior_memories:
                if not behavior_mem.date_info.occurred:
                    continue
                if not value_mem.date_info.occurred:
                    continue
                if behavior_mem.date_info.occurred < value_mem.date_info.occurred:
                    continue

                score = self.score_contradiction(value_mem, behavior_mem)
                if score > 0.6:
                    contradictions.append(Contradiction(
                        type="value_action",
                        memory_a_id=value_mem.id,
                        memory_b_id=behavior_mem.id,
                        description=(
                            f"You stated the value: '{belief_statement[:80]}' "
                            f"but subsequent behavior in {', '.join(str(d) for d in behavior_mem.domains[:2])} "
                            f"may contradict this."
                        ),
                        contradiction_score=score,
                        temporal_gap_days=(
                            (behavior_mem.date_info.occurred - value_mem.date_info.occurred).days
                            if (behavior_mem.date_info.occurred and value_mem.date_info.occurred)
                            else None
                        ),
                        same_domain=bool(
                            {d if isinstance(d, str) else d.value for d in value_mem.domains}
                            & {d if isinstance(d, str) else d.value for d in behavior_mem.domains}
                        )
                    ))

        return sorted(contradictions, key=lambda c: c.contradiction_score, reverse=True)


# ─────────────────────────────────────────────
# TEMPORAL PATTERN ANALYZER
# ─────────────────────────────────────────────

class TemporalPatternAnalyzer:
    """
    Detects time-based patterns:
    - Seasonal patterns (Q4 overcommitment, January depression)
    - Cyclical patterns (burnout every 18 months)
    - Anniversary effects
    - Growth vs regression across life phases
    """

    def detect_seasonal_patterns(
        self, memories: list[MemoryObject]
    ) -> list[SeasonalPattern]:
        """Find patterns that correlate with time of year."""
        from collections import defaultdict

        quarterly_emotions: dict[str, dict[str, list[float]]] = {
            "Q1": defaultdict(list),
            "Q2": defaultdict(list),
            "Q3": defaultdict(list),
            "Q4": defaultdict(list),
        }

        for m in memories:
            if not m.date_info.occurred:
                continue
            month = m.date_info.occurred.month
            quarter = f"Q{(month - 1) // 3 + 1}"
            emotion_str = str(m.emotion.primary)
            quarterly_emotions[quarter][emotion_str].append(m.emotion.intensity)

        patterns = []
        for quarter, emotions in quarterly_emotions.items():
            for emotion, intensities in emotions.items():
                if len(intensities) >= 3:
                    avg_intensity = sum(intensities) / len(intensities)
                    if avg_intensity > 0.55:
                        patterns.append(SeasonalPattern(
                            quarter=quarter,
                            emotion=emotion,
                            avg_intensity=avg_intensity,
                            occurrence_count=len(intensities),
                            description=f"In {quarter} you tend to experience heightened {emotion} (avg intensity {avg_intensity:.2f})"
                        ))

        return sorted(patterns, key=lambda p: p.avg_intensity, reverse=True)

    def detect_anniversary_memories(
        self, memories: list[MemoryObject], reference_date: date
    ) -> list[MemoryObject]:
        """Find memories that fall close to the same date as reference_date in past years."""
        same_period = []
        ref_month_day = (reference_date.month, reference_date.day)

        for m in memories:
            if not m.date_info.occurred:
                continue
            if m.date_info.occurred.year == reference_date.year:
                continue
            m_month_day = (m.date_info.occurred.month, m.date_info.occurred.day)
            day_diff = abs(
                (m.date_info.occurred.replace(year=reference_date.year) - reference_date).days
            )
            if day_diff <= 14:  # Within 2 weeks
                same_period.append(m)

        return sorted(same_period, key=lambda m: m.salience_score, reverse=True)


@dataclass
class SeasonalPattern:
    quarter: str
    emotion: str
    avg_intensity: float
    occurrence_count: int
    description: str
