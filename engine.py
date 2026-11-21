"""
PHASE 5 — WISDOM LAYER
Cognitive Memory OS

The Wisdom Layer is the highest-order processing in the system.
It doesn't retrieve memories — it synthesizes them into life intelligence.

Components:
1. LifeNarrativeEngine     — reconstructs coherent story arcs from memory data
2. StrategicInsightEngine  — generates non-obvious insights from pattern synthesis
3. PredictiveBehaviorEngine — projects behavioral tendencies into future scenarios
4. LongTermReflectionEngine — scheduled deep reflections across years of data
5. WisdomCrystallizer      — distills life data into earned personal philosophy

Design principle:
Wisdom is not retrieved — it is synthesized.
Individual memories are data. Patterns are intelligence. Wisdom is knowing
which of your patterns to trust, which to question, and which to transcend.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import httpx

from models import (
    MemoryObject, MemoryType, LifeDomain,
    LifePattern, Contradiction, PsychologicalModel
)
from windows_compat import call_gemini_openai

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIFE NARRATIVE ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NARRATIVE_RECONSTRUCTION_PROMPT = """You are reconstructing a coherent life narrative from documented personal memories.

Your task: weave these memories into a narrative arc that reveals the deeper meaning,
not just the sequence of events.

Scope: {scope}
Theme/Focus: {theme}

MEMORIES (chronological):
{memories}

DETECTED PATTERNS active during this period:
{patterns}

Generate a narrative that:
1. Flows chronologically but explains causally — why did one thing lead to another?
2. Names the psychological arc — what was growing, changing, or being tested?
3. Identifies turning points — the moments that changed the direction
4. Uses the person's own language (from original_language fields) where possible
5. Is honest — names what went wrong AND what grew
6. Ends with what this period MEANT for who they became

Rules:
- Every specific claim must come from a specific memory in the context
- If a gap exists in the data, acknowledge it explicitly: "[Limited data for this period]"
- Write in second person ("You were...")
- Keep to 400-600 words
- NEVER fabricate events or emotions not in the source memories

Format:
{{
  "title": "evocative title for this narrative arc",
  "period": "time span covered",
  "arc_type": "growth|struggle|transformation|stagnation|discovery|recovery",
  "narrative": "full narrative text",
  "turning_points": ["moment 1", "moment 2"],
  "psychological_theme": "the deeper psychological journey",
  "what_it_produced": "who you became / what capacity developed / what wound formed",
  "confidence": 0.0-1.0,
  "data_gaps": ["any gaps in the data noted"]
}}"""


class LifeNarrativeEngine:
    """
    Reconstructs coherent narrative arcs from memory data.

    Use cases:
    - "Tell me the story of my first company"
    - "Reconstruct the arc of my relationship with X"
    - "What was happening during my burnout period?"
    - "Trace my relationship with work from 2016 to 2022"
    """

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key
        self.model = model

    async def reconstruct(
        self,
        memories: list[MemoryObject],
        patterns: list[LifePattern],
        scope: str,
        theme: Optional[str] = None,
    ) -> NarrativeArc:
        """
        Reconstruct a narrative arc from provided memories.
        Memories should already be filtered and sorted by the caller.
        """
        if not memories:
            return NarrativeArc(
                title="Insufficient data",
                period="unknown",
                arc_type="unknown",
                narrative="Not enough documented memories found for this period or theme.",
                turning_points=[],
                psychological_theme="",
                what_it_produced="",
                confidence=0.0,
                data_gaps=["No memories available"],
                memory_ids=[],
                pattern_ids=[]
            )

        # Sort chronologically
        sorted_memories = sorted(
            memories,
            key=lambda m: m.date_info.occurred or date.min
        )

        memories_text = self._format_memories_for_narrative(sorted_memories)
        patterns_text = self._format_patterns(patterns)

        raw = await self._llm_reconstruct(
            scope=scope,
            theme=theme or "general life arc",
            memories_text=memories_text,
            patterns_text=patterns_text
        )

        if not raw:
            return NarrativeArc(
                title="Reconstruction failed",
                period=scope,
                arc_type="unknown",
                narrative="Unable to reconstruct narrative at this time.",
                turning_points=[],
                psychological_theme="",
                what_it_produced="",
                confidence=0.0,
                data_gaps=["LLM error"],
                memory_ids=[m.id for m in sorted_memories],
                pattern_ids=[p.id for p in patterns]
            )

        return NarrativeArc(
            title=raw.get("title", "Untitled Arc"),
            period=raw.get("period", scope),
            arc_type=raw.get("arc_type", "unknown"),
            narrative=raw.get("narrative", ""),
            turning_points=raw.get("turning_points", []),
            psychological_theme=raw.get("psychological_theme", ""),
            what_it_produced=raw.get("what_it_produced", ""),
            confidence=raw.get("confidence", 0.5),
            data_gaps=raw.get("data_gaps", []),
            memory_ids=[m.id for m in sorted_memories],
            pattern_ids=[p.id for p in patterns]
        )

    def _format_memories_for_narrative(
        self, memories: list[MemoryObject]
    ) -> str:
        blocks = []
        for m in memories[:12]:  # Cap at 12 for context budget
            date_str = str(m.date_info.occurred) if m.date_info.occurred else "unknown"
            phase = m.date_info.life_phase or ""
            emotion = f"{m.emotion.primary} (intensity {m.emotion.intensity:.1f})"
            block = (
                f"[{date_str} | {phase}]\n"
                f"Type: {m.memory_type} | Emotion: {emotion}\n"
                f"Title: {m.title}\n"
                f"Content: {m.content}"
            )
            if m.original_language:
                block += f'\nIn their words: "{m.original_language}"'
            if m.lesson_content:
                block += f"\nLesson: {m.lesson_content}"
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    def _format_patterns(self, patterns: list[LifePattern]) -> str:
        if not patterns:
            return "No patterns detected for this period."
        return "\n".join([
            f"- {p.name}: {p.description[:100]}"
            for p in patterns[:4]
        ])

    async def _llm_reconstruct(
        self,
        scope: str,
        theme: str,
        memories_text: str,
        patterns_text: str
    ) -> Optional[dict]:
        prompt = NARRATIVE_RECONSTRUCTION_PROMPT.format(
            scope=scope,
            theme=theme,
            memories=memories_text,
            patterns=patterns_text
        )

        try:
            raw = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Narrative reconstruction error: {e}")
            return None


@dataclass
class NarrativeArc:
    title: str
    period: str
    arc_type: str
    narrative: str
    turning_points: list[str]
    psychological_theme: str
    what_it_produced: str
    confidence: float
    data_gaps: list[str]
    memory_ids: list[str]
    pattern_ids: list[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGIC INSIGHT ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIC_INSIGHT_PROMPT = """You are a life strategist who has studied one person's complete documented history.

Your task: generate a NON-OBVIOUS strategic insight from their patterns and history.

A strategic insight is NOT:
- Generic advice ("try to set better boundaries")
- A restatement of an obvious pattern
- What they already know consciously

A strategic insight IS:
- A connection across domains they haven't made themselves
- An asymmetry in how they apply wisdom (wise in one domain, blind in another)
- A second-order consequence of a pattern they're aware of
- A hidden strength they're underusing
- A recurring cost they haven't quantified
- A belief that's protecting them from growth in a specific area

THEIR PATTERNS:
{patterns}

THEIR CONTRADICTIONS:
{contradictions}

THEIR VALUE GAPS:
{value_gaps}

THEIR PSYCHOLOGICAL MODEL:
{psych_model}

Generate 3 strategic insights. For each:
{{
  "insight": "the core insight in 2 sentences",
  "evidence": "what from their data leads to this",
  "why_non_obvious": "why they might not see this themselves",
  "implication": "what this means for their decisions going forward",
  "domain": "which life domain this most applies to",
  "confidence": 0.0-1.0
}}

Return as JSON array. Be specific to THIS person. No generic advice."""


class StrategicInsightEngine:
    """
    Generates high-value, non-obvious insights by synthesizing
    patterns, contradictions, value gaps, and psychological model.

    This goes beyond what pattern detection finds — it finds the
    connections BETWEEN patterns, the hidden costs, the asymmetric blindspots.
    """

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key
        self.model = model

    async def generate_insights(
        self,
        patterns: list[LifePattern],
        contradictions: list[Contradiction],
        value_gaps: list,           # list[ValueGap]
        psych_model: PsychologicalModel,
        memories: list[MemoryObject],
    ) -> list[StrategicInsight]:
        """Generate the top strategic insights from the full synthesis."""

        patterns_text = self._format_patterns(patterns)
        contradictions_text = self._format_contradictions(contradictions)
        value_gaps_text = self._format_value_gaps(value_gaps)
        psych_text = self._format_psych_model(psych_model)

        raw_insights = await self._llm_generate(
            patterns_text, contradictions_text, value_gaps_text, psych_text
        )

        if not raw_insights:
            return []

        return [
            StrategicInsight(
                insight=r.get("insight", ""),
                evidence=r.get("evidence", ""),
                why_non_obvious=r.get("why_non_obvious", ""),
                implication=r.get("implication", ""),
                domain=r.get("domain", "general"),
                confidence=r.get("confidence", 0.5),
            )
            for r in raw_insights
        ]

    def _format_patterns(self, patterns: list[LifePattern]) -> str:
        return "\n".join([
            f"- [{p.name}] {p.description[:150]} | Instances: {p.payload.instance_count} | Adaptive: {p.payload.is_adaptive}"
            for p in patterns[:8]
        ]) or "No patterns available"

    def _format_contradictions(self, contradictions: list[Contradiction]) -> str:
        return "\n".join([
            f"- {c.type}: {c.description[:150]} (score: {c.contradiction_score:.2f})"
            for c in contradictions[:5]
        ]) or "No contradictions found"

    def _format_value_gaps(self, value_gaps: list) -> str:
        if not value_gaps:
            return "No value gap analysis available"
        return "\n".join([
            f"- {g.value}: espoused={g.espoused_strength:.2f}, enacted={g.enacted_strength:.2f}, gap={g.gap:.2f}"
            for g in value_gaps[:6]
        ])

    def _format_psych_model(self, model: PsychologicalModel) -> str:
        return json.dumps({
            "core_fears": model.core_fears[:3],
            "core_motivators": model.core_motivators[:3],
            "self_concept": model.self_concept_beliefs[:4],
            "decision_style": model.decision_style,
        }, indent=2)

    async def _llm_generate(
        self,
        patterns: str,
        contradictions: str,
        value_gaps: str,
        psych_model: str
    ) -> list[dict]:
        prompt = STRATEGIC_INSIGHT_PROMPT.format(
            patterns=patterns,
            contradictions=contradictions,
            value_gaps=value_gaps,
            psych_model=psych_model
        )

        try:
            raw = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Strategic insight generation error: {e}")
            return []


@dataclass
class StrategicInsight:
    insight: str
    evidence: str
    why_non_obvious: str
    implication: str
    domain: str
    confidence: float
    generated_at: datetime = field(default_factory=datetime.utcnow)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICTIVE BEHAVIOR ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREDICTION_PROMPT = """You are analyzing one person's behavioral history to predict likely behavioral tendencies in a future scenario.

IMPORTANT: This is NOT a psychic prediction. It is a pattern-based probability assessment.
The output should feel like: "Given your history, here is what your data suggests you'll likely do —
and here's what would need to be different for a different outcome."

THEIR BEHAVIORAL PATTERNS:
{patterns}

THEIR DECISION HISTORY:
{decisions}

THEIR PSYCHOLOGICAL MODEL:
{psych_model}

SCENARIO TO ANALYZE:
{scenario}

Output:
{{
  "scenario_type": "what type of situation this is for them",
  "historical_parallels": ["2-3 most relevant past situations"],
  "most_likely_response": "what their pattern predicts they'll do",
  "emotional_trajectory": "how this is likely to feel for them",
  "likely_outcome_if_pattern_holds": "what will probably result",
  "pattern_override_conditions": "what would need to be true for them to respond differently",
  "growth_edge": "what this scenario is specifically asking them to develop",
  "confidence": 0.0-1.0,
  "caveat": "important limitations of this prediction"
}}"""


class PredictiveBehaviorEngine:
    """
    Projects behavioral tendencies into future scenarios.
    NOT prediction as certainty — prediction as probability grounded in pattern data.

    Use cases:
    - "If I take this job offer, what does my pattern say will happen?"
    - "How am I likely to handle this relationship challenge based on my history?"
    - "What does my data say about how I'll feel 6 months into this decision?"
    """

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key
        self.model = model

    async def predict(
        self,
        scenario: str,
        patterns: list[LifePattern],
        decisions: list[MemoryObject],
        psych_model: PsychologicalModel,
    ) -> BehavioralPrediction:
        """Generate a pattern-grounded behavioral prediction for a scenario."""

        patterns_text = "\n".join([
            f"- {p.name}: trigger={p.payload.trigger_description[:80]} | "
            f"behavior={p.payload.behavior_description[:80]} | "
            f"instances={p.payload.instance_count}"
            for p in patterns[:6]
        ])

        decisions_text = "\n".join([
            f"- [{m.date_info.occurred}] {m.title}: {m.content[:100]}"
            + (f" → Outcome: {m.decision_payload.get('outcome_short', '') if isinstance(m.decision_payload, dict) else ''}"
               if m.decision_payload else "")
            for m in decisions[:6]
        ])

        psych_text = json.dumps({
            "core_fears": psych_model.core_fears[:3],
            "core_motivators": psych_model.core_motivators[:3],
            "decision_style": psych_model.decision_style,
            "coping_maladaptive": psych_model.coping_strategies_maladaptive[:3],
        }, indent=2)

        raw = await self._llm_predict(scenario, patterns_text, decisions_text, psych_text)

        if not raw:
            return BehavioralPrediction(
                scenario=scenario,
                scenario_type="unknown",
                historical_parallels=[],
                most_likely_response="Insufficient pattern data for reliable prediction.",
                emotional_trajectory="Unknown",
                likely_outcome_if_pattern_holds="Unknown",
                pattern_override_conditions="Unknown",
                growth_edge="Unknown",
                confidence=0.0,
                caveat="Insufficient data",
            )

        return BehavioralPrediction(
            scenario=scenario,
            scenario_type=raw.get("scenario_type", ""),
            historical_parallels=raw.get("historical_parallels", []),
            most_likely_response=raw.get("most_likely_response", ""),
            emotional_trajectory=raw.get("emotional_trajectory", ""),
            likely_outcome_if_pattern_holds=raw.get("likely_outcome_if_pattern_holds", ""),
            pattern_override_conditions=raw.get("pattern_override_conditions", ""),
            growth_edge=raw.get("growth_edge", ""),
            confidence=raw.get("confidence", 0.5),
            caveat=raw.get("caveat", ""),
        )

    async def _llm_predict(
        self, scenario: str, patterns: str, decisions: str, psych_model: str
    ) -> Optional[dict]:
        prompt = PREDICTION_PROMPT.format(
            patterns=patterns,
            decisions=decisions,
            psych_model=psych_model,
            scenario=scenario
        )

        try:
            raw = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None


@dataclass
class BehavioralPrediction:
    scenario: str
    scenario_type: str
    historical_parallels: list[str]
    most_likely_response: str
    emotional_trajectory: str
    likely_outcome_if_pattern_holds: str
    pattern_override_conditions: str
    growth_edge: str
    confidence: float
    caveat: str
    generated_at: datetime = field(default_factory=datetime.utcnow)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LONG-TERM REFLECTION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LONG_TERM_REFLECTION_PROMPT = """You have access to one person's complete documented history across {years} years.

Perform a deep long-term reflection — the kind of synthesis that only becomes
possible when you can see patterns across years, not just weeks.

PERIOD SUMMARY:
{period_summary}

PATTERNS ACROSS THIS PERIOD:
{patterns}

IDENTITY EVOLUTION:
{identity_evolution}

STRATEGIC INSIGHTS:
{strategic_insights}

Generate a Long-Term Reflection with these components:

1. THE ARC (what was the overarching story of this period?)
2. THE GROWTH (what genuinely changed — not aspirational, but evidential?)
3. THE PERSISTENT LOOPS (what patterns are you still running that were present at the start?)
4. THE HIDDEN COSTS (what did you pay in this period that you may not have fully accounted for?)
5. THE EMERGING STRENGTHS (what is developing that deserves more attention and trust?)
6. THE UNRESOLVED (what from this period is still asking for resolution?)
7. THE QUESTION THIS PERIOD IS ASKING (the one central question your life is posing right now)

Rules:
- Ground every claim in specific patterns, memories, or documented evidence
- No generic wisdom — this is specific to this person
- The question at the end should be precise and uncomfortable in the right way
- Write in second person ("You...")
- 500-700 words total

Output JSON:
{{
  "arc": "...",
  "growth": "...",
  "persistent_loops": "...",
  "hidden_costs": "...",
  "emerging_strengths": "...",
  "unresolved": "...",
  "the_question": "...",
  "confidence": 0.0-1.0
}}"""


class LongTermReflectionEngine:
    """
    Generates deep, multi-year reflections that synthesize the full sweep
    of documented life data. Run periodically (monthly/quarterly/annually).

    These reflections become high-salience memory objects themselves,
    and inform the psychological model.
    """

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key
        self.model = model

    async def generate_annual_reflection(
        self,
        year: int,
        memories: list[MemoryObject],
        patterns: list[LifePattern],
        identity_evolution: dict,
        strategic_insights: list[StrategicInsight],
    ) -> LongTermReflection:
        """Generate an annual psychological review."""

        year_memories = [
            m for m in memories
            if m.date_info.occurred and m.date_info.occurred.year == year
        ]

        period_summary = self._build_period_summary(year, year_memories)
        years_covered = max(1, (
            max((m.date_info.occurred.year for m in memories if m.date_info.occurred), default=year)
            - min((m.date_info.occurred.year for m in memories if m.date_info.occurred), default=year)
            + 1
        ))

        raw = await self._llm_reflect(
            years=years_covered,
            period_summary=period_summary,
            patterns=self._format_patterns(patterns),
            identity_evolution=json.dumps(identity_evolution.get("dimension_summaries", [])[:5], indent=2),
            strategic_insights=self._format_insights(strategic_insights)
        )

        if not raw:
            return LongTermReflection(
                period=str(year),
                arc="",
                growth="",
                persistent_loops="",
                hidden_costs="",
                emerging_strengths="",
                unresolved="",
                the_question="What does this year most want you to understand?",
                confidence=0.0,
                memories_covered=len(year_memories),
            )

        return LongTermReflection(
            period=str(year),
            arc=raw.get("arc", ""),
            growth=raw.get("growth", ""),
            persistent_loops=raw.get("persistent_loops", ""),
            hidden_costs=raw.get("hidden_costs", ""),
            emerging_strengths=raw.get("emerging_strengths", ""),
            unresolved=raw.get("unresolved", ""),
            the_question=raw.get("the_question", ""),
            confidence=raw.get("confidence", 0.5),
            memories_covered=len(year_memories),
        )

    def _build_period_summary(
        self, year: int, memories: list[MemoryObject]
    ) -> str:
        if not memories:
            return f"Year {year}: No memories documented."

        # Emotional distribution
        emotions = [str(m.emotion.primary) for m in memories if m.emotion.primary]
        emotion_counts: dict[str, int] = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:4]

        # Avg valence
        valences = [m.emotion.valence for m in memories if m.emotion.valence != 0]
        avg_valence = sum(valences) / len(valences) if valences else 0

        # High-salience events
        high_salience = sorted(memories, key=lambda m: m.salience_score, reverse=True)[:5]

        lines = [
            f"YEAR {year} SUMMARY:",
            f"Total memories documented: {len(memories)}",
            f"Average emotional valence: {avg_valence:.2f} ({'positive lean' if avg_valence > 0.1 else 'negative lean' if avg_valence < -0.1 else 'neutral'})",
            f"Dominant emotions: {', '.join(f'{e}({c})' for e, c in top_emotions)}",
            "",
            "HIGHEST SALIENCE EVENTS:",
        ]
        for m in high_salience:
            date_str = str(m.date_info.occurred) if m.date_info.occurred else "unknown"
            lines.append(f"  - [{date_str}] {m.title} (salience {m.salience_score:.2f})")

        return "\n".join(lines)

    def _format_patterns(self, patterns: list[LifePattern]) -> str:
        return "\n".join([
            f"- {p.name}: {p.description[:120]} | Status: {p.payload.status} | Instances: {p.payload.instance_count}"
            for p in patterns[:6]
        ]) or "No patterns available"

    def _format_insights(self, insights: list[StrategicInsight]) -> str:
        return "\n".join([
            f"- {ins.insight[:120]}"
            for ins in insights[:3]
        ]) or "No strategic insights available"

    async def _llm_reflect(
        self,
        years: int,
        period_summary: str,
        patterns: str,
        identity_evolution: str,
        strategic_insights: str,
    ) -> Optional[dict]:
        prompt = LONG_TERM_REFLECTION_PROMPT.format(
            years=years,
            period_summary=period_summary,
            patterns=patterns,
            identity_evolution=identity_evolution,
            strategic_insights=strategic_insights,
        )

        try:
            raw = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500
            )
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Long-term reflection error: {e}")
            return None


@dataclass
class LongTermReflection:
    period: str
    arc: str
    growth: str
    persistent_loops: str
    hidden_costs: str
    emerging_strengths: str
    unresolved: str
    the_question: str
    confidence: float
    memories_covered: int
    generated_at: datetime = field(default_factory=datetime.utcnow)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WISDOM CRYSTALLIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WISDOM_CRYSTALLIZATION_PROMPT = """You have studied one person's complete documented life.

Your task: extract their EARNED WISDOM — the principles and knowledge they have
genuinely won through lived experience, not borrowed from books or culture.

Earned wisdom is:
- Only counted if there is behavioral or reflective evidence it was truly learned
- Specific to this person — not a generic platitude
- Won through experience — there must be a lesson-producing event in the history
- Tested — ideally there is evidence the person applied it

LESSONS DOCUMENTED:
{lessons}

PATTERNS (for evidence of repeated learning):
{patterns}

INSIGHTS (explicitly stated realizations):
{insights}

Generate their personal wisdom document:
{{
  "title": "Personal Wisdom: [their name or 'Your] Earned Philosophy",
  "principles": [
    {{
      "principle": "stated as they would say it, in their voice",
      "won_through": "the experience or pattern that earned this",
      "evidence_strength": 0.0-1.0,
      "domain": "where this applies most",
      "caveat": "when this principle has limits or exceptions"
    }}
  ],
  "areas_of_genuine_mastery": ["things they genuinely understand well through experience"],
  "areas_of_acknowledged_blindness": ["patterns they know about but haven't yet changed"],
  "working_philosophy": "2-3 sentences synthesizing their core life philosophy from evidence"
}}

Generate 5-8 principles. Use their own language where possible."""


class WisdomCrystallizer:
    """
    Distills all life data into a Personal Wisdom Document —
    the accumulated earned philosophy of one person's life.

    This is the highest-order synthesis in the system.
    It turns years of diary entries into a living document of
    hard-won personal wisdom, stated in the person's own voice.
    """

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key
        self.model = model

    async def crystallize(
        self,
        memories: list[MemoryObject],
        patterns: list[LifePattern],
    ) -> WisdomDocument:
        """Generate the personal wisdom document."""

        lessons = self._extract_all_lessons(memories)
        insights = self._extract_explicit_insights(memories)

        lessons_text = "\n".join([f"- {l}" for l in lessons[:20]])
        patterns_text = "\n".join([
            f"- {p.name}: {p.description[:100]}"
            for p in patterns[:8]
        ])
        insights_text = "\n".join([f"- {i}" for i in insights[:15]])

        raw = await self._llm_crystallize(lessons_text, patterns_text, insights_text)

        if not raw:
            return WisdomDocument(
                title="Wisdom Document",
                principles=[],
                areas_of_mastery=[],
                areas_of_blindness=[],
                working_philosophy="Insufficient data for wisdom synthesis.",
                memory_count=len(memories),
            )

        principles = [
            WisdomPrinciple(
                principle=p.get("principle", ""),
                won_through=p.get("won_through", ""),
                evidence_strength=p.get("evidence_strength", 0.5),
                domain=p.get("domain", "general"),
                caveat=p.get("caveat"),
            )
            for p in raw.get("principles", [])
        ]

        return WisdomDocument(
            title=raw.get("title", "Personal Wisdom Document"),
            principles=principles,
            areas_of_mastery=raw.get("areas_of_genuine_mastery", []),
            areas_of_blindness=raw.get("areas_of_acknowledged_blindness", []),
            working_philosophy=raw.get("working_philosophy", ""),
            memory_count=len(memories),
        )

    def _extract_all_lessons(self, memories: list[MemoryObject]) -> list[str]:
        lessons = []
        for m in memories:
            if m.lesson_content:
                lessons.append(m.lesson_content)
        return lessons

    def _extract_explicit_insights(self, memories: list[MemoryObject]) -> list[str]:
        insights = []
        for m in memories:
            if m.memory_type == MemoryType.INSIGHT and m.original_language:
                insights.append(m.original_language[:120])
        return insights

    async def _llm_crystallize(
        self, lessons: str, patterns: str, insights: str
    ) -> Optional[dict]:
        prompt = WISDOM_CRYSTALLIZATION_PROMPT.format(
            lessons=lessons,
            patterns=patterns,
            insights=insights
        )

        try:
            raw = await call_gemini_openai(
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000
            )
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Wisdom crystallization error: {e}")
            return None


@dataclass
class WisdomPrinciple:
    principle: str
    won_through: str
    evidence_strength: float
    domain: str
    caveat: Optional[str] = None


@dataclass
class WisdomDocument:
    title: str
    principles: list[WisdomPrinciple]
    areas_of_mastery: list[str]
    areas_of_blindness: list[str]
    working_philosophy: str
    memory_count: int
    generated_at: datetime = field(default_factory=datetime.utcnow)
