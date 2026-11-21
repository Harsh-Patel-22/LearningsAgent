"""
PHASE 3 — MASTER REASONING AGENT
Cognitive Memory OS

The reasoning agent is NOT a chatbot wrapper.
It is a multi-stage cognitive reasoning engine that:

1. Receives a ContextPackage (retrieved memories + patterns + contradictions)
2. Executes a structured chain-of-thought reasoning process
3. Generates a response that is:
   - Grounded in specific memories with citations
   - Pattern-aware (connects to behavioral loops)
   - Temporally precise (respects chronological integrity)
   - Psychologically honest (names contradictions, not just comforting)
   - Confidence-calibrated (explicit uncertainty acknowledgment)

The agent NEVER:
- Makes claims without a supporting memory ID
- Cites external psychology over personal experience
- Speaks generically when personal data exists
- Fabricates memories or fills gaps with plausible-sounding content
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, AsyncGenerator

import httpx

from models import (
    ContextPackage, MemoryObject, MemoryType,
    LifePattern, Contradiction, RetrievalResult
)
from windows_compat import call_gemini_openai

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

MASTER_AGENT_SYSTEM_PROMPT = """You are a Cognitive Memory Agent — a deeply personalized AI built entirely from one person's documented life: their diary entries, reflections, lessons, emotional patterns, behavioral loops, decisions, and psychological evolution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU ARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are not a general AI assistant.
You are the user's wiser, more memory-complete self.
You have access to everything they have ever written about their inner and outer life.
You speak like someone who has read every journal entry they've ever written, observed every pattern across years, and understands their psychology in precise, specific terms — grounded entirely in what they themselves documented.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE BEHAVIORAL LAWS (NEVER VIOLATE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GROUND EVERY CLAIM IN A SPECIFIC MEMORY
   Before stating any fact about the user's life, a memory must exist in your context that supports it.
   Format: "In your [date] entry..." or "Your [life phase] notes show..."
   If no memory supports a claim → don't make it. State the gap explicitly.

2. PERSONAL EXPERIENCE BEFORE GENERAL KNOWLEDGE
   If your context contains relevant personal data → use it FIRST.
   Only reference general knowledge when zero personal data applies.
   Never say "research shows" or "psychologists say" when your data says it better.

3. PRESERVE CHRONOLOGICAL INTEGRITY
   A 2018 belief must be understood as a 2018 belief.
   Do not interpret past entries through the lens of current understanding.
   Always state the time context: "At that time (2018) you believed..."

4. CALIBRATE CONFIDENCE EXPLICITLY
   High confidence: "Your data clearly shows..." (3+ supporting memories)
   Medium confidence: "Your pattern suggests, though evidence is limited..."
   Low confidence: "I have only one entry about this — hold this loosely."
   No data: "I don't have documented evidence about this period/topic."

5. NAME CONTRADICTIONS, DON'T SMOOTH THEM
   If your context contains a contradiction between belief and behavior → surface it.
   Do this with precision and compassion, not judgment.

6. NEVER FABRICATE MEMORIES
   If a specific date, event, or statement isn't in your context → say so.
   "I don't have your entry from that period" is better than a plausible-sounding invention.

7. CITE SOURCES IN EVERY SUBSTANTIVE RESPONSE
   End factual claims with: [Entry: mem_id (Date: YYYY-MM-DD) — Brief Description] or [Pattern: {pattern_name}]
   This makes the response verifiable by the user against their actual source material.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pattern queries → Precise, analytical. Name the loop clearly.
Emotional queries → Witnessing, non-judgmental. Hold the weight of what's shared.
Decision support → Strategic. Use historical precedent, not generic frameworks.
Self-criticism → Honest but compassionate. Name the pattern AND the growth.
Sensitive material → Slower, more careful. Acknowledge the weight before the insight.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Anchor → Specific memory reference that grounds the response
2. Pattern → Connect to broader behavioral loop if applicable
3. Current relevance → Bridge historical pattern to present question
4. Contradiction check → Surface relevant contradictions if present
5. Confidence statement → Be explicit about data quality
6. Single reflective question → One question to prompt their thinking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU NEVER DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Never give generic life advice not grounded in their data
- Never pretend to know something you don't have data for
- Never be sycophantic ("Great question!")
- Never act as a therapist or make clinical interpretations
- Never moralize — observe, reflect, illuminate
- Never use bullet points when a specific narrative serves better"""


REASONING_CHAIN_PROMPT = """You are about to answer a query using a Context Package of retrieved personal memories.

Before generating the final response, work through this reasoning chain internally:

STEP 1 — CONTEXT GROUNDING
What life phase is this person currently in?
What major themes are active in their life right now?
What emotional state does the most recent data suggest?

STEP 2 — HISTORICAL PARALLEL DETECTION
Which retrieved memories are most structurally similar to the current question?
Similarity is multi-dimensional: emotional structure + situation type + decision stakes + relationship dynamic.
Not just surface content — what FELT similar, what HAD similar stakes, what involved similar patterns?

STEP 3 — PATTERN APPLICATION
For the closest historical parallels:
- What behavioral pattern was active?
- What did they do?
- What was the outcome?
- Has this pattern changed or persisted?
- What did they say about it at the time vs later?

STEP 4 — CONTRADICTION CHECK
Does the current situation involve:
- A lesson they've already documented but may have forgotten?
- A value they've stated that conflicts with the current direction?
- A pattern they've identified as problematic that's showing up again?

STEP 5 — GROWTH TRAJECTORY
Where is this person on their growth arc regarding this issue?
Have they already worked through this? Is this regression? Is this new territory?

STEP 6 — CONFIDENCE CALIBRATION
How many memories support the main insight?
How recent is the most relevant data?
What gaps exist in the data for this topic?

Now generate the response following the structure above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT PACKAGE:
{context_package}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUERY: {query}"""


# ─────────────────────────────────────────────
# CONTEXT PACKAGE FORMATTER
# ─────────────────────────────────────────────

class ContextPackageFormatter:
    """
    Formats the ContextPackage into a structured text representation
    for injection into the reasoning prompt.
    Optimized for LLM comprehension and citation traceability.
    """

    def format(self, package: ContextPackage, max_tokens: int = 12000) -> str:
        sections = []

        # Current life context
        sections.append(self._format_life_context(package))

        # Retrieved memories (chronologically grouped)
        sections.append(self._format_memories(package.retrieved_memories))

        # Active patterns
        if package.active_patterns:
            sections.append(self._format_patterns(package.active_patterns))

        # Relevant contradictions
        if package.relevant_contradictions:
            sections.append(self._format_contradictions(package.relevant_contradictions))

        # Psych model snapshot
        if package.psych_model_snapshot:
            sections.append(self._format_psych_model(package.psych_model_snapshot))

        full_text = "\n\n".join(sections)

        # Token budget management
        if len(full_text) > max_tokens * 4:  # rough char estimate
            full_text = full_text[:max_tokens * 4]
            full_text += "\n\n[Context truncated for length — additional memories available]"

        return full_text

    def _format_life_context(self, package: ContextPackage) -> str:
        domains = [d if isinstance(d, str) else d.value for d in package.query_domains]
        lines = [
            "═══ CURRENT LIFE CONTEXT ═══",
            f"Query intent: {package.query_intent}",
            f"Relevant domains: {', '.join(domains) if domains else 'general'}",
        ]
        if package.time_filter:
            start, end = package.time_filter
            lines.append(f"Time filter: {start} to {end}")
        if package.psych_model_snapshot:
            psm = package.psych_model_snapshot
            if psm.get("current_life_phase"):
                lines.append(f"Current life phase: {psm['current_life_phase']}")
            if psm.get("dominant_themes"):
                lines.append(f"Active themes: {', '.join(psm['dominant_themes'][:3])}")
        return "\n".join(lines)

    def _format_memories(self, results: list[RetrievalResult]) -> str:
        if not results:
            return "═══ RETRIEVED MEMORIES ═══\nNo relevant memories found."

        # Sort chronologically for narrative coherence
        sorted_results = sorted(
            results,
            key=lambda r: r.memory.date_info.occurred or datetime.min.date(),
        )

        lines = [f"═══ RETRIEVED MEMORIES ({len(results)} total) ═══"]

        for i, result in enumerate(sorted_results):
            mem = result.memory
            date_str = str(mem.date_info.occurred) if mem.date_info.occurred else "date unknown"
            date_conf = f"(date confidence: {mem.date_info.confidence:.0%})" if mem.date_info.confidence < 0.8 else ""
            phase_str = f" | {mem.date_info.life_phase}" if mem.date_info.life_phase else ""
            emotion_str = f"Emotion: {mem.emotion.primary} (intensity {mem.emotion.intensity:.2f})"
            domains = [d if isinstance(d, str) else d.value for d in mem.domains]

            block = [
                f"\n[Memory {i+1}] — ID: {mem.id}",
                f"Date: {date_str} {date_conf}{phase_str}",
                f"Type: {mem.memory_type} | Domains: {', '.join(domains[:3])}",
                f"{emotion_str}",
                f"Retrieval score: {result.retrieval_score:.3f}",
                f"",
                f"TITLE: {mem.title}",
                f"CONTENT: {mem.content}",
            ]

            if mem.original_language:
                block.append(f'ORIGINAL LANGUAGE: "{mem.original_language}"')
            if mem.lesson_content:
                block.append(f"LESSON DOCUMENTED: {mem.lesson_content}")
            if mem.pattern_tags:
                block.append(f"Pattern tags: {', '.join(mem.pattern_tags)}")
            if result.graph_hops > 0:
                block.append(f"[Retrieved via graph traversal, {result.graph_hops} hops from direct match]")

            lines.append("\n".join(block))
            lines.append("─" * 50)

        return "\n".join(lines)

    def _format_patterns(self, patterns: list[LifePattern]) -> str:
        lines = [f"═══ ACTIVE BEHAVIORAL PATTERNS ({len(patterns)}) ═══"]
        for p in patterns:
            loop = {}
            try:
                loop = json.loads(p.loop_description)
            except Exception:
                pass

            block = [
                f"\n[Pattern: {p.name}]",
                f"Description: {p.description}",
                f"Instances: {p.payload.instance_count} | Since: {p.payload.first_occurrence or 'unknown'}",
                f"Status: {p.payload.status} | Significance: {p.significance_score:.2f}",
            ]
            if loop:
                block.append(f"Loop: trigger={loop.get('trigger','')} → behavior={loop.get('behavior','')} → outcome={loop.get('outcome','')}")
            if p.payload.is_adaptive is not None:
                block.append(f"Adaptive: {'Yes' if p.payload.is_adaptive else 'No'}")

            lines.append("\n".join(block))

        return "\n".join(lines)

    def _format_contradictions(self, contradictions: list[Contradiction]) -> str:
        lines = [f"═══ DETECTED CONTRADICTIONS ({len(contradictions)}) ═══"]
        for c in contradictions:
            block = [
                f"\n[Contradiction: {c.type}]",
                f"Score: {c.contradiction_score:.2f}",
                f"Description: {c.description}",
                f"Memory A: {c.memory_a_id} | Memory B: {c.memory_b_id}",
            ]
            if c.temporal_gap_days:
                block.append(f"Gap: {c.temporal_gap_days} days between these memories")
            lines.append("\n".join(block))

        return "\n".join(lines)

    def _format_psych_model(self, psm: dict) -> str:
        lines = ["═══ PSYCHOLOGICAL MODEL SNAPSHOT ═══"]
        for key, val in psm.items():
            if val and key not in ("user_id", "model_version"):
                lines.append(f"{key}: {val}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# CITATION VERIFIER (Anti-hallucination layer)
# ─────────────────────────────────────────────

class CitationVerifier:
    """
    Post-generation verification pass.
    Checks that every specific factual claim in the response
    is traceable to a memory in the context package.

    This is the primary anti-hallucination mechanism.
    """

    CLAIM_PATTERNS = [
        r"In (?:your )?\d{4}",             # "In 2021..."
        r"In (?:your )?(?:January|February|March|April|May|June|July|August|September|October|November|December)",
        r"You (?:wrote|documented|noted|described|mentioned)",
        r"Your (?:entry|diary|notes?) (?:from|in|about)",
        r"\[Entry: .+?\]",
        r"\[Pattern: .+?\]",
    ]

    def verify(
        self,
        response_text: str,
        context_package: ContextPackage
    ) -> VerificationResult:
        """
        Scan response for factual claims and check against context.
        Returns verification result with unverifiable claims flagged.
        """
        # Extract all memory IDs from context
        available_memory_ids = {r.memory.id for r in context_package.retrieved_memories}
        available_dates = set()
        for r in context_package.retrieved_memories:
            if r.memory.date_info.occurred:
                available_dates.add(str(r.memory.date_info.occurred))
                available_dates.add(str(r.memory.date_info.occurred.year))

        # Find all citation markers in response
        cited_ids = re.findall(r"\[Entry: (mem_[a-zA-Z0-9_]+)(?:\s+[^\]]+)?\]", response_text)
        cited_patterns = re.findall(r"\[Pattern: ([a-zA-Z0-9_-]+)\]", response_text)

        unverifiable = []
        for cid in cited_ids:
            if cid not in available_memory_ids:
                unverifiable.append(f"Memory ID {cid} not in context")

        # Check for year claims that don't match available data
        year_claims = re.findall(r"In (?:your )?(20\d{2})", response_text)
        for year in year_claims:
            if year not in available_dates:
                unverifiable.append(f"Year {year} claimed but no memories from that year in context")

        verified = len(unverifiable) == 0
        return VerificationResult(
            is_verified=verified,
            unverifiable_claims=unverifiable,
            citation_count=len(cited_ids) + len(cited_patterns),
            verification_notes=f"{len(cited_ids)} memory citations, {len(cited_patterns)} pattern citations"
        )


@dataclass
class VerificationResult:
    is_verified: bool
    unverifiable_claims: list[str]
    citation_count: int
    verification_notes: str


# ─────────────────────────────────────────────
# MASTER REASONING AGENT
# ─────────────────────────────────────────────

@dataclass
class AgentResponse:
    response_text: str
    reasoning_trace: Optional[str]
    citations: list[str]
    confidence_level: str
    query_intent: str
    memories_used: list[str]
    patterns_referenced: list[str]
    contradictions_surfaced: list[str]
    verification_result: Optional[VerificationResult]
    latency_ms: float
    model_used: str


class MasterReasoningAgent:
    """
    The cognitive core of the system.
    Receives a ContextPackage, executes chain-of-thought reasoning,
    generates a cited, grounded, psychologically honest response.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-20250514",
        enable_streaming: bool = True,
        enable_citation_verification: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.enable_streaming = enable_streaming
        self.enable_citation_verification = enable_citation_verification
        self.formatter = ContextPackageFormatter()
        self.verifier = CitationVerifier()

    async def respond(
        self,
        query: str,
        context_package: ContextPackage,
        conversation_history: Optional[list[dict]] = None
    ) -> AgentResponse:
        """
        Generate a grounded, cited response to the user's query.
        Includes internal reasoning chain before final generation.
        """
        import time
        start = time.time()

        # Format context for prompt
        context_text = self.formatter.format(context_package, max_tokens=10000)

        # Build messages
        messages = self._build_messages(
            query=query,
            context_text=context_text,
            conversation_history=conversation_history or []
        )

        # Generate with reasoning chain
        response_text, reasoning_trace = await self._generate_with_reasoning(messages)

        # Citation verification
        verification = None
        if self.enable_citation_verification:
            verification = self.verifier.verify(response_text, context_package)
            if not verification.is_verified and verification.unverifiable_claims:
                logger.warning(
                    f"Unverifiable claims detected: {verification.unverifiable_claims}"
                )
                # In production: trigger a correction pass here

        # Extract metadata from response
        citations = re.findall(r"\[(?:Entry|Pattern|Memory): [^\]]+\]", response_text)
        memory_ids_used = re.findall(r"\[Entry: (mem_[a-zA-Z0-9_]+)(?:\s+[^\]]+)?\]", response_text)
        patterns_referenced = re.findall(r"\[Pattern: ([a-zA-Z0-9_-]+)\]", response_text)

        # Update reinforcement counts for used memories
        for result in context_package.retrieved_memories:
            if result.memory.id in memory_ids_used:
                result.memory.reinforcement_count += 1

        latency = (time.time() - start) * 1000

        return AgentResponse(
            response_text=response_text,
            reasoning_trace=reasoning_trace,
            citations=citations,
            confidence_level=self._assess_confidence(context_package),
            query_intent=context_package.query_intent,
            memories_used=memory_ids_used,
            patterns_referenced=patterns_referenced,
            contradictions_surfaced=[c.id for c in context_package.relevant_contradictions],
            verification_result=verification,
            latency_ms=round(latency, 2),
            model_used=self.model
        )

    async def stream_respond(
        self,
        query: str,
        context_package: ContextPackage
    ) -> AsyncGenerator[str, None]:
        """Streaming response for real-time UI display."""
        from pii_masker import PIIMasker
        masker = PIIMasker()

        context_text = self.formatter.format(context_package)
        messages = self._build_messages(query, context_text)
        
        # Mask system prompt and messages
        masked_sys_prompt, sys_mapping = masker.mask_text(MASTER_AGENT_SYSTEM_PROMPT)
        masked_messages, msg_mapping = masker.mask_messages(messages)
        combined_mapping = {**sys_mapping, **msg_mapping}

        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gemini-3.1-flash" if "claude" in self.model or "gpt" in self.model else self.model,
            "max_tokens": 2048,
            "messages": [{"role": "system", "content": masked_sys_prompt}] + masked_messages,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=payload
            ) as response:
                full_text = ""
                yielded_len = 0
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            delta = event.get("choices", [{}])[0].get("delta", {})
                            content_delta = delta.get("content")
                            if content_delta:
                                full_text += content_delta
                                unmasked = masker.unmask_text(full_text, combined_mapping)
                                
                                # Do not yield trailing partial placeholders
                                last_bracket = unmasked.rfind("[")
                                if last_bracket != -1 and "]" not in unmasked[last_bracket:]:
                                    to_yield = unmasked[yielded_len:last_bracket]
                                    yielded_len = last_bracket
                                else:
                                    to_yield = unmasked[yielded_len:]
                                    yielded_len = len(unmasked)
                                
                                if to_yield:
                                    yield to_yield
                        except json.JSONDecodeError:
                            continue
                
                # Yield any remaining text
                unmasked = masker.unmask_text(full_text, combined_mapping)
                if yielded_len < len(unmasked):
                    yield unmasked[yielded_len:]

    async def _generate_with_reasoning(
        self, messages: list[dict]
    ) -> tuple[str, Optional[str]]:
        """
        Generate response with extended thinking / reasoning chain.
        Uses Gemini for reasoning steps.
        """
        # First: generate reasoning trace
        reasoning_messages = messages.copy()
        reasoning_messages[-1]["content"] = (
            "Before answering, work through the 6-step reasoning chain:\n"
            "STEP 1 - CONTEXT GROUNDING:\n"
            "STEP 2 - HISTORICAL PARALLEL DETECTION:\n"
            "STEP 3 - PATTERN APPLICATION:\n"
            "STEP 4 - CONTRADICTION CHECK:\n"
            "STEP 5 - GROWTH TRAJECTORY:\n"
            "STEP 6 - CONFIDENCE CALIBRATION:\n\n"
            f"ORIGINAL QUERY: {messages[-1]['content']}"
        )

        reasoning_trace = await call_gemini_openai(
            api_key=self.api_key,
            model=self.model,
            messages=reasoning_messages,
            system_prompt=MASTER_AGENT_SYSTEM_PROMPT,
            max_tokens=2048
        )

        # Second: generate final response using reasoning as context
        final_messages = messages + [
            {"role": "assistant", "content": f"[Internal reasoning complete]\n\n{reasoning_trace}"},
            {"role": "user", "content": "Now generate the final response to the user based on your reasoning above. Do not repeat the reasoning steps — deliver the insight directly, grounded in specific memories."}
        ]

        final_text = await call_gemini_openai(
            api_key=self.api_key,
            model=self.model,
            messages=final_messages,
            system_prompt=MASTER_AGENT_SYSTEM_PROMPT,
            max_tokens=2048
        )

        return final_text, reasoning_trace

    def _build_messages(
        self,
        query: str,
        context_text: str,
        conversation_history: Optional[list[dict]] = None
    ) -> list[dict]:
        messages = []

        # Add conversation history (last 6 turns max)
        if conversation_history:
            messages.extend(conversation_history[-12:])

        # Main query with context
        messages.append({
            "role": "user",
            "content": REASONING_CHAIN_PROMPT.format(
                context_package=context_text,
                query=query
            )
        })

        return messages

    def _assess_confidence(self, package: ContextPackage) -> str:
        """Assess overall response confidence based on retrieval quality."""
        if not package.retrieved_memories:
            return "very_low"

        avg_score = sum(r.retrieval_score for r in package.retrieved_memories) / len(package.retrieved_memories)
        count = len(package.retrieved_memories)

        if avg_score > 0.7 and count >= 5:
            return "high"
        elif avg_score > 0.5 and count >= 3:
            return "medium"
        elif count >= 2:
            return "low"
        else:
            return "very_low"
