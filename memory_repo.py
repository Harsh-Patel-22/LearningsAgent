import os
import json
import logging
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Dict, Any

import httpx
from pydantic import BaseModel
from models import (
    SourceDocument, MemoryObject, LifePattern, Contradiction,
    PsychologicalModel, RetrievalResult, ContextPackage, MemoryType, LifeDomain
)

logger = logging.getLogger(__name__)

# Base directory for local storage
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

class SimpleMemoryRepo:
    def __init__(self):
        self.docs_dir = DATA_DIR / "source_docs"
        self.memories_dir = DATA_DIR / "memories"
        self.docs_dir.mkdir(exist_ok=True)
        self.memories_dir.mkdir(exist_ok=True)

    async def save_source_doc(self, doc: SourceDocument):
        file_path = self.docs_dir / f"{doc.id}.json"
        file_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved source doc {doc.id} to {file_path}")

    async def update_source_doc(self, doc: SourceDocument):
        await self.save_source_doc(doc)

    async def save_memory(self, memory: MemoryObject):
        file_path = self.memories_dir / f"{memory.id}.json"
        file_path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved memory {memory.id} to {file_path}")

    async def get_all(self, limit: int = 5000) -> List[MemoryObject]:
        memories = []
        for file_path in self.memories_dir.glob("*.json"):
            try:
                data = file_path.read_text(encoding="utf-8")
                mem = MemoryObject.model_validate_json(data)
                memories.append(mem)
                if len(memories) >= limit:
                    break
            except Exception as e:
                logger.error(f"Error loading memory from {file_path}: {e}")
        return memories

    async def get_memory(self, memory_id: str) -> Optional[MemoryObject]:
        file_path = self.memories_dir / f"{memory_id}.json"
        if file_path.exists():
            try:
                return MemoryObject.model_validate_json(file_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Error reading memory {memory_id}: {e}")
        return None


class SimplePatternRepo:
    def __init__(self):
        self.patterns_dir = DATA_DIR / "patterns"
        self.patterns_dir.mkdir(exist_ok=True)

    async def upsert(self, pattern: LifePattern):
        file_path = self.patterns_dir / f"{pattern.id}.json"
        file_path.write_text(pattern.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved pattern {pattern.id} to {file_path}")

    async def get_all(self) -> List[LifePattern]:
        patterns = []
        for file_path in self.patterns_dir.glob("*.json"):
            try:
                data = file_path.read_text(encoding="utf-8")
                pat = LifePattern.model_validate_json(data)
                patterns.append(pat)
            except Exception as e:
                logger.error(f"Error loading pattern from {file_path}: {e}")
        return patterns


class SimpleContradictionRepo:
    def __init__(self):
        self.contradictions_dir = DATA_DIR / "contradictions"
        self.contradictions_dir.mkdir(exist_ok=True)

    async def save(self, contradiction: Contradiction):
        file_path = self.contradictions_dir / f"{contradiction.id}.json"
        file_path.write_text(contradiction.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved contradiction {contradiction.id} to {file_path}")

    async def get_all(self) -> List[Contradiction]:
        contradictions = []
        for file_path in self.contradictions_dir.glob("*.json"):
            try:
                data = file_path.read_text(encoding="utf-8")
                con = Contradiction.model_validate_json(data)
                contradictions.append(con)
            except Exception as e:
                logger.error(f"Error loading contradiction from {file_path}: {e}")
        return contradictions


class SimplePsychModelStore:
    def __init__(self):
        self.file_path = DATA_DIR / "psych_model.json"

    async def get_current(self) -> Optional[PsychologicalModel]:
        if self.file_path.exists():
            try:
                data = self.file_path.read_text(encoding="utf-8")
                return PsychologicalModel.model_validate_json(data)
            except Exception as e:
                logger.error(f"Error reading psych model: {e}")
        # Return default if not found
        return PsychologicalModel()

    async def save(self, model: PsychologicalModel):
        self.file_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved psychological model to {self.file_path}")


# ─────────────────────────────────────────────
# MISSING INFRASTRUCTURE / UTILITIES CLASSES
# ─────────────────────────────────────────────

class MultiSpaceEmbeddingEngine:
    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        # We prefer gemini_api_key, fallback to env var
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    async def embed_memory(self, memory: MemoryObject) -> Dict[str, List[float]]:
        # Use primary text content (title + content) for embedding
        text_to_embed = f"{memory.title}\n{memory.content}"
        vector = await self.get_gemini_embedding(text_to_embed)
        
        # Return same vector for all spaces (semantic, emotional, etc.)
        return {
            "semantic": vector,
            "emotional": vector
        }

    async def get_gemini_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not configured, using dummy embedding vector")
            return [0.0] * 3072

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["embedding"]["values"]
        except Exception as e:
            logger.error(f"Error generating embedding via Gemini API: {e}. Falling back to dummy vector.")
            return [0.0] * 3072


class QdrantMemoryStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
        self._connected = False
        self._init_client()

    def _init_client(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
            # Simple connection check
            self.client.get_collections()
            self._connected = True
            logger.info("Connected to Qdrant successfully.")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}. Fallback to in-memory matching will be used.")

    def initialize_collections(self):
        if not self._connected:
            return
        from qdrant_client.models import VectorParams, Distance
        for space in ["semantic", "emotional"]:
            try:
                self.client.create_collection(
                    collection_name=space,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {space}")
            except Exception as e:
                # Already exists or failure
                pass

    def upsert_memory(self, memory_id: str, embeddings: Dict[str, List[float]], payload: Dict[str, Any]):
        if not self._connected:
            return
        from qdrant_client.models import PointStruct
        for space, vector in embeddings.items():
            try:
                self.client.upsert(
                    collection_name=space,
                    points=[
                        PointStruct(
                            id=abs(hash(memory_id)) % (10 ** 10),  # Numerical ID for Qdrant
                            vector=vector,
                            payload=payload
                        )
                    ]
                )
            except Exception as e:
                logger.error(f"Error upserting memory to Qdrant collection '{space}': {e}")


class GraphMemoryExpander:
    def __init__(self):
        pass

    def expand(self, memory_ids: List[str]) -> List[str]:
        # Return empty list as fallback or pass-through
        return []


class MemoryRetrievalOrchestrator:
    def __init__(
        self,
        embedding_engine: MultiSpaceEmbeddingEngine,
        vector_store: QdrantMemoryStore,
        graph_expander: GraphMemoryExpander,
        memory_repository: SimpleMemoryRepo,
        patterns_store: SimplePatternRepo,
        contradictions_store: SimpleContradictionRepo,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.graph_expander = graph_expander
        self.memory_repo = memory_repository
        self.pattern_repo = patterns_store
        self.contradiction_repo = contradictions_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 12,
        include_patterns: bool = True,
        include_contradictions: bool = True,
    ) -> ContextPackage:
        import time
        start_time = time.time()
        
        all_memories = await self.memory_repo.get_all()
        retrieved_results = []

        # Try to use Qdrant if connected
        if self.vector_store._connected:
            try:
                query_vector = await self.embedding_engine.get_gemini_embedding(query)
                from qdrant_client.models import SearchRequest
                if hasattr(self.vector_store.client, "query_points"):
                    hits = self.vector_store.client.query_points(
                        collection_name="semantic",
                        query=query_vector,
                        limit=top_k
                    ).points
                else:
                    hits = self.vector_store.client.search(
                        collection_name="semantic",
                        query_vector=query_vector,
                        limit=top_k
                    )
                
                # Resolve MemoryObjects from repo based on search hits
                for hit in hits:
                    mem_id = hit.payload.get("memory_id")
                    if mem_id:
                        mem = await self.memory_repo.get_memory(mem_id)
                        if mem:
                            retrieved_results.append(
                                RetrievalResult(
                                    memory=mem,
                                    retrieval_score=hit.score,
                                    semantic_score=hit.score,
                                    retrieval_path="vector"
                                )
                            )
            except Exception as e:
                logger.error(f"Qdrant search failed, falling back to local scan: {e}")
                
        # Fallback keyword scan if retrieved_results is empty
        if not retrieved_results:
            query_words = set(query.lower().split())
            scored_memories = []
            for mem in all_memories:
                score = 0.0
                content_lower = mem.content.lower() + " " + mem.title.lower()
                for word in query_words:
                    if word in content_lower:
                        score += 1.0
                if score > 0.0:
                    scored_memories.append((mem, score / len(query_words)))

            scored_memories.sort(key=lambda x: x[1], reverse=True)
            for mem, score in scored_memories[:top_k]:
                retrieved_results.append(
                    RetrievalResult(
                        memory=mem,
                        retrieval_score=score,
                        semantic_score=score,
                        retrieval_path="keyword"
                    )
                )

        # Fallback to top_k default if no matches found
        if not retrieved_results and all_memories:
            for mem in all_memories[:top_k]:
                retrieved_results.append(
                    RetrievalResult(
                        memory=mem,
                        retrieval_score=0.5,
                        semantic_score=0.5,
                        retrieval_path="default"
                    )
                )

        # Get active patterns and contradictions
        patterns = await self.pattern_repo.get_all() if include_patterns else []
        contradictions = await self.contradiction_repo.get_all() if include_contradictions else []

        latency = (time.time() - start_time) * 1000

        # Infer LifeDomains from query
        domains = []
        for domain in LifeDomain:
            if domain.value in query.lower():
                domains.append(domain)
        if not domains:
            domains = [LifeDomain.GENERAL]

        return ContextPackage(
            query=query,
            query_intent="specific_recall",
            query_domains=domains,
            retrieved_memories=retrieved_results,
            active_patterns=patterns,
            relevant_contradictions=contradictions,
            total_memories_searched=len(all_memories),
            retrieval_latency_ms=round(latency, 2)
        )
