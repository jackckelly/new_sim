import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    content: str
    timestamp: datetime
    importance: float
    context: Dict[str, str]


class ShortTermMemory:
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.memories: List[Memory] = []

    def add_memory(
        self, content: str, importance: float = 1.0, context: Dict[str, str] = None
    ):
        if context is None:
            context = {}

        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            context=context,
        )

        self.memories.append(memory)
        if len(self.memories) > self.capacity:
            # Remove oldest memory with lowest importance
            self.memories.sort(key=lambda x: (x.importance, x.timestamp))
            self.memories.pop(0)

    def get_recent_memories(self, n: int = None) -> List[Memory]:
        if n is None:
            n = self.capacity
        return sorted(self.memories, key=lambda x: x.timestamp, reverse=True)[:n]


class LongTermMemory:
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.memories: List[Memory] = []
        self.embeddings: List[np.ndarray] = []

    def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        importance: float = 1.0,
        context: Dict[str, str] = None,
    ):
        if context is None:
            context = {}

        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            context=context,
        )

        self.memories.append(memory)
        self.embeddings.append(embedding)

        if len(self.embeddings) > 1:
            embeddings_array = np.array(self.embeddings).astype("float32")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings_array)

    def search_memories(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[Memory, float]]:
        if not self.memories:
            return []

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), k
        )
        results = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.memories):
                results.append((self.memories[idx], float(distance)))

        return results

    def get_all_memories(self) -> List[Memory]:
        return self.memories.copy()
