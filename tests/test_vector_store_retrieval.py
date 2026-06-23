"""Isolated vector-store retrieval test.

Stores a sample document in a temporary Chroma directory, runs similarity
search, and confirms the indexed chunk is returned. Does not use the app's
production ``chroma_db/`` collection.
"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from vector_store import create_or_update_vector_store, get_embeddings

SAMPLE_SOURCE = "sample-hr-policy.pdf"
SAMPLE_TEXT = (
    "Employees receive twelve days of annual leave per calendar year. "
    "Sick leave is separate and provides seven days annually."
)
SEARCH_QUERY = "How many annual leave days do employees get?"


class VectorStoreRetrievalTest(unittest.TestCase):
    """Verify a stored sample document can be retrieved by similarity search."""

    def setUp(self) -> None:
        """Create an isolated Chroma persist directory for this test run."""
        session_id = uuid.uuid4().hex[:8]
        self.persist_dir = str(
            Path("./chroma_db") / "tests_vector_store" / session_id
        )
        self.sample_chunk = Document(
            page_content=SAMPLE_TEXT,
            metadata={
                "source": SAMPLE_SOURCE,
                "page": 0,
                "page_label": 1,
                "line": 1,
                "start_index": 0,
            },
        )

    def tearDown(self) -> None:
        """Remove the temporary test vector store from disk."""
        shutil.rmtree(self.persist_dir, ignore_errors=True)

    def test_store_search_and_retrieve_sample_document(self) -> None:
        """Store one chunk, search for it, and confirm it comes back."""
        get_embeddings()
        vector_store = create_or_update_vector_store(
            [self.sample_chunk],
            persist_dir=self.persist_dir,
        )

        results = vector_store.similarity_search(SEARCH_QUERY, k=1)

        self.assertGreaterEqual(len(results), 1, "Expected at least one result")
        top_result = results[0]
        self.assertIn(
            "annual leave",
            top_result.page_content.lower(),
            "Retrieved text should mention annual leave",
        )
        self.assertEqual(
            top_result.metadata.get("source"),
            SAMPLE_SOURCE,
            "Retrieved chunk should come from the sample document",
        )


if __name__ == "__main__":
    unittest.main()
