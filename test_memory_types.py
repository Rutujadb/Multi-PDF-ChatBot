"""Standalone memory-type comparison script for learning and isolated testing.

Does not modify any project modules. Run with::

    python test_memory_types.py

Uses OpenRouter via the same configuration values as ``rag_chain.py``
(``OPENROUTER_API_KEY``, ``OPENROUTER_MODEL``, temperature, top_p,
frequency_penalty, max tokens).
"""

from __future__ import annotations

import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from config import (
    LLM_FREQUENCY_PENALTY,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_MODEL,
)
from vector_store import get_embeddings

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Fixed multi-turn HR policy conversation used for every memory type.
FAKE_CONVERSATION: List[Tuple[str, str]] = [
    (
        "What is the annual leave policy?",
        "The HR policy states employees receive 12 days of annual leave per year.",
    ),
    (
        "How many sick leave days are allowed?",
        "Sick leave is separate from annual leave. "
        "Employees are entitled to 7 days of sick leave per year.",
    ),
    (
        "What about maternity leave?",
        "Maternity leave is 26 weeks as per the company HR policy.",
    ),
]

SUMMARY_QUESTION = "Summarize everything discussed so far"

CHECKLIST = (
    "Check manually: Did the summary correctly mention 12 days annual leave? "
    "Did it mention 7 days sick leave? Did it mention 26 weeks maternity leave?"
)


class OpenRouterLLMForMemoryTests(ChatOpenAI):
    """OpenRouter chat model with rough token counting for summary-buffer memory.

    OpenRouter models do not implement ``get_num_tokens_from_messages``, which
    ``ConversationSummaryBufferMemory`` needs to enforce ``max_token_limit``.
    This subclass estimates tokens as ~4 characters per token for test purposes.
    """

    def _estimate_tokens(self, text: str) -> int:
        """Return a rough token estimate for arbitrary text."""
        cleaned = " ".join(text.split())
        return max(1, len(cleaned) // 4)

    def get_num_tokens(self, text: str) -> int:
        """Estimate token count for a plain string."""
        return self._estimate_tokens(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Estimate token count for a list of chat messages."""
        total = 0
        for message in messages:
            content = message.content
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        total += self._estimate_tokens(part)
                    elif isinstance(part, dict):
                        total += self._estimate_tokens(str(part.get("text", "")))
            else:
                total += self._estimate_tokens(str(content))
        return max(1, total)


def get_openrouter_llm() -> OpenRouterLLMForMemoryTests:
    """Load OpenRouter using the same settings as the OpenRouter branch in ``rag_chain.py``."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file before running "
            "this test script."
        )
    return OpenRouterLLMForMemoryTests(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=OPENROUTER_MODEL,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        frequency_penalty=LLM_FREQUENCY_PENALTY,
        max_tokens=LLM_MAX_TOKENS,
        default_headers={
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )


def feed_conversation(memory: Any, turns: List[Tuple[str, str]]) -> None:
    """Save each (question, answer) pair into a LangChain memory object."""
    for question, answer in turns:
        memory.save_context({"input": question}, {"output": answer})


def format_memory_content(variables: Dict[str, Any]) -> str:
    """Turn memory variables into a printable string for comparison."""
    if not variables:
        return ""

    chunks: List[str] = []
    for key, value in variables.items():
        if value is None or value == "":
            continue
        if isinstance(value, list):
            rendered_items = []
            for item in value:
                if hasattr(item, "type") and hasattr(item, "content"):
                    rendered_items.append(f"{item.type}: {item.content}")
                else:
                    rendered_items.append(str(item))
            chunks.append(f"{key}:\n" + "\n".join(rendered_items))
        else:
            chunks.append(f"{key}:\n{value}")
    return "\n\n".join(chunks)


def ask_llm_with_memory(llm: BaseLanguageModel, memory_text: str) -> str:
    """Ask the LLM to summarize using the memory content as context."""
    prompt = (
        "You are a helpful assistant. Based on the conversation history below, "
        "answer the user's question.\n\n"
        f"Conversation history:\n{memory_text or '(empty)'}\n\n"
        f"Question: {SUMMARY_QUESTION}\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def print_checklist() -> None:
    """Print the manual verification checklist after each memory test."""
    print(CHECKLIST)


def run_memory_test(
    name: str,
    build_memory: Callable[[BaseLanguageModel], Any],
    load_variables: Callable[[Any], Dict[str, Any]],
    llm: BaseLanguageModel,
) -> None:
    """Feed conversation, print memory state, and run one summary LLM call."""
    print(f"\n{'=' * 10} {name} {'=' * 10}")

    memory = build_memory(llm)
    feed_conversation(memory, FAKE_CONVERSATION)

    variables = load_variables(memory)
    memory_text = format_memory_content(variables)

    print("\n--- Memory content sent to LLM ---")
    print(memory_text if memory_text else "(empty)")
    print(f"\n--- Character count (rough token-cost proxy) ---")
    print(len(memory_text))

    print("\n--- LLM summary answer ---")
    answer = ask_llm_with_memory(llm, memory_text)
    print(answer)

    print("\n--- Manual checklist ---")
    print_checklist()


def test_conversation_buffer_memory(llm: BaseLanguageModel) -> None:
    """Test full-history buffer memory."""
    from langchain_classic.memory import ConversationBufferMemory

    def build_memory(_llm: BaseLanguageModel) -> ConversationBufferMemory:
        return ConversationBufferMemory(memory_key="history", return_messages=False)

    def load_variables(memory: ConversationBufferMemory) -> Dict[str, Any]:
        return memory.load_memory_variables({})

    run_memory_test("ConversationBufferMemory", build_memory, load_variables, llm)


def test_conversation_buffer_window_memory(llm: BaseLanguageModel) -> None:
    """Test sliding-window buffer memory (k=2)."""
    from langchain_classic.memory import ConversationBufferWindowMemory

    def build_memory(_llm: BaseLanguageModel) -> ConversationBufferWindowMemory:
        return ConversationBufferWindowMemory(
            k=2,
            memory_key="history",
            return_messages=False,
        )

    def load_variables(memory: ConversationBufferWindowMemory) -> Dict[str, Any]:
        return memory.load_memory_variables({})

    run_memory_test(
        "ConversationBufferWindowMemory (k=2)",
        build_memory,
        load_variables,
        llm,
    )


def test_conversation_summary_memory(llm: BaseLanguageModel) -> None:
    """Test LLM-compressed summary memory."""
    from langchain_classic.memory import ConversationSummaryMemory

    def build_memory(test_llm: BaseLanguageModel) -> ConversationSummaryMemory:
        return ConversationSummaryMemory(llm=test_llm, memory_key="history")

    def load_variables(memory: ConversationSummaryMemory) -> Dict[str, Any]:
        return memory.load_memory_variables({})

    run_memory_test("ConversationSummaryMemory", build_memory, load_variables, llm)


def test_conversation_summary_buffer_memory(llm: BaseLanguageModel) -> None:
    """Test hybrid summary + recent-turn buffer memory."""
    from langchain_classic.memory import ConversationSummaryBufferMemory

    def build_memory(test_llm: BaseLanguageModel) -> ConversationSummaryBufferMemory:
        return ConversationSummaryBufferMemory(
            llm=test_llm,
            max_token_limit=100,
            memory_key="history",
        )

    def load_variables(memory: ConversationSummaryBufferMemory) -> Dict[str, Any]:
        return memory.load_memory_variables({})

    run_memory_test(
        "ConversationSummaryBufferMemory (max_token_limit=100)",
        build_memory,
        load_variables,
        llm,
    )


def test_conversation_entity_memory(llm: BaseLanguageModel) -> None:
    """Test entity-extraction memory."""
    from langchain_classic.memory import ConversationEntityMemory

    def build_memory(test_llm: BaseLanguageModel) -> ConversationEntityMemory:
        return ConversationEntityMemory(llm=test_llm)

    def load_variables(memory: ConversationEntityMemory) -> Dict[str, Any]:
        return memory.load_memory_variables({"input": SUMMARY_QUESTION})

    run_memory_test("ConversationEntityMemory", build_memory, load_variables, llm)


def test_vectorstore_memory(llm: BaseLanguageModel) -> None:
    """Test vector-store-backed conversational memory using project embeddings."""
    from langchain_classic.memory import VectorStoreRetrieverMemory
    from langchain_chroma import Chroma

    session_id = uuid.uuid4().hex[:8]
    persist_dir = Path("./chroma_db") / "test_memory_types" / session_id
    collection_name = f"memory_test_{session_id}"

    def build_memory(_llm: BaseLanguageModel) -> VectorStoreRetrieverMemory:
        embeddings = get_embeddings()
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="history",
            input_key="input",
        )

    def load_variables(memory: VectorStoreRetrieverMemory) -> Dict[str, Any]:
        return memory.load_memory_variables({"input": SUMMARY_QUESTION})

    try:
        run_memory_test(
            "VectorStoreRetrieverMemory (ChromaDB)",
            build_memory,
            load_variables,
            llm,
        )
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


def main() -> None:
    """Run all memory-type comparisons sequentially."""
    print("Multi-PDF ChatBot — isolated memory type comparison")
    print(f"LLM: OpenRouter ({OPENROUTER_MODEL})")
    print(f"Conversation turns: {len(FAKE_CONVERSATION)}")

    llm = get_openrouter_llm()

    tests = [
        ("ConversationBufferMemory", test_conversation_buffer_memory),
        ("ConversationBufferWindowMemory", test_conversation_buffer_window_memory),
        ("ConversationSummaryMemory", test_conversation_summary_memory),
        ("ConversationSummaryBufferMemory", test_conversation_summary_buffer_memory),
        ("ConversationEntityMemory", test_conversation_entity_memory),
        ("VectorStoreRetrieverMemory", test_vectorstore_memory),
    ]

    for label, test_fn in tests:
        try:
            test_fn(llm)
        except Exception as exc:
            print(f"\n{'=' * 10} {label} FAILED {'=' * 10}")
            print(f"Error: {exc}")
            print("Continuing with remaining memory types...\n")

    print(
        "\nRemember to pip install any missing packages with "
        "--break-system-packages if needed, and delete this test file before "
        "final submission if it's not meant to be part of the deliverable."
    )


if __name__ == "__main__":
    main()
