from __future__ import annotations



import asyncio

import json

import tempfile

import uuid

from pathlib import Path

from typing import Any

from uuid import UUID



import streamlit as st
from sqlalchemy import text



from app.agent.general_agent import GeneralBusinessAgent

from app.company.onboarding import CompanyOnboardingService

from app.config import get_settings

from app.database import create_engine, create_session_factory

from app.llm.factory import create_llm_client

from app.memory.redis_memory import RedisMemory

from app.rag.embedding.factory import create_embedder

from app.retrieval.hybrid import HybridRetriever

from app.tools.db.readonly import ReadOnlyDatabase

from app.tools.registry import create_tool_registry

from app.tools.search_cache import SearchCache

from app.tools.search_provider import SerperSearchProvider





st.set_page_config(

    page_title="AI Office Intelligence",

    page_icon="AI",

    layout="wide",

    initial_sidebar_state="expanded",

)





@st.cache_resource

def create_llm_client_cached():

    return create_llm_client(get_settings())





@st.cache_resource
def create_embedder_cached():
    # Keep the local BGE model resident between Streamlit reruns/questions.
    # This prevents reloading the embedding model for every request.
    return create_embedder(get_settings())


def safe_dump(value: Any) -> Any:

    if hasattr(value, "model_dump"):

        return value.model_dump(mode="json")

    if isinstance(value, dict):

        return {str(k): safe_dump(v) for k, v in value.items()}

    if isinstance(value, list):

        return [safe_dump(v) for v in value]

    return value





def get_conversation_id() -> str:

    existing = st.query_params.get("conversation_id")

    if existing and len(existing) <= 128:

        return existing



    value = str(uuid.uuid4())

    st.query_params["conversation_id"] = value

    return value





def memory_client() -> RedisMemory:

    settings = get_settings()

    return RedisMemory(

        settings.redis_url,

        ttl_seconds=settings.redis_memory_ttl_seconds,

        max_messages=settings.redis_memory_max_messages,

        max_turns=settings.redis_memory_max_turns,

    )





def build_memory_context(

    messages: list[dict[str, str]],

    turns: list[dict[str, Any]],

    meta: dict[str, str],

) -> str:

    recent = messages[-20:]

    turn_lines: list[str] = []



    for turn in turns[-10:]:

        turn_lines.append(

            f"Q: {turn.get('question', '')}\n"

            f"A: {turn.get('final_text', '')}\n"

            f"Verified: {turn.get('verified', False)}"

        )



    completed = "\n---\n".join(turn_lines) if turn_lines else "(none)"



    return (

        "Conversation metadata:\n"

        f"{json.dumps(meta, ensure_ascii=False)}\n\n"

        "Active company:\n"

        f"Company name: {meta.get('company_name', '(not set)')}\n"

        f"Company ID: {meta.get('company_id', '(not set)')}\n"

        f"Website: {meta.get('company_website', '(not set)')}\n\n"

        f"Recent conversation:\n{json.dumps(recent, ensure_ascii=False)}\n\n"

        f"Recent completed turns:\n{completed}"

    )[-20000:]





def save_uploaded_files(uploaded_files):

    temp_dir = tempfile.TemporaryDirectory(prefix="aoi_company_docs_")

    root = Path(temp_dir.name)

    paths: list[Path] = []



    for uploaded in uploaded_files:

        suffix = Path(uploaded.name).suffix.lower()

        if suffix not in {".pdf", ".txt", ".md", ".html", ".htm"}:

            temp_dir.cleanup()

            raise ValueError(

                f"Unsupported upload type: {uploaded.name}. "

                "Use PDF, TXT, MD, or HTML."

            )



        destination = root / Path(uploaded.name).name

        destination.write_bytes(uploaded.getvalue())

        paths.append(destination)



    return temp_dir, paths





async def _get_company_knowledge_stats_async(company_id: str) -> tuple[int, int]:
    """Read durable document/chunk counts directly from PostgreSQL for one company."""
    if not company_id:
        return 0, 0

    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            document_count = int(
                (await session.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM documents
                        WHERE company_id = CAST(:company_id AS UUID)
                        """
                    ),
                    {"company_id": company_id},
                )).scalar_one()
            )

            chunk_count = int(
                (await session.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM document_chunks
                        WHERE company_id = CAST(:company_id AS UUID)
                        """
                    ),
                    {"company_id": company_id},
                )).scalar_one()
            )

            return document_count, chunk_count
    finally:
        await engine.dispose()


def get_company_knowledge_stats(company_id: str) -> tuple[int, int]:
    """Synchronous Streamlit wrapper around the async PostgreSQL count query."""
    return asyncio.run(_get_company_knowledge_stats_async(company_id))


def is_out_of_business_scope(result: dict[str, Any]) -> bool:
    """Detect a business-scope rejection without depending on one result key."""
    for key in (
        "business_scope_allowed",
        "scope_allowed",
        "is_business_related",
        "in_scope",
        "allowed",
    ):
        if key in result and result.get(key) is False:
            return True

    final_text = str(result.get("final_text", "")).strip().lower()
    return (
        "designed to answer business-related questions about this company" in final_text
        or "designed to answer business related questions about this company" in final_text
        or "unrelated or personal questions" in final_text
    )

async def onboard_company_async(

    *,

    company_name: str,

    website: str | None,

    uploaded_files,

    conversation_id: str,

) -> dict[str, Any]:

    settings = get_settings()

    engine = create_engine(settings)

    session_factory = create_session_factory(engine)

    embedder = create_embedder_cached()



    temp_dir = None



    try:

        document_paths: list[Path] = []



        if uploaded_files:

            temp_dir, document_paths = save_uploaded_files(uploaded_files)



        service = CompanyOnboardingService(

            settings,

            session_factory,

            embedder,

            max_website_pages=12,

        )



        result = await service.onboard(

            company_name=company_name,

            website=website,

            document_paths=document_paths,

        )



        memory = memory_client()

        memory.ping()

        memory.set_meta(

            conversation_id,

            company_id=str(result.company_id),

            company_name=result.company_name,

            company_website=result.website or "",

            knowledge_documents=str(result.documents_created),

            knowledge_chunks=str(result.chunks_created),

            knowledge_embeddings=str(result.embeddings_embedded),

            knowledge_pending=str(result.embedding_pending),

        )

        memory.save_sources(conversation_id, [])

        memory.close()



        return {

            "company_id": str(result.company_id),

            "company_name": result.company_name,

            "website": result.website,

            "pages_discovered": result.pages_discovered,

            "documents_created": result.documents_created,

            "documents_skipped_duplicate": result.documents_skipped_duplicate,

            "chunks_created": result.chunks_created,

            "embeddings_embedded": result.embeddings_embedded,

            "embeddings_reused": result.embeddings_reused,

            "embedding_pending": result.embedding_pending,

            "failures": result.failures,

        }

    finally:

        if temp_dir is not None:

            temp_dir.cleanup()

        await engine.dispose()





def onboard_company(

    *,

    company_name: str,

    website: str | None,

    uploaded_files,

    conversation_id: str,

) -> dict[str, Any]:

    return asyncio.run(

        onboard_company_async(

            company_name=company_name,

            website=website,

            uploaded_files=uploaded_files,

            conversation_id=conversation_id,

        )

    )





async def run_agent_async(question: str) -> dict[str, Any]:

    settings = get_settings()

    engine = create_engine(settings)

    session_factory = create_session_factory(engine)

    db = ReadOnlyDatabase(settings)

    embedder = create_embedder_cached()

    retriever = HybridRetriever(session_factory, embedder)

    provider = SerperSearchProvider()

    cache = SearchCache()



    conversation_id = get_conversation_id()

    memory = memory_client()



    try:

        memory.ping()

        snapshot = memory.snapshot(conversation_id)



        company_id_raw = snapshot["meta"].get("company_id")

        if not company_id_raw:

            raise RuntimeError(

                "No active company is selected. Create or load a company in the sidebar first."

            )



        try:

            active_company_id = UUID(company_id_raw)

        except ValueError as exc:

            raise RuntimeError(

                f"Invalid active company_id in Redis: {company_id_raw}"

            ) from exc



        registry = create_tool_registry(

            db=db,

            retriever=retriever,

            provider=provider,

            cache=cache,

            active_company_id=active_company_id,

        )



        agent = GeneralBusinessAgent(create_llm_client_cached(), registry)



        return await agent.ainvoke(

            question,

            messages=snapshot["messages"],

            source_context="",

            memory_context=build_memory_context(

                snapshot["messages"],

                snapshot["turns"],

                snapshot["meta"],

            ),

        )

    finally:

        memory.close()

        await engine.dispose()





def run_agent(question: str) -> dict[str, Any]:

    return asyncio.run(run_agent_async(question))





conversation_id = get_conversation_id()



if "memory_loaded" not in st.session_state:

    memory = None

    try:

        memory = memory_client()

        memory.ping()

        snapshot = memory.snapshot(conversation_id)

        st.session_state.messages = snapshot["messages"]

        st.session_state.sources = snapshot["sources"]

        st.session_state.turns = snapshot["turns"]

        st.session_state.meta = snapshot["meta"]

        st.session_state.redis_ok = True

    except Exception as exc:

        st.session_state.messages = []

        st.session_state.sources = []

        st.session_state.turns = []

        st.session_state.meta = {}

        st.session_state.redis_ok = False

        st.session_state.redis_error = str(exc)

    finally:

        if memory is not None:

            memory.close()

    st.session_state.memory_loaded = True





active_company_id = st.session_state.meta.get("company_id", "")

active_company_name = st.session_state.meta.get("company_name", "")

active_company_website = st.session_state.meta.get("company_website", "")



with st.sidebar:

    st.title("AI Office Intelligence")

    st.caption("Company-specific AI assistant")

    st.divider()



    settings = get_settings()

    st.subheader("System")

    st.write(f"**LLM:** `{settings.llm_provider}`")

    st.write(f"**Fast:** `{settings.llm_model_fast}`")

    st.write(f"**Strong:** `{settings.llm_model_strong}`")

    st.write("**Database:** `agent_ro` â€” Read-only")

    st.write(f"**Embedding:** `{settings.embedding_model}`")



    if st.session_state.redis_ok:

        st.success("Redis memory: connected")

    else:

        st.error("Redis memory: unavailable")

        st.caption(

            st.session_state.get("redis_error", "Unknown Redis error")

        )



    st.caption(f"Conversation: `{conversation_id[:12]}...`")

    st.divider()



    st.subheader("Company AI")



    company_name_input = st.text_input(

        "Company name",

        value=active_company_name,

        placeholder="Login Realty",

    )

    website_input = st.text_input(

        "Website URL (optional)",

        value=active_company_website,

        placeholder="https://example.com",

    )

    uploaded = st.file_uploader(

        "Add company documents (optional)",

        type=["pdf", "txt", "md", "html", "htm"],

        accept_multiple_files=True,

    )



    if st.button("Create / Update Company AI", use_container_width=True):

        if not company_name_input.strip():

            st.error("Company name is required.")

        else:

            with st.spinner("Building company knowledge..."):

                try:

                    onboarding_result = onboard_company(

                        company_name=company_name_input.strip(),

                        website=website_input.strip() or None,

                        uploaded_files=uploaded,

                        conversation_id=conversation_id,

                    )

                    memory = memory_client()

                    memory.ping()

                    st.session_state.meta = memory.get_meta(conversation_id)

                    memory.close()



                    if onboarding_result["failures"]:

                        st.warning(

                            "Company saved, but some sources failed. "

                            f"Failures: {len(onboarding_result['failures'])}"

                        )

                        for failure in onboarding_result["failures"][:10]:

                            st.caption(failure)

                    else:

                        st.success("Company AI knowledge is ready.")



                    st.rerun()

                except Exception as exc:

                    st.error(

                        f"Company onboarding failed: {type(exc).__name__}: {exc}"

                    )



    if active_company_id:

        st.success(f"Active company: {active_company_name}")

        st.caption(f"Company ID: `{active_company_id}`")

        if active_company_website:

            st.caption(f"Website: {active_company_website}")



        document_count, chunk_count = get_company_knowledge_stats(active_company_id)

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Documents", document_count)

        with c2:
            st.metric("Chunks", chunk_count)

        st.caption(
            "Durable company knowledge is stored in PostgreSQL + pgvector. "
            "Redis stores conversation memory and company selection."
        )

    else:

        st.info("Create a company to start a company-specific chatbot.")



    st.divider()



    if st.button("New conversation", use_container_width=True):

        old_meta = dict(st.session_state.meta)

        memory = memory_client()



        try:

            memory.clear(conversation_id)

            new_id = str(uuid.uuid4())

            st.query_params["conversation_id"] = new_id



            company_keys = {

                key: old_meta[key]

                for key in (

                    "company_id",

                    "company_name",

                    "company_website",

                    "knowledge_documents",

                    "knowledge_chunks",

                    "knowledge_embeddings",

                    "knowledge_pending",

                )

                if key in old_meta

            }

            if company_keys:

                memory.set_meta(new_id, **company_keys)

        finally:

            memory.close()



        st.session_state.clear()

        st.rerun()





st.title(

    f"{active_company_name} AI"

    if active_company_name

    else "Company AI"

)



if active_company_name:

    st.caption(

        "Company-specific assistant grounded in indexed company knowledge, "

        "with PostgreSQL/pgvector retrieval and persistent Redis memory."

    )

else:

    st.warning(

        "Create or load a company from the sidebar before asking questions."

    )



for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



question = st.chat_input(

    "Ask about the company..."

    if active_company_name

    else "Create a company first..."

)



if question:

    question = question.strip()

    if not question:

        st.stop()



    if not active_company_id:

        st.warning("Create or load a company first.")

        st.stop()



    st.session_state.messages.append(

        {"role": "user", "content": question}

    )



    try:

        memory = memory_client()

        memory.ping()

        memory.append_message(conversation_id, "user", question)

        memory.close()

    except Exception as exc:

        st.warning(f"Redis memory write failed: {exc}")



    with st.chat_message("user"):

        st.markdown(question)



    with st.chat_message("assistant"):

        progress = st.status(

            "Thinking with company knowledge...",

            expanded=False,

        )



        try:

            result = run_agent(question)

            progress.update(

                label="Completed",

                state="complete",

                expanded=False,

            )

        except Exception as exc:

            progress.update(

                label="Agent failed",

                state="error",

                expanded=True,

            )

            st.error(f"{type(exc).__name__}: {exc}")

            with st.expander("Technical error"):

                st.exception(exc)

            st.stop()



        final_text = result.get("final_text", "")
        out_of_scope = is_out_of_business_scope(result)

        if final_text:
            st.markdown(final_text)
        else:
            st.warning("No answer was generated.")

        # A scope rejection is a terminal response. Do not show evidence,
        # verification, tools, or planner details for rejected questions.
        if not out_of_scope:
            if result.get("verified"):
                st.success("Verified against retrieved evidence")
            elif result.get("verification_errors"):
                st.warning("Some claims need more evidence or review.")

            with st.expander("Sources / evidence", expanded=False):
                evidence_context = result.get("evidence_context", "")
                if evidence_context:
                    st.code(evidence_context, language="text")
                else:
                    st.info("No tool evidence was required.")

            with st.expander("Tools used", expanded=False):
                st.json(safe_dump(result.get("tool_results", {})))

            with st.expander("Agent plan", expanded=False):
                st.json(safe_dump(result.get("plan_history", [])))


    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": final_text,

            "result": result,

        }

    )



    try:

        memory = memory_client()

        memory.ping()

        memory.append_message(conversation_id, "assistant", final_text)

        memory.append_turn(

            conversation_id,

            question=question,

            final_text=final_text,

            result=result,

        )

        memory.close()

    except Exception as exc:

        st.warning(

            f"Redis memory write failed after response: {exc}"

        )

