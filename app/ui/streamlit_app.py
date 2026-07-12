from __future__ import annotations

import json
import os
import requests
from dataclasses import asdict
from typing import Any

import streamlit as st
import pandas as pd
import altair as alt

from app.services.ingestion.chunkers import ChunkResult, get_chunker
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.parsers import SUPPORTED_EXTENSIONS, WebScraper, parse_document


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _chunks_to_rows(chunks: list[ChunkResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in chunks:
        rows.append(
            {
                "index": c.index,
                "tokens": _count_tokens(c.text),
                "chars": len(c.text),
                "metadata": json.dumps(c.metadata, ensure_ascii=False),
                "preview": (c.text[:200] + "…") if len(c.text) > 200 else c.text,
            }
        )
    return rows


def _build_chunker(strategy: str) -> tuple[object, dict[str, Any]]:
    params: dict[str, Any] = {}

    if strategy in {"fixed", "recursive"}:
        params["chunk_size"] = st.sidebar.number_input(
            "Chunk size (tokens)", min_value=50, max_value=4000, value=1000, step=50
        )
        params["overlap"] = st.sidebar.number_input(
            "Overlap (tokens)", min_value=0, max_value=1000, value=100, step=10
        )
    elif strategy == "semantic":
        if not os.getenv("OPENAI_API_KEY"):
            st.sidebar.warning(
                "OPENAI_API_KEY not set. Semantic chunker will fall back to zero embeddings.",
                icon="⚠️",
            )
        params["model"] = st.sidebar.text_input("Embedding model", value="text-embedding-3-small")
        params["threshold"] = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.75, 0.01)
        params["min_chunk_tokens"] = st.sidebar.number_input(
            "Min chunk tokens", min_value=10, max_value=2000, value=100, step=10
        )
        params["max_chunk_tokens"] = st.sidebar.number_input(
            "Max chunk tokens", min_value=50, max_value=4000, value=1000, step=50
        )
    elif strategy == "hierarchical":
        params["parent_size"] = st.sidebar.number_input(
            "Parent size (tokens)", min_value=200, max_value=8000, value=2000, step=100
        )
        params["child_size"] = st.sidebar.number_input(
            "Child size (tokens)", min_value=50, max_value=4000, value=500, step=50
        )
        params["overlap"] = st.sidebar.number_input(
            "Overlap (tokens)", min_value=0, max_value=1000, value=50, step=10
        )

    return get_chunker(strategy, **params), params


@st.cache_data(ttl=5)
def fetch_datasets(api_url: str):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/api/v1/datasets", timeout=10)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception:
        return []


def main() -> None:
    st.set_page_config(page_title="PyRAG Dev Console", layout="wide", page_icon="⚙️")
    
    st.title("⚙️ PyRAG Developer Console")
    st.markdown("A comprehensive dashboard to test **PyRAG** backend features: Ingestion, Search, Chat, Agents, and Analytics.")

    # --- SIDEBAR CONFIG ---
    st.sidebar.header("🔌 Connection Settings")
    default_api = os.getenv("PYRAG_API_URL", "http://api:8000" if os.path.exists("/.dockerenv") else "http://localhost:8000")
    api_url = st.sidebar.text_input("API Base URL", value=default_api)
    
    st.sidebar.divider()
    
    st.sidebar.header("🗂️ Active Dataset")
    datasets = fetch_datasets(api_url)
    dataset_options = {d["id"]: d["name"] for d in datasets}
    
    selected_dataset_id = None
    if dataset_options:
        selected_dataset_name = st.sidebar.selectbox("Select a Dataset", options=list(dataset_options.values()))
        # reverse lookup
        for d_id, d_name in dataset_options.items():
            if d_name == selected_dataset_name:
                selected_dataset_id = d_id
                break
    else:
        st.sidebar.warning("No datasets found. Create one in the first tab or check if backend is running.")

    # --- TABS ---
    tab_manage, tab_search, tab_chat, tab_agent, tab_analytics, tab_logs, tab_local_chunk = st.tabs([
        "🗂️ Manage Datasets", 
        "🔎 Search (Retrieval)", 
        "💬 Standard RAG", 
        "🤖 Agentic RAG",
        "📈 Analytics",
        "📜 System Logs",
        "⚙️ Local Chunk Tester"
    ])

    # =========================================================================
    # TAB 1: Manage Datasets & Docs
    # =========================================================================
    with tab_manage:
        st.header("🗂️ Manage Datasets")
        with st.expander("➕ Create New Dataset", expanded=False):
            with st.form("create_dataset_form"):
                new_ds_name = st.text_input("Dataset Name", placeholder="e.g. documentation-v1")
                new_ds_desc = st.text_area("Description", placeholder="Enter dataset description...")
                new_ds_chunk = st.selectbox("Chunk Strategy", ["recursive", "fixed", "semantic", "hierarchical"])
                submit_create_ds = st.form_submit_button("Create Dataset")
                
                if submit_create_ds and new_ds_name:
                    try:
                        r = requests.post(f"{api_url.rstrip('/')}/api/v1/datasets", json={
                            "name": new_ds_name,
                            "description": new_ds_desc,
                            "chunk_strategy": new_ds_chunk,
                            "embedding_dimensions": 1536
                        }, timeout=30)
                        r.raise_for_status()
                        st.success("Dataset created! It will appear in the sidebar shortly.")
                        fetch_datasets.clear() # clear cache
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 409:
                            st.error(f"A dataset named '{new_ds_name}' already exists. Please choose a different name.")
                        else:
                            st.error(f"Error creating dataset: {e}")
                    except Exception as e:
                        st.error(f"Error creating dataset: {e}")

        st.divider()
        st.header("📄 Documents & Ingestion Status")
        if not selected_dataset_id:
            st.info("Please select or create a dataset first.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                with st.form("upload_doc_form", clear_on_submit=True):
                    st.write(f"Upload to dataset: **{dataset_options.get(selected_dataset_id)}**")
                    up_file = st.file_uploader("Choose a file", type=list(SUPPORTED_EXTENSIONS))
                    submit_upload = st.form_submit_button("Upload & Ingest")
                    if submit_upload and up_file:
                        try:
                            files = {'file': (up_file.name, up_file.getvalue(), up_file.type)}
                            data = {'dataset_id': selected_dataset_id}
                            r = requests.post(f"{api_url.rstrip('/')}/api/v1/documents/upload", files=files, data=data, timeout=120)
                            r.raise_for_status()
                            st.success("Document uploaded and ingestion job queued!")
                        except Exception as e:
                            st.error(f"Error uploading document: {e}")
            with c2:
                # Fetch existing documents
                st.subheader("Current Documents")
                try:
                    r_docs = requests.get(f"{api_url.rstrip('/')}/api/v1/documents", params={"dataset_id": selected_dataset_id}, timeout=10)
                    if r_docs.status_code == 200:
                        docs_data = r_docs.json().get("items", [])
                        if docs_data:
                            docs_df = pd.DataFrame(docs_data)
                            if 'original_name' in docs_df.columns:
                                display_df = docs_df[['original_name', 'status', 'file_size']]
                                st.dataframe(display_df, use_container_width=True, hide_index=True)
                                
                                if any(d.get("status") == "pending" for d in docs_data):
                                    st.info("🔄 Some documents are still pending ingestion. Refresh the page to check status.")
                                elif all(d.get("status") == "success" for d in docs_data):
                                    st.success("✅ All documents successfully ingested and ready for search/chat!")
                        else:
                            st.write("No documents in this dataset yet.")
                except Exception:
                    st.warning("Could not fetch documents for this dataset.")

    # =========================================================================
    # TAB 2: Search (Retrieval)
    # =========================================================================
    with tab_search:
        st.header("🔎 Test Retrieval Engine")
        st.caption("Test the raw vector database chunk retrieval without LLM generation.")
        if not selected_dataset_id:
            st.warning("Please select a dataset from the sidebar.")
        else:
            query = st.text_input("Search Query")
            c1, c2, c3 = st.columns(3)
            search_mode = c1.selectbox("Search Mode", ["hybrid", "standard"], index=0)
            top_k = c2.number_input("Top K Results", min_value=1, max_value=50, value=5)
            rerank = c3.checkbox("Enable Reranker", value=True)
            
            if st.button("Search") and query:
                with st.spinner("Searching vector database..."):
                    try:
                        r = requests.post(f"{api_url.rstrip('/')}/api/v1/search", json={
                            "dataset_id": selected_dataset_id,
                            "query": query,
                            "mode": search_mode,
                            "top_k": int(top_k),
                            "rerank": rerank
                        }, timeout=60)
                        r.raise_for_status()
                        data = r.json()
                        results = data.get("results", [])
                        st.success(f"Found {len(results)} chunks in {data.get('latency_ms', 0)}ms")
                        
                        for i, res in enumerate(results):
                            with st.expander(f"#{i+1} | Score: {res['score']:.4f} | Chunk ID: {res['chunk_id']}"):
                                st.code(res['text'], language="text")
                                st.json(res['metadata'])
                    except Exception as e:
                        st.error(f"Search failed: {e}")

    # =========================================================================
    # TAB 3: Chat (Standard RAG)
    # =========================================================================
    with tab_chat:
        st.header("💬 Standard RAG Chat")
        st.caption("Chat with your dataset using standard RAG (Retrieve -> Generate).")
        if not selected_dataset_id:
            st.warning("Please select a dataset from the sidebar.")
        else:
            with st.expander("⚙️ Chat Configuration"):
                cc1, cc2 = st.columns(2)
                chat_provider = cc1.selectbox("LLM Provider", ["openai", "anthropic", "google"])
                chat_model = cc2.text_input("LLM Model", value="gpt-4o")
                
                cc3, cc4 = st.columns(2)
                chat_temp = cc3.slider("Temperature", 0.0, 2.0, 0.1, step=0.1)
                chat_mode = cc4.selectbox("Retrieval Mode", ["hybrid", "standard"], index=0)
                
            if "chat_msgs" not in st.session_state:
                st.session_state.chat_msgs = []
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.chat_msgs = []
                st.rerun()
                
            for m in st.session_state.chat_msgs:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if "sources" in m and m["sources"]:
                        with st.expander(f"View {len(m['sources'])} Sources"):
                            st.json(m["sources"])
                    
            if prompt := st.chat_input("Ask a question about the dataset (Standard RAG)..."):
                st.session_state.chat_msgs.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_msgs[:-1]]
                        try:
                            r = requests.post(f"{api_url.rstrip('/')}/api/v1/chat", json={
                                "dataset_id": selected_dataset_id,
                                "message": prompt,
                                "conversation_history": history,
                                "provider": chat_provider,
                                "model": chat_model,
                                "temperature": chat_temp,
                                "mode": chat_mode,
                                "top_k": 5
                            }, timeout=120)
                            r.raise_for_status()
                            data = r.json()
                            ans = data.get("answer", "No answer returned.")
                            st.markdown(ans)
                            
                            sources = data.get("sources", [])
                            if sources:
                                with st.expander(f"View {len(sources)} Sources"):
                                    st.json(sources)
                                    
                            st.session_state.chat_msgs.append({"role": "assistant", "content": ans, "sources": sources})
                        except Exception as e:
                            st.error(f"Chat failed: {e}")

    # =========================================================================
    # TAB 4: Agent (Agentic RAG)
    # =========================================================================
    with tab_agent:
        st.header("🤖 Agentic RAG Chat")
        st.caption("Chat with a LangGraph ReAct agent capable of multi-step reasoning.")
        if not selected_dataset_id:
            st.warning("Please select a dataset from the sidebar.")
        else:
            with st.expander("⚙️ Agent Configuration"):
                ac1, ac2 = st.columns(2)
                agent_provider = ac1.selectbox("Agent Provider", ["openai", "anthropic", "google"], key="agent_provider")
                agent_model = ac2.text_input("Agent Model", value="gpt-4o", key="agent_model")
            
            if "agent_msgs" not in st.session_state:
                st.session_state.agent_msgs = []
                
            if st.button("Clear Agent Chat"):
                st.session_state.agent_msgs = []
                st.rerun()
                
            for m in st.session_state.agent_msgs:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if "steps" in m and m["steps"]:
                        with st.expander(f"View Reasoning Trace ({len(m['steps'])} steps)"):
                            for s in m["steps"]:
                                st.markdown(f"**Step {s['step']} | Tool: `{s['tool']}`**")
                                st.code(f"Input: {s['input']}", language="json")
                                st.code(f"Output: {s['output']}", language="text")
                                st.divider()

            if prompt_agent := st.chat_input("Ask a complex question for the Agent..."):
                st.session_state.agent_msgs.append({"role": "user", "content": prompt_agent})
                with st.chat_message("user"):
                    st.markdown(prompt_agent)
                
                with st.chat_message("assistant"):
                    with st.spinner("Agent is reasoning (this may take a while)..."):
                        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.agent_msgs[:-1]]
                        try:
                            r = requests.post(f"{api_url.rstrip('/')}/api/v1/agents/chat", json={
                                "dataset_id": selected_dataset_id,
                                "message": prompt_agent,
                                "conversation_history": history,
                                "provider": agent_provider,
                                "model": agent_model,
                                "max_iterations": 5
                            }, timeout=300)
                            r.raise_for_status()
                            resp = r.json()
                            ans = resp.get("answer", "No answer returned.")
                            st.markdown(ans)
                            
                            steps = resp.get("steps", [])
                            if steps:
                                with st.expander(f"View Reasoning Trace ({len(steps)} steps)"):
                                    for s in steps:
                                        st.markdown(f"**Step {s['step']} | Tool: `{s['tool']}`**")
                                        st.code(f"Input: {s['input']}", language="json")
                                        st.code(f"Output: {s['output']}", language="text")
                                        st.divider()

                            st.session_state.agent_msgs.append({"role": "assistant", "content": ans, "steps": steps})
                        except Exception as e:
                            st.error(f"Agent chat failed: {e}")

    # =========================================================================
    # TAB 5: Analytics
    # =========================================================================
    with tab_analytics:
        st.header("📈 API Analytics")
        if st.button("Refresh Analytics"):
            with st.spinner("Fetching usage data..."):
                try:
                    r = requests.get(f"{api_url.rstrip('/')}/api/v1/analytics/summary", timeout=30)
                    r.raise_for_status()
                    stats = r.json()
                    
                    st.subheader("Global Metrics")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Requests", stats.get("total_requests", 0))
                    c2.metric("Total Tokens", stats.get("total_tokens", 0))
                    c3.metric("Total Cost", f"${stats.get('total_cost_usd', 0.0):.4f}")
                    c4.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.0f}ms")
                    
                    st.divider()
                    st.subheader("Cost by Provider")
                    if stats.get("cost_by_provider"):
                        st.json(stats.get("cost_by_provider"))
                    else:
                        st.info("No provider cost data available.")
                        
                except Exception as e:
                    st.error(f"Failed to fetch analytics: {e}")

    # =========================================================================
    # TAB 6: System Logs
    # =========================================================================
    with tab_logs:
        st.header("📜 System Logs")
        st.caption("View centralized logs from the API and Celery workers.")
        
        c1, c2 = st.columns([1, 5])
        num_lines = c1.number_input("Lines to fetch", min_value=10, max_value=5000, value=100, step=50)
        
        if st.button("🔄 Refresh Logs"):
            log_path = os.path.join("logs", "pyrag.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        last_lines = lines[-int(num_lines):]
                        log_text = "".join(last_lines)
                        st.code(log_text, language="text")
                except Exception as e:
                    st.error(f"Error reading logs: {e}")
            else:
                st.info(f"Log file not found at `{log_path}`. The backend might not have written any logs yet, or you need to restart the containers for the new logging config to apply.")

    # =========================================================================
    # TAB 7: Local Chunk Tester (Preserved Original App)
    # =========================================================================
    with tab_local_chunk:
        st.header("⚙️ Local Chunk Tester")
        st.caption("Test chunking and parsing locally without communicating with the backend DB.")
        
        # Local configuration specific to this tab
        st.subheader("Local Configuration")
        strategy = st.selectbox("Strategy", ["recursive", "fixed", "semantic", "hierarchical"], index=0, key="local_strategy")
        run_cleaning = st.checkbox("Run clean_text() before chunking", value=True, key="local_clean")
        
        chunker, chunker_params = _build_chunker(strategy)

        lt_input, lt_results, lt_analysis = st.tabs(["📥 Source", "🧩 Chunks", "📊 Analysis"])

        raw_text = ""
        parsed_meta: dict[str, Any] = {}

        with lt_input:
            source = st.radio("Select Input Source:", ["Paste text", "Upload file", "Scrape URL"], index=0, horizontal=True)

            if source == "Paste text":
                raw_text = st.text_area(
                    "Text",
                    height=300,
                    value=(
                        "Paste text here (or upload a file). This UI runs the same ingestion building blocks used by the "
                        "pipeline: parse → clean → chunk.\n\n"
                        "Try switching chunking strategies above."
                    ),
                )
            elif source == "Upload file":
                st.info(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
                up = st.file_uploader("Document (Local test only)", type=sorted(SUPPORTED_EXTENSIONS))
                if up is not None:
                    with st.spinner("Parsing document locally..."):
                        try:
                            parsed = parse_document(up.getvalue(), up.name)
                            raw_text = parsed.text
                            parsed_meta = parsed.metadata
                        except Exception as exc:
                            st.error("Failed to parse document.")
                            st.exception(exc)
            else:
                url = st.text_input("URL", placeholder="https://example.com/some-article")
                timeout = st.number_input("Timeout (sec)", min_value=5, max_value=120, value=30, step=5)
                if url:
                    with st.spinner("Scraping URL..."):
                        try:
                            parsed = WebScraper(timeout=int(timeout)).scrape(url)
                            raw_text = parsed.text
                            parsed_meta = parsed.metadata
                        except Exception as exc:
                            st.error("Failed to scrape URL.")
                            st.exception(exc)
            
            if parsed_meta:
                with st.expander("View Parsed Metadata", expanded=False):
                    st.json(parsed_meta)

        if not raw_text.strip():
            with lt_results:
                st.info("👈 Please provide input text to see chunking results.")
            with lt_analysis:
                st.info("👈 Please provide input text to view analysis.")
            return

        cleaned = clean_text(raw_text) if run_cleaning else raw_text

        with st.spinner(f"Chunking with {strategy}..."):
            try:
                chunks = chunker.chunk(cleaned)  # type: ignore[attr-defined]
            except Exception as exc:
                with lt_results:
                    st.error("Chunking failed.")
                    st.exception(exc)
                return

        rows = _chunks_to_rows(chunks)

        with lt_results:
            st.subheader("Summary Metrics")
            cols = st.columns(4)
            cols[0].metric("Raw Chars", len(raw_text))
            cols[1].metric("Cleaned Chars", len(cleaned))
            cols[2].metric("Raw Tokens (est.)", _count_tokens(raw_text))
            cols[3].metric("Cleaned Tokens (est.)", _count_tokens(cleaned))
            
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"Generated {len(chunks)} Chunk(s)")
            with col2:
                export = {
                    "strategy": strategy,
                    "params": chunker_params,
                    "parsed_metadata": parsed_meta,
                    "chunks": [asdict(c) for c in chunks],
                }
                st.download_button(
                    "📥 Download JSON",
                    data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="chunks.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.dataframe(rows, use_container_width=True, hide_index=True)
            
            st.subheader("Chunk Previews")
            if len(chunks) > 200:
                st.warning("Preview limited to first 200 chunks for performance.")
                
            for c in chunks[:200]:
                title = f"📦 Chunk #{c.index} | { _count_tokens(c.text) } tokens | {len(c.text)} chars"
                with st.expander(title, expanded=False):
                    st.code(c.text, language="text")
                    if c.metadata:
                        st.json(c.metadata)

        with lt_analysis:
            st.subheader("📊 Chunk Size Distribution")
            if not chunks:
                st.info("No chunks generated.")
            else:
                try:
                    df = pd.DataFrame(rows)
                    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color='#4CAF50').encode(
                        alt.X("tokens:Q", bin=alt.Bin(maxbins=20), title="Tokens per Chunk"),
                        alt.Y('count():Q', title="Number of Chunks"),
                        tooltip=["count()", alt.Tooltip("tokens:Q", bin=True, title="Tokens")]
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to render chart: {e}")
                    
                st.divider()
                st.subheader("⚙️ Chunk Config JSON")
                st.json({"strategy": strategy, **chunker_params})


if __name__ == "__main__":
    main()
