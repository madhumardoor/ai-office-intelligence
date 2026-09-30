from pathlib import Path

path = Path("app/ui.py")
text = path.read_text(encoding="utf-8")

marker = """@st.cache_resource
def create_llm_client_cached():
    return create_llm_client(get_settings())


"""

addition = """@st.cache_resource
def create_embedder_cached():
    # Keep the local BGE model resident between Streamlit reruns/questions.
    # Without this cache, ordinary company questions can reload the model.
    return create_embedder(get_settings())


"""

if "def create_embedder_cached" not in text:
    if marker not in text:
        raise SystemExit("Could not find create_llm_client_cached block in app/ui.py")
    text = text.replace(marker, marker + addition, 1)

text = text.replace(
    "embedder = create_embedder(settings)",
    "embedder = create_embedder_cached()",
)

path.write_text(text, encoding="utf-8")
print("Updated app/ui.py: cached local embedder.")
