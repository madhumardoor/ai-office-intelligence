
from pathlib import Path

path = Path("app/ui.py")
text = path.read_text(encoding="utf-8")

if "def create_embedder_cached" not in text:
    needle = "def create_llm_client_cached():"
    start = text.find(needle)
    if start == -1:
        raise SystemExit(
            'create_llm_client_cached was not found. '
            'Run: findstr /n /C:"create_llm_client_cached" app\\ui.py'
        )

    next_def = text.find("\ndef ", start + len(needle))
    if next_def == -1:
        raise SystemExit("Could not locate the end of create_llm_client_cached().")

    helper = '''
@st.cache_resource
def create_embedder_cached():
    # Keep the local BGE model resident between Streamlit reruns/questions.
    # This prevents reloading the embedding model for every request.
    return create_embedder(get_settings())

'''
    text = text[:next_def] + helper + text[next_def:]

text = text.replace(
    "embedder = create_embedder(settings)",
    "embedder = create_embedder_cached()",
)

text = text.replace(
    "embedder = create_embedder(get_settings())",
    "embedder = create_embedder_cached()",
)

path.write_text(text, encoding="utf-8")

print("Patch applied successfully.")
print("Cached helper present:", "def create_embedder_cached" in text)
print("Direct create_embedder(settings) remaining:", "embedder = create_embedder(settings)" in text)
