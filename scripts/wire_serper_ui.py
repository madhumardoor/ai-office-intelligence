from pathlib import Path

path = Path("app/ui.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "from app.tools.search_provider import MockSearchProvider",
    "from app.tools.search_provider import SerperSearchProvider",
)

text = text.replace(
    "provider = MockSearchProvider()",
    "provider = SerperSearchProvider()",
)

path.write_text(text, encoding="utf-8")
print("Updated app/ui.py: MockSearchProvider -> SerperSearchProvider")
