import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_engine import suggest

def test_suggest_person():
    results = suggest("person")
    assert len(results) > 0
    assert any("person walking" in r for r in results)

def test_suggest_car():
    results = suggest("car")
    assert len(results) > 0
    assert any("car driving" in r for r in results)

def test_suggest_backpack():
    results = suggest("backpack")
    assert len(results) > 0
    assert any("backpack" in r for r in results)

def test_suggest_unknown():
    results = suggest("zzzzunknownzzzz")
    assert len(results) > 0
    assert any("find" in r for r in results)

def test_suggest_empty():
    results = suggest("")
    assert len(results) == 0

def test_suggest_limit():
    results = suggest("person car red backpack enter leave")
    assert len(results) <= 6

def test_suggest_case_insensitive():
    results = suggest("PERSON")
    assert len(results) > 0
