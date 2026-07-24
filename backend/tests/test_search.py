import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_engine import suggest, _compute_iou, _extract_class_names, _extract_search_plan


class TestSuggest:
    def test_person(self):
        results = suggest("person")
        assert len(results) > 0
        assert any("person walking" in r for r in results)

    def test_car(self):
        results = suggest("car")
        assert len(results) > 0
        assert any("car driving" in r for r in results)

    def test_backpack(self):
        results = suggest("backpack")
        assert len(results) > 0
        assert any("backpack" in r for r in results)

    def test_unknown(self):
        results = suggest("zzzzunknownzzzz")
        assert len(results) > 0
        assert any("find" in r for r in results)

    def test_empty(self):
        results = suggest("")
        assert len(results) == 0

    def test_limit(self):
        results = suggest("person car red backpack enter leave")
        assert len(results) <= 6

    def test_case_insensitive(self):
        results = suggest("PERSON")
        assert len(results) > 0


class TestComputeIoU:
    def test_perfect_overlap(self):
        box = [0, 0, 10, 10]
        assert _compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        a = [0, 0, 10, 10]
        b = [20, 20, 30, 30]
        assert _compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0, 0, 10, 10]
        b = [5, 0, 15, 10]
        iou = _compute_iou(a, b)
        assert 0.3 < iou < 0.4

    def test_one_inside_other(self):
        a = [0, 0, 20, 20]
        b = [5, 5, 15, 15]
        assert _compute_iou(a, b) == 0.25

    def test_zero_area_box(self):
        a = [0, 0, 0, 0]
        b = [0, 0, 10, 10]
        assert _compute_iou(a, b) == 0.0


class TestExtractClassNames:
    def test_single_class(self):
        assert _extract_class_names("car") == ["car"]

    def test_multiple_classes(self):
        names = _extract_class_names("person with car and cat")
        assert "person" in names
        assert "car" in names
        assert "cat" in names

    def test_no_match_returns_empty_list(self):
        names = _extract_class_names("something unusual")
        assert len(names) == 0

    def test_query_with_numbers(self):
        names = _extract_class_names("find 2 cars and 3 people")
        assert "car" in names
        assert "person" not in names

    def test_partial_word_does_not_match(self):
        names = _extract_class_names("pers")
        assert "person" not in names


class TestExtractSearchPlan:
    def test_basic_query(self):
        plan = _extract_search_plan("a person walking with a dog")
        assert "person" in plan.get("objects", []) or "person" in str(plan)
        assert "dog" in plan.get("objects", []) or "dog" in str(plan)

    def test_with_location(self):
        plan = _extract_search_plan("car near the entrance")
        assert "car" in plan.get("objects", []) or "car" in str(plan)

    def test_with_attributes(self):
        plan = _extract_search_plan("red car")
        assert "car" in plan.get("objects", []) or "car" in str(plan)

    def test_empty_query(self):
        plan = _extract_search_plan("")
        assert isinstance(plan, dict)
