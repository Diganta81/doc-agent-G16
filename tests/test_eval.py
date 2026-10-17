"""Unit tests for the evaluation metrics."""
import json

import pytest

from doc_agent.contracts import Answer, Citation
from doc_agent.eval import ablation, metrics


def answer(text="ইহারা দ্বি-ভাজন পদ্ধতিতে প্রজনন করে।", cites=(("c0", (0, 10)),), grounded=True):
    return Answer(text=text, citations=[Citation(chunk_id=c, span=s) for c, s in cites],
                  grounded=grounded, confidence=0.9)


def test_ocr_f1_bounds():
    assert metrics.ocr_f1("ধানের রোগ", "ধানের রোগ") == 1.0
    assert metrics.ocr_f1("", "ধান") == 0.0
    assert 0 < metrics.ocr_f1("ধানের রোগ", "ধানের পাতা") < 1


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert metrics.recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert metrics.recall_at_k(["a"], ["a", "z"], 5) == 0.5
    assert metrics.recall_at_k(["a"], [], 5) == 1.0


def test_contains_gold_normalises_digits_and_punctuation():
    assert metrics.contains_gold("রোগটি ১৮৯২ সালে, প্রথম দেখা যায়।", "1892")
    assert not metrics.contains_gold("কিছু নেই", "1892")
    assert not metrics.contains_gold("anything", "")


def test_groundedness_and_citation_accuracy():
    evidence = {"c0": "ইহারা দ্বি-ভাজন পদ্ধতিতে প্রজনন করে।"}
    assert metrics.groundedness(answer()) == 1.0
    assert metrics.groundedness(answer(grounded=False)) == 0.0
    assert metrics.groundedness(answer(), evidence) == 1.0
    assert metrics.groundedness(answer(text="সম্পূর্ণ অন্য বিষয় এখানে"), evidence) < 0.5
    assert metrics.citation_accuracy(answer(), {"c0": "x" * 50}) == 1.0
    assert metrics.citation_accuracy(answer(), {"c0": "short"}) == 0.0     # span beyond text
    assert metrics.citation_accuracy(answer(cites=(("zz", (0, 3)),)), {"c0": "x" * 50}) == 0.0


def test_ece_perfect_and_overconfident():
    assert metrics.ece([1.0, 1.0], [1, 1]) == 0.0
    assert metrics.ece([0.9, 0.9, 0.9, 0.9], [1, 0, 0, 0]) == pytest.approx(0.65)


def test_subgroup_gap():
    assert metrics.subgroup_gap({"a": [0.9, 0.7], "b": 0.5}) == pytest.approx(0.3)


def _line(run, step, tool, args, obs):
    return json.dumps({"run_id": run, "step": step, "tool": tool, "args": args, "obs": obs})


def test_trace_completeness(tmp_path):
    good = [_line("a", 1, "retrieve", {"query": "q", "k": 10}, {"top_score": 0.5, "k": 10}),
            _line("a", 2, "answer", {}, {"abstained": False})]
    no_answer = [_line("b", 1, "retrieve", {"query": "q", "k": 10}, {"top_score": 0.5, "k": 10})]
    gap = [_line("c", 1, "retrieve", {"query": "q", "k": 10}, {"top_score": 0.5, "k": 10}),
           _line("c", 3, "answer", {}, {"abstained": False})]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(good + no_answer + gap) + "\n")
    result = metrics.trace_completeness(path)
    assert result["runs"] == 3 and result["complete"] == 1


def test_ablation_reports_deltas(cfg):
    def evaluate(c):
        return {"recall": 0.9 - 0.2 * (not c["retrieve"]["rerank"]) - 0.1 * (not c["agent"]["research"])}
    cfg["retrieve"]["rerank"] = True
    out = ablation.run(cfg, evaluate)
    assert out["baseline"]["recall"] == 0.9
    assert out["delta"]["no_rerank"]["recall"] == pytest.approx(-0.2)
    assert out["delta"]["no_research"]["recall"] == pytest.approx(-0.1)


def test_gold_similarity_separates_near_miss_from_absent():
    assert metrics.gold_similarity("যথা বোর্দোমিশ্চার ইত্যাদি", "বোর্দোমিশ্চার") == 1.0
    near = metrics.gold_similarity("যথা বোর্দোমিক্সচ্যার, বার্গান্ডিমিকচ্যার", "বোর্দোমিশ্চার")
    assert 0.6 < near < 1.0                          # OCR spelling variant of the right word
    assert metrics.gold_similarity("কিছুই নেই এখানে অন্য বিষয়", "বোর্দোমিশ্চার") < 0.4
    assert metrics.gold_similarity("", "x") == 0.0


def test_numeric_gold_must_not_match_inside_a_longer_number():
    """Kaggle run: gold "৮০" passed for an answer saying "৮০০ সেন্টিগ্রেড" (OCR read the degree sign as 0)."""
    assert metrics.contains_gold("৮০ ডিগ্রি সেন্টিগ্রেডে", "৮০")
    assert metrics.contains_gold("তাপমাত্রা 80° সেঃ", "৮০")
    assert not metrics.contains_gold("৮০০ সেন্টিগ্রেড উষ্ণতায়", "৮০")
    assert metrics.contains_gold("১০ এবং ৮০ ডিগ্রি", "৮০")
    assert metrics.contains_gold("প্রথম ব্যাকটেরিয়া ১৬৭৬ খ্রিস্টাব্দে", "১৬৭৬")


def test_success_check_uses_the_same_numeric_rule():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "success_check", Path(__file__).resolve().parent.parent / "grading_kit/success_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = {"verifiable": True, "gold": "৮০"}
    cited = {"grounded": True, "citations": [{"chunk_id": "c"}]}
    assert module.check(task, {**cited, "text": "৮০ ডিগ্রি"})
    assert not module.check(task, {**cited, "text": "৮০০ সেন্টিগ্রেড"})
