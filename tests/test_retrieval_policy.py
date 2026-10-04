"""Version-aware retrieval must use document evidence, not expected answers."""

from copy import deepcopy
from pathlib import Path

from src.retrieval_policy import annotate_documents, prefer_current_candidates


def policy(title, version, effective, body, source):
    return {"text": f"# {title}\n> Version: {version} | Effective date: {effective}\n\n{body}",
            "metadata": {"source": source, "tenant": "example"}}


def test_explicit_replacement_and_metadata_without_input_mutation():
    documents = [
        policy("Access Controls (Version 0.7)", "0.7", "2021-02-03", "Original policy.", "access_v7.md"),
        policy("Access Controls (Version 3.1)", "3.1", "2025-06-07", "This policy supersedes version 0.7.", "access_v31.md"),
    ]
    original = deepcopy(documents)
    annotated = annotate_documents(documents)
    assert documents == original
    old, new = [document["metadata"] for document in annotated]
    assert old["is_superseded"] and not old["is_current"]
    assert old["superseded_by"] == "access_v31.md"
    assert new["is_current"] and not new["is_superseded"]
    assert old["policy_family"] == new["policy_family"] == "access controls"
    assert old["effective_date"] == "2021-02-03"
    assert old["tenant"] == "example"


def test_newer_document_does_not_imply_replacement():
    documents = [
        policy("Travel policy (Version 1)", "1", "2020-01-01", "Original terms.", "travel_v1.md"),
        policy("Travel policy (Version 2)", "2", "2025-01-01", "Additional terms.", "travel_v2.md"),
    ]
    annotated = annotate_documents(documents)
    assert all(document["metadata"]["is_current"] for document in annotated)
    assert prefer_current_candidates("What are the travel terms?", annotated) == annotated


def test_replacement_does_not_cross_policy_families():
    documents = [
        policy("Access (Version 1.0)", "1.0", "2020-01-01", "Access terms.", "access_v1.md"),
        policy("Meals (Version 1.0)", "1.0", "2020-01-01", "Meal terms.", "meals_v1.md"),
        policy("Access (Version 2.0)", "2.0", "2025-01-01", "Replaces version 1.0.", "access_v2.md"),
    ]
    annotated = annotate_documents(documents)
    assert annotated[0]["metadata"]["is_superseded"]
    assert not annotated[1]["metadata"]["is_superseded"]


def test_only_actual_latest_document_proves_replacement():
    documents = [
        policy("Access (Version 1)", "1", "2020-01-01", "Initial terms.", "access_v1.md"),
        policy("Access (Version 2)", "2", "2021-01-01", "Replaces version 1.", "access_v2.md"),
        policy("Access (Version 3)", "3", "2022-01-01", "Supplemental terms.", "access_v3.md"),
    ]
    assert all(not document["metadata"]["is_superseded"] for document in annotate_documents(documents))


def test_actual_leave_and_password_policy_families():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    sources = ["nghi_phep_nam_v2023.md", "nghi_phep_nam_v2024.md", "mat_khau_v1.md", "mat_khau_v2.md"]
    documents = [{"text": (data_dir / source).read_text(encoding="utf-8"),
                  "metadata": {"source": source}} for source in sources]
    annotated = annotate_documents(documents)
    assert [document["metadata"]["is_superseded"] for document in annotated] == [True, False, True, False]
    assert annotated[0]["metadata"]["effective_date"] == "2023-01-01"
    assert annotated[3]["metadata"]["effective_date"] == "2024-07-01"
    assert annotated[0]["metadata"]["policy_family"] == annotated[1]["metadata"]["policy_family"]
    assert annotated[2]["metadata"]["policy_family"] == annotated[3]["metadata"]["policy_family"]


def test_historical_versions_years_and_comparisons_keep_old_candidates():
    documents = [
        policy("Travel (Version 1.0)", "1.0", "2023-01-01", "Initial terms.", "travel_v2023.md"),
        policy("Travel (Version 2.0)", "2.0", "2024-01-01", "Replaces version 1.0.", "travel_v2024.md"),
        policy("Salary", "5.0", "2024-01-01", "Salary bands.", "salary.md"),
    ]
    candidates = annotate_documents(documents)
    assert prefer_current_candidates("Travel allowance?", candidates) == candidates[1:]
    for query in ["Travel năm 2023", "Travel theo phiên bản 1.0", "Travel version 1.0", "Old travel policy"]:
        preferred = prefer_current_candidates(query, candidates)
        assert preferred[0] == candidates[0]
        assert candidates[1] not in preferred
        assert candidates[2] in preferred
    assert prefer_current_candidates("Compare versions 1.0 and 2.0", candidates) == candidates
    assert prefer_current_candidates("So sánh chính sách cũ và hiện hành", candidates) == candidates


def test_source_version_fallback_and_all_prior_versions():
    documents = [
        {"text": "Original terms.", "metadata": {"source": "arbitrary_v17.4.md"}},
        {"text": "This document replaces all previous versions.", "metadata": {"source": "arbitrary_v23.8.md"}},
    ]
    old, new = annotate_documents(documents)
    assert old["metadata"]["policy_family"] == new["metadata"]["policy_family"] == "arbitrary"
    assert old["metadata"]["version"] == "17.4"
    assert old["metadata"]["is_superseded"]


def test_historical_year_from_source_without_date_header():
    documents = [
        {"text": "Old terms.", "metadata": {"source": "arbitrary_v2021.md"}},
        {"text": "Replaces version 2021.", "metadata": {"source": "arbitrary_v2025.md"}},
    ]
    candidates = annotate_documents(documents)
    assert prefer_current_candidates("What were the terms in 2021?", candidates) == candidates[:1]


def test_metadata_authority_and_unknown_candidates():
    document = policy("Access [v2.0]", "2.0", "2024-01-01", "Terms.", "access_v2.md")
    document["metadata"].update({"version": "7.0", "effective_date": "03/04/2025", "parent_id": "preserved"})
    metadata = annotate_documents([document])[0]["metadata"]
    assert metadata["version"] == "7.0"
    assert metadata["effective_date"] == "2025-04-03"
    assert metadata["parent_id"] == "preserved"
    unknown = [{"text": "Other information.", "metadata": {}}]
    assert prefer_current_candidates("Unknown historical policy from 1991", unknown) == unknown


def test_inverse_replacement_statement_is_not_used_as_forward_replacement():
    documents = [
        policy("Access (Version 1)", "1", "2020-01-01", "Initial terms.", "access_v1.md"),
        policy("Access (Version 2)", "2", "2021-01-01", "This is replaced by version 1.", "access_v2.md"),
    ]
    assert all(not document["metadata"]["is_superseded"] for document in annotate_documents(documents))


def test_negated_replacement_does_not_supersede_old_document():
    documents = [
        policy("Access (Version 1)", "1", "2020-01-01", "Initial terms.", "access_v1.md"),
        policy("Access (Version 2)", "2", "2021-01-01", "This policy does not replace version 1.", "access_v2.md"),
    ]
    assert all(not document["metadata"]["is_superseded"] for document in annotate_documents(documents))


def test_named_replacement_of_another_family_is_not_misattributed():
    documents = [
        policy("Access (Version 1)", "1", "2020-01-01", "Access terms.", "access_v1.md"),
        policy("Meals (Version 1)", "1", "2020-01-01", "Meal terms.", "meals_v1.md"),
        policy("Access (Version 2)", "2", "2025-01-01", "This document replaces Meals version 1.", "access_v2.md"),
    ]
    assert all(not document["metadata"]["is_superseded"] for document in annotate_documents(documents))
