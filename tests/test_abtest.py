import math

import pandas as pd

from abtest.traffic_split import TrafficSplitter
from abtest.uplift_eval import UpliftEvaluator, evaluate_uplift


def test_traffic_splitter_deterministic_audit_and_category_override():
    splitter = TrafficSplitter(
        "exp-1",
        {"control": 0.4, "treatment": 0.6},
        category_allocations={"books": {"control": 0.2, "treatment": 0.8}},
        salt="v1",
    )

    asin = "B012345678"
    default_variant = splitter.assign(asin)
    # Deterministic assignments for the same asin/category pair
    assert splitter.assign(asin) == default_variant

    books_variant = splitter.assign(asin, category="books")
    assert splitter.assign(asin, category="books") == books_variant
    # Category-specific allocations are stored as expected and can produce
    # different assignments from the defaults.
    assert splitter.allocations_for("books") == {"control": 0.2, "treatment": 0.8}
    assert books_variant in {"control", "treatment"}

    audit_entries = splitter.get_audit_trail()
    assert len(audit_entries) == 4
    assert audit_entries[0].asin == asin
    assert audit_entries[0].variant == default_variant

    splitter.clear_audit_trail()
    assert splitter.get_audit_trail() == []


def test_uplift_evaluator_reports_metrics_and_lifts():
    data = pd.DataFrame(
        [
            {
                "variant": "control",
                "impressions": 1000,
                "clicks": 100,
                "conversions": 20,
                "gmv": 5000.0,
                "gmv_sq": 20 * (250.0**2),
                "spend": 1000.0,
                "roas": 20 * 250.0,
                "roas_sq": 20 * (250.0**2),
            },
            {
                "variant": "treatment",
                "impressions": 1000,
                "clicks": 130,
                "conversions": 30,
                "gmv": 8250.0,
                "gmv_sq": 30 * (275.0**2),
                "spend": 1000.0,
                "roas": 30 * 275.0,
                "roas_sq": 30 * (275.0**2),
            },
        ]
    )

    report = evaluate_uplift(data, control_variant="control")
    df = report.to_dataframe()

    # Ensure all variants and metrics are present
    assert set(df["variant"]) == {"control", "treatment"}
    expected_metrics = {"CTR", "CR", "GMV per impression", "ROAS"}
    assert set(df["metric"]) == expected_metrics

    treatment_ctr = df[(df["variant"] == "treatment") & (df["metric"] == "CTR")].iloc[0]
    expected_ctr = 130 / 1000
    assert math.isclose(treatment_ctr["value"], expected_ctr, rel_tol=1e-6)
    control_ctr = 100 / 1000
    assert math.isclose(treatment_ctr["lift_vs_control"], (expected_ctr - control_ctr) / control_ctr, rel_tol=1e-6)
    assert treatment_ctr["p_value"] is not None
    assert treatment_ctr["p_value"] < 0.05

    treatment_roas = df[(df["variant"] == "treatment") & (df["metric"] == "ROAS")].iloc[0]
    expected_roas = 8250.0 / 1000.0
    assert math.isclose(treatment_roas["value"], expected_roas, rel_tol=1e-6)
    control_roas = 5000.0 / 1000.0
    assert math.isclose(
        treatment_roas["lift_vs_control"], (expected_roas - control_roas) / control_roas, rel_tol=1e-6
    )
    assert treatment_roas["p_value"] is not None
    assert treatment_roas["p_value"] < 0.1

    markdown = report.to_markdown()
    assert "variant" in markdown

    evaluator = UpliftEvaluator(control_variant="control")
    evaluated_report = evaluator.evaluate(data)
    pd.testing.assert_frame_equal(evaluated_report.to_dataframe(), df)


def test_uplift_evaluator_requires_impressions_column():
    data = pd.DataFrame([
        {"variant": "control", "clicks": 10},
    ])

    try:
        evaluate_uplift(data, control_variant="control")
    except ValueError as exc:
        assert "impressions" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing impressions column")
