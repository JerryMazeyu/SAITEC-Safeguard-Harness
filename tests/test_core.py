from safeguard_harness.core import Decision, MethodResult, RunTrace, SafetyCase, TraceStep


def test_safety_case_from_dict_defaults_and_roundtrip():
    case = SafetyCase.from_dict({"id": "c1", "question": "hello"})

    assert case.id == "c1"
    assert case.question == "hello"
    assert case.modality == "text"
    assert case.attachments == []
    assert case.to_dict()["question"] == "hello"


def test_safety_case_from_dict_promotes_image_fields_to_attachments():
    case = SafetyCase.from_dict({"id": "img", "question": "describe", "image": "/tmp/pic.png"})

    assert case.modality == "image"
    assert case.attachments == ["/tmp/pic.png"]
    assert case.has_image() is True
    assert case.metadata["has_image"] is True


def test_method_result_serializes_evidence_and_metadata():
    result = MethodResult(
        method_id="rules",
        label="unsafe",
        unsafe_score=0.97,
        confidence=0.91,
        evidence=["matched high term"],
        metadata={"term": "demo"},
    )

    payload = result.to_dict()

    assert payload["method_id"] == "rules"
    assert payload["label"] == "unsafe"
    assert payload["unsafe_score"] == 0.97
    assert payload["evidence"] == ["matched high term"]
    assert payload["metadata"]["term"] == "demo"


def test_decision_and_trace_roundtrip_contains_step_observations():
    result = MethodResult(
        method_id="rules",
        label="unsafe",
        unsafe_score=1.0,
        confidence=0.99,
        evidence=["high risk"],
    )
    trace = RunTrace(case_id="c1")
    trace.add_step(TraceStep(step_id="s1", method_id="rules", result=result))
    trace.stop_reason = "short_circuit"
    decision = Decision(
        case_id="c1",
        label="unsafe",
        unsafe_score=1.0,
        confidence=0.99,
        reasons=["high risk"],
        trace=trace,
    )

    payload = decision.to_dict()

    assert payload["label"] == "unsafe"
    assert payload["trace"]["stop_reason"] == "short_circuit"
    assert payload["trace"]["steps"][0]["result"]["evidence"] == ["high risk"]
