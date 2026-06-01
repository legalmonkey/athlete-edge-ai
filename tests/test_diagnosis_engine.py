from src.diagnosis_model.hybrid import DiagnosisEngine
from src.questionnaire.schemas import DiagnosisRequest


def test_acl_like_case_ranks_acl_first():
    payload = DiagnosisRequest(
        pain_location="knee",
        onset="sudden",
        symptoms={
            "pain_severity": 8,
            "swelling": True,
            "instability": True,
            "popping_sound": True,
            "weight_bearing_ability": "difficult",
        },
        mechanism={"twisting": True, "cutting": True},
        risk_model_output={"injury_risk_probability": 0.82, "fatigue_score": 76},
    )
    response = DiagnosisEngine().diagnose(payload)
    assert response.top_predictions
    assert response.top_predictions[0].injury == "ACL Tear"
    assert response.top_predictions[0].probability > 0
    assert "Pain located in knee" in response.explanation


def test_non_matching_region_filters_knee_injuries():
    payload = DiagnosisRequest(
        pain_location="ankle",
        onset="sudden",
        symptoms={"swelling": True, "bruising": True},
        mechanism={"landing": True, "twisting": True},
    )
    response = DiagnosisEngine().diagnose(payload)
    names = [item.injury for item in response.top_predictions]
    assert "ACL Tear" not in names
    assert response.top_predictions[0].injury == "Lateral Ankle Sprain"
