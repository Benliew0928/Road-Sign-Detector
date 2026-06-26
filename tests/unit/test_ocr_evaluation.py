from roadsign_assist.evaluation.ocr import character_error_rate, edit_distance


def test_edit_distance_and_cer() -> None:
    assert edit_distance("STOP", "STOP") == 0
    assert edit_distance("STOP", "SOP") == 1
    assert character_error_rate("Had Laju 50", "HAD LAJU 50") == 0.0
    assert abs(character_error_rate("STOP", "SOP") - 0.25) < 1e-9
