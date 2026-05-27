from preprocessing.operators.labelers.person_attribute import normalize_annotation


def test_normalize_annotation_for_multiple_people_sets_other_fields_to_na():
    ann = normalize_annotation({"person_count": "N", "gender": "male"}, file_name="a.jpg")

    assert ann["file_name"] == "a.jpg"
    assert ann["person_count"] == "N"
    assert ann["gender"] == "N/A"
    assert ann["head_visible"] == "N/A"


def test_normalize_annotation_for_single_person_keeps_allowed_values():
    ann = normalize_annotation(
        {
            "person_count": "1",
            "gender": "female",
            "head_visible": "yes",
            "shot_type": "half_body",
            "clothes_visible": "yes",
            "holding_object": "no",
            "obvious_makeup": "yes",
            "expression": "smile",
            "hair_visible": "yes",
            "hair_color": "black",
            "facial_features_clear": "yes",
            "face_direction": "frontal",
            "hand_hold_feasible": "yes",
            "person_prominence": "close",
            "person_size_in_frame": "large",
        },
        file_name="a.jpg",
    )

    assert ann["gender"] == "female"
    assert ann["expression"] == "smile"
    assert ann["hair_color"] == "black"
