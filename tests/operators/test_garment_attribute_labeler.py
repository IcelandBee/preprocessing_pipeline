from preprocessing.operators.labelers.garment_attribute import normalize_annotation, CHOICES


def test_normalize_annotation_valid_single():
    raw = {
        "is_valid_garment_image": "yes",
        "garment_count": "single",
        "display_mode": "flat",
        "garment_position_type": "top",
        "garment_category": "t_shirt",
        "target_user_group": "female",
        "background_type": "white",
        "garment_completeness": "complete",
        "pattern_type": "solid",
        "image_quality": "high",
    }
    ann = normalize_annotation(raw, file_name="shirt_001.jpg")
    assert ann["file_name"] == "shirt_001.jpg"
    assert ann["is_valid_garment_image"] == "yes"
    assert ann["garment_count"] == "single"
    assert ann["garment_position_type"] == "top"
    assert ann["garment_category"] == "t_shirt"
    assert ann["pattern_type"] == "solid"


def test_normalize_annotation_invalid_garment_sets_all_na():
    raw = {
        "is_valid_garment_image": "no",
        "garment_count": "single",
        "display_mode": "flat",
    }
    ann = normalize_annotation(raw, file_name="landscape.jpg")
    assert ann["is_valid_garment_image"] == "no"
    assert ann["garment_count"] == "N/A"
    assert ann["display_mode"] == "N/A"
    assert ann["garment_position_type"] == "N/A"
    assert ann["pattern_type"] == "N/A"


def test_normalize_annotation_multiple_sets_conditional_na():
    raw = {
        "is_valid_garment_image": "yes",
        "garment_count": "multiple",
        "display_mode": "real_person",
        "garment_position_type": "top",
        "garment_category": "t_shirt",
        "target_user_group": "male",
        "background_type": "complex",
        "garment_completeness": "slightly_cut",
        "pattern_type": "striped",
        "image_quality": "acceptable",
    }
    ann = normalize_annotation(raw, file_name="multi.jpg")
    assert ann["garment_count"] == "multiple"
    assert ann["garment_position_type"] == "N/A"
    assert ann["garment_category"] == "N/A"
    assert ann["pattern_type"] == "N/A"
    assert ann["display_mode"] == "real_person"
    assert ann["target_user_group"] == "male"


def test_normalize_annotation_set_type_conditional_na():
    raw = {
        "is_valid_garment_image": "yes",
        "garment_count": "set",
        "display_mode": "mannequin",
        "garment_position_type": "top",
        "garment_category": "coat",
        "pattern_type": "plaid",
        "target_user_group": "neutral",
        "background_type": "light_solid",
        "garment_completeness": "complete",
        "image_quality": "high",
    }
    ann = normalize_annotation(raw, file_name="suit.jpg")
    assert ann["garment_count"] == "set"
    assert ann["garment_position_type"] == "N/A"
    assert ann["garment_category"] == "N/A"
    assert ann["pattern_type"] == "N/A"


def test_normalize_annotation_uncertain_count_sets_conditional_na():
    raw = {
        "is_valid_garment_image": "yes",
        "garment_count": "uncertain",
        "display_mode": "close_up",
    }
    ann = normalize_annotation(raw, file_name="partial.jpg")
    assert ann["garment_count"] == "uncertain"
    assert ann["garment_position_type"] == "N/A"
    assert ann["garment_category"] == "N/A"
    assert ann["pattern_type"] == "N/A"


def test_normalize_annotation_invalid_choice_gets_fallback():
    raw = {
        "is_valid_garment_image": "maybe",
        "garment_count": "3",
        "display_mode": "runway",
        "target_user_group": "adults",
        "background_type": "dark",
        "garment_completeness": "half",
        "image_quality": "medium",
    }
    ann = normalize_annotation(raw, file_name="weird.jpg")
    assert ann["is_valid_garment_image"] == "no"  # "maybe" not in choices, fallback "no"
    # All fields should be N/A since is_valid=no
    assert ann["garment_count"] == "N/A"
    assert ann["display_mode"] == "N/A"


def test_normalize_annotation_valid_choice_preserved():
    raw = {
        "is_valid_garment_image": "yes",
        "garment_count": "single",
        "display_mode": "hanging",
        "garment_position_type": "outerwear",
        "garment_category": "coat",
        "target_user_group": "infant",
        "background_type": "transparent",
        "garment_completeness": "severely_cut",
        "pattern_type": "denim",
        "image_quality": "low",
    }
    ann = normalize_annotation(raw, file_name="jacket.jpg")
    assert ann["display_mode"] == "hanging"
    assert ann["garment_position_type"] == "outerwear"
    assert ann["garment_category"] == "coat"
    assert ann["target_user_group"] == "infant"
    assert ann["background_type"] == "transparent"
    assert ann["garment_completeness"] == "severely_cut"
    assert ann["pattern_type"] == "denim"
    assert ann["image_quality"] == "low"