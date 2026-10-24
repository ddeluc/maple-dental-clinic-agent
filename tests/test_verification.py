from backend.verification import normalize_name, normalize_phone


def test_normalize_name_ignores_case_and_accents():
    assert normalize_name("Hélène Tremblay") == "helene tremblay"
    assert normalize_name("helene tremblay") == "helene tremblay"
    assert normalize_name("HÉLÈNE TREMBLAY") == "helene tremblay"


def test_normalize_name_collapses_extra_spaces():
    assert normalize_name("  Hélène   Tremblay ") == "helene tremblay"


def test_normalize_name_keeps_hyphens():
    assert normalize_name("Jean-François Côté-Lavoie") == "jean-francois cote-lavoie"


def test_normalize_phone_keeps_only_digits():
    assert normalize_phone("(514) 555-0142") == "5145550142"
    assert normalize_phone("514-555-0142") == "5145550142"
    assert normalize_phone("514.555.0142") == "5145550142"
    assert normalize_phone("514 555 0142") == "5145550142"


def test_normalize_phone_drops_country_code():
    assert normalize_phone("+1 514 555 0104") == "5145550104"
    assert normalize_phone("1-514-555-0104") == "5145550104"


def test_normalize_phone_ignores_non_ascii_digits():
    # "²" counts as a digit for str.isdigit(), but isn't part of a number.
    assert normalize_phone("514-555-0142²") == "5145550142"
