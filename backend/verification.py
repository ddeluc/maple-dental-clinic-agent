"""Turn names and phone numbers into a standard form so they can be compared.

Callers say names without accents and read phone numbers in any format, so
verification compares these normalized forms instead of the raw text.
"""

import string
import unicodedata

NORTH_AMERICAN_NUMBER_LENGTH = 10
COUNTRY_CODE = "1"


def normalize_name(name: str) -> str:
    """Lowercase a name, remove its accents, and collapse extra spaces.

    "  HÉLÈNE   Tremblay " becomes "helene tremblay".
    """
    # NFD splits an accented letter into the plain letter followed by a
    # separate "combining" accent mark ("é" becomes "e" + "´"), so dropping
    # the combining marks leaves just the plain letters.
    decomposed = unicodedata.normalize("NFD", name)
    letters = []
    for character in decomposed:
        if not unicodedata.combining(character):
            letters.append(character)
    without_accents = "".join(letters)

    # split() with no argument splits on any run of whitespace and ignores
    # leading and trailing spaces, so joining the words back with one space
    # collapses "Hélène   Tremblay" to "helene tremblay".
    words = without_accents.casefold().split()
    return " ".join(words)


def normalize_phone(phone: str) -> str:
    """Keep only the digits of a phone number, without the +1 country code.

    "(514) 555-0142", "514.555.0142" and "+1 514 555 0142" all become
    "5145550142".
    """
    digits = []
    for character in phone:
        # str.isdigit() also accepts characters like "²", so check against
        # the ten ASCII digits instead.
        if character in string.digits:
            digits.append(character)
    number = "".join(digits)

    # A number stored as "+1 514 555 0104" must still match a caller who
    # says "514 555 0104", so drop the leading country code when present.
    has_country_code = (
        len(number) == NORTH_AMERICAN_NUMBER_LENGTH + 1
        and number.startswith(COUNTRY_CODE)
    )
    if has_country_code:
        return number[1:]
    return number
