import re
from typing import Optional, Tuple


def parse_likert_response(text: str) -> Tuple[Optional[int], bool]:
    if not text:
        return None, False

    matches = re.findall(r"\b([1-7])\b", text)

    if len(matches) == 1:
        return int(matches[0]), True

    return None, False
