import re


def extract_click_coordinates(text: str) -> tuple[int, int]:
    pattern = r"<click>(\d+),\s*(\d+)"
    matches = re.findall(pattern, text)
    if not matches:
        msg = f"No click coordinates found in text: {text}"
        raise ValueError(msg)
    x, y = matches[-1]
    return int(x), int(y)
