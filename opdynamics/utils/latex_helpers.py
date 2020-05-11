# Helper functions to clean math text
def math_clean(_s: str):
    """Remove all '$' symbols in a string"""
    return _s.replace("$", "")


def math_fix(_s: str):
    """Keep only first and last '$' in a math expression"""
    num_dollar = 0
    first = 0
    last = 0
    for idx, c in enumerate(_s):
        if c == "$":
            num_dollar += 1
            if num_dollar == 1:
                first = idx
            last = idx
    if num_dollar > 2:
        return f"{_s[:first + 1]}{math_clean(_s[first + 1:last])}{_s[last:]}"
    elif num_dollar % 2 == 1:
        return f"${math_clean(_s)}$"
    else:
        return _s
