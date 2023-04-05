def get_digit_subscript(i: int):
    assert i in range(10), f"Subscripts are only available for digits 0-9, but got {i}."
    return chr(0x2080 + i)