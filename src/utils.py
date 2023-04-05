from typing import List


def get_digit_subscript(i: int):
    assert i in range(10), f"Subscripts are only available for digits 0-9, but got {i}."
    return chr(0x2080 + i)


def all_equal(l: List):
    return l.count(l[0]) == len(l)
