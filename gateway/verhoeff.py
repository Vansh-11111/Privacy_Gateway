
# Multiplication table of the dihedral group D5
D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
# Position-dependent permutation table
P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

ASCII_DIGITS = frozenset("0123456789")

def is_ascii_digits(s: str)->bool:
    return bool(s) and all(ch in ASCII_DIGITS for ch in s)

def is_valid(number: str)->bool:
    if not is_ascii_digits(number):
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c= D[c][P[i%8][int(ch)]]
    return c==0

def check_digit(number: str)->str:
    if not is_ascii_digits(number):
        raise ValueError("Number must be a non empty string of Ascii digits")
    c = 0
    for i,ch in enumerate(reversed(number)):
        c=D[c][P[(i + 1) % 8][int(ch)]]
    return str(INV[c])