def lavenstein_distance(s1, s2):
    """
    Calculates the Levenshtein distance between two strings.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The Levenshtein distance between s1 and s2.
    """

    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    d = [[i + j for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost  # substitution
            )

    return d[len(s1)][len(s2)]