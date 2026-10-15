"""Validation of secrets read from the environment; placeholders are never accepted."""

# ============= Constants =============

MIN_SECRET_LENGTH = 16
PLACEHOLDERS = frozenset({"", "change-me", "changeme", "change_me", "password", "secret", "todo"})


# ============= Validation =============


def secret_problem(value: str) -> str | None:
    """
    Describe why a secret is unusable, or return None when it is acceptable.

    Parameters
    ----------
    value : str
        Secret value from the environment.

    Returns
    -------
    problem : str or None
        `empty`, `a known placeholder`, or `shorter than 16 characters`.
    """
    stripped = value.strip()
    if not stripped:
        return "empty"
    if stripped.lower() in PLACEHOLDERS:
        return "a known placeholder"
    if len(stripped) < MIN_SECRET_LENGTH:
        return f"shorter than {MIN_SECRET_LENGTH} characters"
    return None


def require_secret(name: str, value: str) -> str:
    """
    Return the secret or raise ValueError naming the variable and the problem.

    Raises
    ------
    ValueError
        If the value is empty, a placeholder, or too short.
    """
    problem = secret_problem(value)
    if problem:
        raise ValueError(f"{name} is {problem}; run `make env` to generate strong secrets")
    return value
