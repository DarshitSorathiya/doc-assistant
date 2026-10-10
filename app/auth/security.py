# ============================================================
# Password Hashing — bcrypt
# ============================================================
# Provides hash_password() and verify_password() utilities.
# bcrypt is deliberately slow to resist brute-force attacks.
# ============================================================

import bcrypt


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        plain_password: The raw password string from the user.

    Returns:
        A bcrypt hash string (starts with ``$2b$``).
    """
    pw_bytes = plain_password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Args:
        plain_password: The raw password to check.
        hashed_password: The stored hash to compare against.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    try:
        pw_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pw_bytes, hash_bytes)
    except Exception:
        return False

