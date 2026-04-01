from __future__ import annotations

"""Security helpers.

Untuk tujuan pembelajaran dan agar mudah dijalankan di environment minimal,
password hashing memakai PBKDF2 dari stdlib (hashlib).

Jika Anda ingin bcrypt/argon2, ganti implementasi ini dan tambahkan dependency.
"""

import base64
import hashlib
import hmac
import os


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 150_000)
    return "pbkdf2$" + base64.b64encode(salt + dk).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash.startswith("pbkdf2$"):
        return False
    raw = base64.b64decode(password_hash.split("$", 1)[1].encode("ascii"))
    salt, dk = raw[:16], raw[16:]
    cand = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 150_000)
    return hmac.compare_digest(cand, dk)
