"""JWT auth, password hashing, and user CRUD."""
import hashlib
import logging
import secrets
import psycopg2
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import Config
from .db import get_db

logger = logging.getLogger("rag.auth")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def hash_password(password: str) -> str:
    # bcrypt has a 72-byte limit
    return pwd_context.hash(password[:72])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password[:72], hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def generate_api_key() -> str:
    return secrets.token_hex(32)


def create_user(email: str, password: str) -> Dict[str, Any]:
    hashed_password = hash_password(password)

    with get_db() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO users (email, password_hash)
                    VALUES (%s, %s)
                    RETURNING id, email, created_at
                """, (email, hashed_password))

                result = cur.fetchone()
                conn.commit()

                return {
                    "id": result[0],
                    "email": result[1],
                    "created_at": result[2]
                }
            except psycopg2.IntegrityError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, password_hash, created_at
                FROM users
                WHERE email = %s
            """, (email,))

            result = cur.fetchone()

            if not result:
                return None

            user_id, user_email, password_hash, created_at = result

            if not verify_password(password, password_hash):
                return None
            cur.execute("""
                UPDATE users
                SET last_login_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (user_id,))
            conn.commit()

            return {
                "id": user_id,
                "email": user_email,
                "created_at": created_at
            }


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, created_at, last_login_at
                FROM users
                WHERE id = %s
            """, (user_id,))

            result = cur.fetchone()

            if not result:
                return None

            return {
                "id": result[0],
                "email": result[1],
                "created_at": result[2],
                "last_login_at": result[3]
            }


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, created_at, last_login_at
                FROM users
                WHERE email = %s
            """, (email,))

            result = cur.fetchone()

            if not result:
                return None

            return {
                "id": result[0],
                "email": result[1],
                "created_at": result[2],
                "last_login_at": result[3]
            }


def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))) -> Dict[str, Any]:
    """FastAPI dependency: auth via Bearer header or auth_token cookie."""
    token = None
    if credentials and getattr(credentials, "credentials", None):
        token = credentials.credentials
    else:
        token = request.cookies.get("auth_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)

    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_user_optional(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict[str, Any]]:
    """Returns None if unauthenticated instead of raising."""
    try:
        return get_current_user(request, credentials)
    except HTTPException:
        return None


# --- Password Reset ---

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cleanup_expired_tokens(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM password_resets WHERE expires_at < NOW() OR used = TRUE"
        )
    conn.commit()


def generate_reset_token(email: str) -> Optional[str]:
    """Returns None if email not found (caller should still return 200)."""
    user = get_user_by_email(email)
    if user is None:
        return None

    raw_token = secrets.token_urlsafe(48)
    token_hash = _hash_token(raw_token)
    expires_at = datetime.utcnow() + timedelta(hours=1)

    with get_db() as conn:
        _cleanup_expired_tokens(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO password_resets (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
                """,
                (user["id"], token_hash, expires_at),
            )
        conn.commit()

    return raw_token


def reset_password(token: str, new_password: str) -> bool:
    """Single-use token validation + password update."""
    token_hash = _hash_token(token)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id
                FROM password_resets
                WHERE token_hash = %s
                  AND used = FALSE
                  AND expires_at > NOW()
                """,
                (token_hash,),
            )
            row = cur.fetchone()

            if not row:
                return False

            reset_id, user_id = row

            new_hash = hash_password(new_password)

            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (new_hash, user_id),
            )

            cur.execute(
                "UPDATE password_resets SET used = TRUE WHERE id = %s",
                (reset_id,),
            )
        conn.commit()

    return True
