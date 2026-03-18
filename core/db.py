"""DB connection pool backed by psycopg2 ThreadedConnectionPool."""
import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.pool

from .config import Config

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_pool() -> None:
    global _pool
    if _pool is not None:
        return

    logger.info(
        "Initialising DB pool (min=%d, max=%d)",
        Config.DB_POOL_MIN,
        Config.DB_POOL_MAX,
    )
    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=Config.DB_POOL_MIN,
        maxconn=Config.DB_POOL_MAX,
        dsn=Config.PG_CONN,
    )


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def get_db():
    """Yield a connection from the pool; auto-inits if needed."""
    global _pool
    if _pool is None:
        init_pool()

    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)
