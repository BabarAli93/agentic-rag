import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "sources.db"

def _connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:

    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_uri TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )


def add_source(
        source_id: str,
        owner_id: str,
        source_name: str,
        source_type: str,
        source_uri : str  | None = None
) -> None:

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sources (
                source_id, owner_id, source_name,
                source_type, source_uri, status
            )
            VALUES (?, ?, ?, ?, ?, 'pending')
            """,
            (
                source_id, owner_id, source_name,
                source_type, source_uri,
            ),
        )


def list_sources(owner_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM sources
            WHERE owner_id = ?
            ORDER BY created_at DESC
            """,
            (owner_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_source(
        source_id: str,
        owner_id: str
) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM sources
            WHERE source_id = ? AND owner_id = ?
            """,
            (source_id, owner_id),
        ).fetchone()

    return dict(row) if row else None


def update_source(
        source_id: str,
        owner_id: str,
        *,
        status: str,
        chunk_count: int | None = None,
        error_message: str | None = None,
) -> None:
    
    with _connect() as conn:
        conn.execute(
            """
            UPDATE sources
            SET status = ?,
                chunk_count = COALESCE(?, chunk_count),
                error_message = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE source_id = ? AND owner_id = ?
            """,
            (
                status, chunk_count, error_message,
                source_id, owner_id,
            ),
        )


def delete_source(
        source_id: str, 
        owner_id: str
) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM sources
            WHERE source_id = ? AND owner_id = ?
            """,
            (source_id, owner_id),
        )

    return cursor.rowcount > 0

init_db()