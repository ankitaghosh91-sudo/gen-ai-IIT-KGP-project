from pathlib import Path
import sqlite3


class ParentChunkStore:
    def __init__(self, db_path: str = "data/processed/parent_chunks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False avoids errors if called across threads
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_table(self) -> None:
        # `with conn:` automatically commits on success and rolls back on error
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    parent_text TEXT NOT NULL
                )
                """
            )

    def upsert_parents(self, parents: list[tuple[str, str, str]]) -> None:
        """Save or update rows of (parent_id, doc_id, parent_text)."""
        if not parents:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO parent_chunks (parent_id, doc_id, parent_text)
                VALUES (?, ?, ?)
                ON CONFLICT(parent_id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    parent_text = excluded.parent_text
                """,
                parents,
            )

    def get_parent_texts(self, parent_ids: list[str]) -> dict[str, str]:
        """Return a mapping of {parent_id: parent_text} for the given IDs."""
        unique_ids = list(dict.fromkeys(pid for pid in parent_ids if pid))
        if not unique_ids:
            return {}

        placeholders = ",".join("?" for _ in unique_ids)
        sql = f"SELECT parent_id, parent_text FROM parent_chunks WHERE parent_id IN ({placeholders})"

        with self._connect() as conn:
            rows = conn.execute(sql, unique_ids).fetchall()

        return {parent_id: parent_text for parent_id, parent_text in rows}