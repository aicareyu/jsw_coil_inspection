from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class InspectionHistory:
    def __init__(self, db_path: str | Path, max_rows: int = 200):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rows = int(max_rows)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    result TEXT NOT NULL,
                    defect_name TEXT,
                    confidence REAL,
                    processing_time_ms REAL,
                    defect_count INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()

    def add(
        self,
        image_name: str,
        result: str,
        defect_name: str | None = None,
        confidence: float | None = None,
        processing_time_ms: float | None = None,
        defect_count: int = 0,
    ) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO inspections (
                    timestamp, image_name, result, defect_name,
                    confidence, processing_time_ms, defect_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stamp,
                    image_name,
                    result,
                    defect_name,
                    confidence,
                    processing_time_ms,
                    defect_count,
                ),
            )
            conn.execute(
                """
                DELETE FROM inspections
                WHERE id NOT IN (
                    SELECT id FROM inspections ORDER BY id DESC LIMIT ?
                )
                """,
                (self.max_rows,),
            )
            conn.commit()

    def add_from_result(self, image_name: str, payload: dict[str, Any]) -> None:
        defects = payload.get("defects") or []
        top = defects[0] if defects else None
        self.add(
            image_name=image_name,
            result=str(payload.get("status", "PASS")),
            defect_name=top.get("display_name") if top else None,
            confidence=float(top["confidence"]) if top else None,
            processing_time_ms=float(payload.get("processing_time_ms") or 0.0),
            defect_count=len(defects),
        )

    def list_recent(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, image_name, result, defect_name,
                       confidence, processing_time_ms, defect_count
                FROM inspections
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM inspections")
            conn.commit()
