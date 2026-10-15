"""Date-partitioned parquet offline store shared by the consumer sinks and the training window."""

import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ============= Constants =============

TABLES = ("flow_features", "labels", "flow_behavior")
TIMESTAMP_COLUMN = {
    "flow_features": "event_timestamp",
    "labels": "label_timestamp",
    "flow_behavior": "event_timestamp",
}


# ============= Store =============


class OfflineStore:
    """
    Append-only parquet tables partitioned by UTC date: `<root>/<dataset>/<table>/date=<d>/`.

    Each append writes new files and never rewrites existing ones, so writes are O(batch).
    The same directories are registered as Feast `FileSource` batch sources.

    Parameters
    ----------
    root : str or Path
        Offline store root (`data/offline`).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def table_path(self, dataset: str, table: str) -> Path:
        """Return the directory of one table."""
        if table not in TABLES:
            raise ValueError(f"unknown offline table {table!r}")
        return self.root / dataset / table

    def append(self, dataset: str, table: str, frame: pd.DataFrame) -> list[Path]:
        """
        Append rows, one file per date present in the frame.

        Parameters
        ----------
        dataset : str
            Dataset name.
        table : str
            Table name from TABLES.
        frame : pd.DataFrame
            Rows including the table's timestamp column (tz-aware UTC datetimes).

        Returns
        -------
        written : list of Path
            Files created by this call.
        """
        if frame.empty:
            return []
        ts_col = TIMESTAMP_COLUMN[table]
        dates = frame[ts_col].dt.strftime("%Y-%m-%d")
        written = []
        for date, part in frame.groupby(dates, sort=True):
            directory = self.table_path(dataset, table) / f"date={date}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{uuid.uuid4().hex}.parquet"
            temp = path.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pandas(part, preserve_index=False), temp)
            temp.rename(path)
            written.append(path)
        return written

    def read(
        self,
        dataset: str,
        table: str,
        columns: list[str] | None = None,
        filter_expression: ds.Expression | None = None,
    ) -> pd.DataFrame:
        """Read a table (optionally a column subset and a pyarrow filter); empty if absent."""
        path = self.table_path(dataset, table)
        files = sorted(path.rglob("*.parquet")) if path.exists() else []
        if not files:
            return pd.DataFrame(columns=columns or [])
        dataset_obj = ds.dataset([str(f) for f in files], format="parquet")
        return dataset_obj.to_table(columns=columns, filter=filter_expression).to_pandas()
