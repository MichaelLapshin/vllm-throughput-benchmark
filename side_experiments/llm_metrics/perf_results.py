from typing import List, Optional
from dataclasses import dataclass
import json

@dataclass
class PerfRow:
    interval: Optional[float] = None
    cpu: Optional[int] = None
    counter_value: Optional[float] = None
    unit: Optional[str] = None
    event: Optional[str] = None
    event_runtime: Optional[float] = None
    pcnt_running: Optional[float] = None
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = None
    metric_threshold: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PerfRow":
        """Maps JSON keys to dataclass fields, preserving None/null values."""

        def safe_cast(key, target_type):
            val = data.get(key)
            if val is None or val == "":
                return None
            try:
                return target_type(val)
            except ValueError:
                return None

        return cls(
            interval=safe_cast("interval", float),
            cpu=safe_cast("cpu", int),
            counter_value=safe_cast("counter-value", float),
            unit=data.get("unit"),  # Already a string or None
            event=data.get("event"),
            event_runtime=safe_cast("event-runtime", float),
            pcnt_running=safe_cast("pcnt-running", float),
            metric_value=safe_cast("metric-value", float),
            metric_unit=data.get("metric-unit"),
            metric_threshold=data.get("metric-threshold")
        )

@dataclass
class PerfResults:
    rows: List[PerfRow]

    @classmethod
    def from_jsonl(cls, file_path: str) -> "PerfResults":
        """Parses a JSONL file line by line into the PerfResults container."""
        rows = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    rows.append(PerfRow.from_dict(data))
                except json.JSONDecodeError as e:
                    pass #print(f"Skipping invalid JSON line: {e}")
        
        return cls(rows=rows)
