"""Deterministic preparation of AGC 2019 greenhouse root-zone data.

The raw Autonomous Greenhouse Challenge files are external data.  This module
keeps every cleaning decision explicit and produces small audit records rather
than treating malformed values as physical observations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator


EXCEL_EPOCH = datetime(1899, 12, 30, tzinfo=timezone.utc)
COMPARTMENTS = (
    "AICU",
    "Automatoes",
    "Digilog",
    "IUACAAS",
    "TheAutomators",
    "Reference",
)


@dataclass(frozen=True)
class IrrigationEvent:
    compartment: str
    timestamp: str
    excel_time: float
    delivered_l_m2: float
    cumulative_l_m2: float
    flag: str = ""


@dataclass(frozen=True)
class RootZoneRecord:
    compartment: str
    timestamp: str
    excel_time: float
    wc_slab1_pct: float | None
    wc_slab2_pct: float | None
    ec_slab1_ds_m: float | None
    ec_slab2_ds_m: float | None
    t_slab1_c: float | None
    t_slab2_c: float | None
    flags: str = ""


def excel_datetime(value: float | str) -> datetime:
    """Convert an Excel serial day using the convention declared by AGC."""
    return EXCEL_EPOCH + timedelta(days=float(value))


def timestamp_text(value: float | str) -> str:
    return excel_datetime(value).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_headers(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip(): value for key, value in row.items()}


def _number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _bounded(value: object, low: float, high: float) -> tuple[float | None, bool]:
    number = _number(value)
    if number is None:
        return None, False
    if number < low or number > high:
        return None, True
    return number, False


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield _clean_headers(row)


def reconstruct_irrigation_events(
    rows: Iterable[dict[str, str]],
    compartment: str,
    *,
    event_limit_l_m2: float = 10.0,
) -> tuple[list[IrrigationEvent], dict[str, int]]:
    """Turn daily cumulative irrigation into timestamped delivery events.

    The AGC documentation says that the counter resets daily, but the observed
    reset timestamp is not the Excel calendar boundary and changes during the
    campaign. Resets are therefore detected from a counter decrease. The first
    row is left-truncated and contributes no invented delivery. Events above
    ``event_limit_l_m2`` are retained but flagged for source review.
    """
    events: list[IrrigationEvent] = []
    counts = {
        "rows": 0,
        "missing_time": 0,
        "missing_counter": 0,
        "resumed_after_missing": 0,
        "counter_resets": 0,
        "large_increment": 0,
        "positive_events": 0,
    }
    previous_time: float | None = None
    previous_cumulative: float | None = None
    missing_since_valid = False
    for raw in rows:
        counts["rows"] += 1
        serial = _number(raw.get("%time", raw.get("%Time")))
        cumulative = _number(raw.get("Cum_irr"))
        if serial is None:
            counts["missing_time"] += 1
            continue
        if cumulative is None or cumulative < 0:
            counts["missing_counter"] += 1
            # Preserve the last valid counter. When readings resume, the counter
            # difference still identifies delivery across a short missing block.
            # A reset is likewise visible as a decrease. Long gaps remain evident
            # in the source audit and are not filled row by row.
            missing_since_valid = previous_cumulative is not None
            continue

        flag = ""
        if previous_time is None or previous_cumulative is None:
            increment, flag = 0.0, "left_truncated_counter"
        else:
            increment = cumulative - previous_cumulative
            if increment < -1e-9:
                counts["counter_resets"] += 1
                increment, flag = cumulative, "counter_reset"
            elif increment < 0:
                increment = 0.0

        if increment > event_limit_l_m2:
            counts["large_increment"] += 1
            flag = ";".join(filter(None, (flag, "large_increment")))
        if increment > 1e-9:
            counts["positive_events"] += 1
            events.append(IrrigationEvent(
                compartment=compartment,
                timestamp=timestamp_text(serial),
                excel_time=serial,
                delivered_l_m2=increment,
                cumulative_l_m2=cumulative,
                flag=flag,
            ))
        if missing_since_valid:
            counts["resumed_after_missing"] += 1
            missing_since_valid = False
        previous_time, previous_cumulative = serial, cumulative
    return events, counts


def clean_root_zone(
    rows: Iterable[dict[str, str]], compartment: str
) -> tuple[list[RootZoneRecord], dict[str, int]]:
    """Apply declared physical bounds and preserve a flag for every rejection."""
    records: list[RootZoneRecord] = []
    counts = {"rows": 0, "missing_time": 0, "rejected_values": 0}
    fields = (
        ("WC_slab1", "wc_slab1_pct", 0.01, 100.0),
        ("WC_slab2", "wc_slab2_pct", 0.01, 100.0),
        ("EC_slab1", "ec_slab1_ds_m", 0.01, 20.0),
        ("EC_slab2", "ec_slab2_ds_m", 0.01, 20.0),
        ("t_slab1", "t_slab1_c", -5.0, 60.0),
        ("t_slab2", "t_slab2_c", -5.0, 60.0),
    )
    for raw in rows:
        counts["rows"] += 1
        serial = _number(raw.get("%time", raw.get("%Time")))
        if serial is None:
            counts["missing_time"] += 1
            continue
        values: dict[str, float | None] = {}
        flags: list[str] = []
        for source, target, low, high in fields:
            value, rejected = _bounded(raw.get(source), low, high)
            values[target] = value
            if rejected:
                counts["rejected_values"] += 1
                flags.append(f"{source}:out_of_bounds")
        records.append(RootZoneRecord(
            compartment=compartment,
            timestamp=timestamp_text(serial),
            excel_time=serial,
            flags=";".join(flags),
            **values,
        ))
    return records, counts


def daily_resource_rows(rows: Iterable[dict[str, str]]) -> dict[int, dict[str, float | None]]:
    result: dict[int, dict[str, float | None]] = {}
    for raw in rows:
        serial = _number(raw.get("%time", raw.get("%Time")))
        if serial is None:
            continue
        result[math.floor(serial)] = {
            "irrigation_l_m2": _number(raw.get("Irr")),
            "drain_l_m2": _number(raw.get("Drain")),
        }
    return result


def water_balance_audit(
    events: Iterable[IrrigationEvent], resources: dict[int, dict[str, float | None]]
) -> list[dict[str, float | int | None]]:
    reconstructed: dict[int, float] = {}
    for event in events:
        day = math.floor(event.excel_time)
        reconstructed[day] = reconstructed.get(day, 0.0) + event.delivered_l_m2
    rows = []
    for day in sorted(set(reconstructed) | set(resources)):
        source = resources.get(day, {})
        reported = source.get("irrigation_l_m2")
        rebuilt = reconstructed.get(day, 0.0)
        rows.append({
            "excel_day": day,
            "reconstructed_irrigation_l_m2": rebuilt,
            "reported_irrigation_l_m2": reported,
            "difference_l_m2": None if reported is None else rebuilt - reported,
            "reported_drain_l_m2": source.get("drain_l_m2"),
        })
    return rows


def write_dataclasses(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_dicts(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_agc2019(source: Path, destination: Path) -> dict:
    """Prepare all six compartments and return a machine-readable audit."""
    destination.mkdir(parents=True, exist_ok=True)
    audit: dict = {
        "source": str(source.resolve()),
        "cleaning_version": 1,
        "compartments": {},
        "source_files": {},
    }
    for compartment in COMPARTMENTS:
        folder = source / compartment
        climate = folder / "GreenhouseClimate.csv"
        root_zone = folder / "GrodanSens.csv"
        resources = folder / "Resources.csv"
        for path in (climate, root_zone, resources):
            if not path.is_file():
                raise FileNotFoundError(path)
            audit["source_files"][str(path.relative_to(source))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        events, event_audit = reconstruct_irrigation_events(iter_csv(climate), compartment)
        roots, root_audit = clean_root_zone(iter_csv(root_zone), compartment)
        resources_by_day = daily_resource_rows(iter_csv(resources))
        balance = water_balance_audit(events, resources_by_day)
        out = destination / compartment
        write_dataclasses(out / "irrigation_events.csv", events)
        write_dataclasses(out / "root_zone.csv", roots)
        write_dicts(out / "daily_water_audit.csv", balance)
        comparable = [abs(float(row["difference_l_m2"])) for row in balance
                      if row["difference_l_m2"] is not None]
        audit["compartments"][compartment] = {
            "irrigation": event_audit,
            "root_zone": root_audit,
            "daily_rows": len(balance),
            "daily_irrigation_max_abs_difference_l_m2": max(comparable, default=None),
            "daily_irrigation_mean_abs_difference_l_m2": (
                sum(comparable) / len(comparable) if comparable else None
            ),
        }

    audit_path = destination / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit
