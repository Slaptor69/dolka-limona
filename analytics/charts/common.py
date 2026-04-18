from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math
import pandas as pd
import plotly.graph_objects as go

RING_ORDER = ["< 5 km", "5-10 km", "10-15 km", "15-20 km", "> 20 km"]
CTE_MIDPOINTS = {
    "< 15": 12.5,
    "15-20": 17.5,
    "20-30": 25.0,
    "30-45": 37.5,
    "45-60": 52.5,
    "60-90": 75.0,
    "90-120": 105.0,
    "> 90": 105.0,
}
CTE_UPPER_BOUNDS = {
    "< 15": 15,
    "15-20": 20,
    "20-30": 30,
    "30-45": 45,
    "45-60": 60,
    "60-90": 90,
    "> 90": 120,
}
PROMISED_UPPER_BOUNDS = {
    "< 30": 30,
    "30-45": 45,
    "45-60": 60,
    "60-90": 90,
    "> 90": 120,
}
RING_BOUNDS_KM = {
    "< 5 km": (0.0, 5.0),
    "5-10 km": (5.0, 10.0),
    "10-15 km": (10.0, 15.0),
    "15-20 km": (15.0, 20.0),
    "> 20 km": (20.0, 30.0),
}
KREMLIN_LAT = 55.7522
KREMLIN_LON = 37.6175


@dataclass(frozen=True)
class MetricConfig:
    key: str
    title: str
    unit: str
    colorscale: List[str]
    output_name: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_path() -> Path:
    return project_root() / "data" / "magnit_data.csv"


def output_path(output_name: str) -> Path:
    out = project_root() / "output" / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def read_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or data_path()
    encodings = ("utf-8", "utf-8-sig", "cp1251", "windows-1251")

    last_exc: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_exc = exc

    raise RuntimeError(f"Could not decode dataset at {csv_path}") from last_exc


def prepare_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "order_day_of_week",
        "order_hour",
        "distance_to_kremlin_bin",
        "cte_bin",
        "promised_time_bin",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    frame = df.copy()
    frame["order_day_of_week"] = pd.to_numeric(frame["order_day_of_week"], errors="coerce")
    frame["order_hour"] = pd.to_numeric(frame["order_hour"], errors="coerce")
    frame = frame.dropna(subset=["order_day_of_week", "order_hour"])
    frame["order_day_of_week"] = frame["order_day_of_week"].astype(int)
    frame["order_hour"] = frame["order_hour"].astype(int)

    frame = frame[frame["distance_to_kremlin_bin"].isin(RING_ORDER)].copy()
    frame["delivery_time"] = frame["cte_bin"].map(CTE_MIDPOINTS)
    frame = frame.dropna(subset=["delivery_time"])
    frame["cte_upper"] = frame["cte_bin"].map(CTE_UPPER_BOUNDS)
    frame["promised_upper"] = frame["promised_time_bin"].map(PROMISED_UPPER_BOUNDS)
    frame["is_late"] = (frame["cte_upper"] > frame["promised_upper"]).astype(float)

    grouped = (
        frame.groupby(["order_day_of_week", "order_hour", "distance_to_kremlin_bin"], as_index=False)
        .agg(
            order_count=("order_id", "count"),
            late_rate=("is_late", "mean"),
            delivery_time=("delivery_time", "mean"),
        )
        .rename(columns={"distance_to_kremlin_bin": "ring"})
    )
    grouped["late_rate"] = grouped["late_rate"] * 100.0

    time_axis = grouped[["order_day_of_week", "order_hour"]].drop_duplicates()
    rings = pd.DataFrame({"ring": RING_ORDER})
    complete = (
        time_axis.assign(_tmp=1)
        .merge(rings.assign(_tmp=1), on="_tmp", how="outer")
        .drop(columns="_tmp")
        .merge(grouped, on=["order_day_of_week", "order_hour", "ring"], how="left")
        .fillna({"order_count": 0, "late_rate": 0.0, "delivery_time": 0.0})
        .sort_values(["order_day_of_week", "order_hour"])
        .reset_index(drop=True)
    )
    complete["time_key"] = complete.apply(
        lambda row: f"{int(row['order_day_of_week'])}-{int(row['order_hour'])}",
        axis=1,
    )
    complete["time_label"] = complete.apply(
        lambda row: f"Day {int(row['order_day_of_week'])}, {int(row['order_hour']):02d}:00",
        axis=1,
    )

    return complete


def _circle_coords(radius_km: float, points: int = 64) -> List[List[float]]:
    coords: List[List[float]] = []
    for step in range(points + 1):
        angle = math.radians(step * 360.0 / points)
        dx = radius_km * math.cos(angle) / (111.32 * math.cos(math.radians(KREMLIN_LAT)))
        dy = radius_km * math.sin(angle) / 110.57
        coords.append([KREMLIN_LON + dx, KREMLIN_LAT + dy])
    return coords


def build_ring_geojson() -> Dict:
    features = []
    for ring, bounds in RING_BOUNDS_KM.items():
        inner_radius, outer_radius = bounds
        outer_ring = _circle_coords(outer_radius)
        inner_ring = _circle_coords(inner_radius)[::-1] if inner_radius > 0 else []

        coordinates = [outer_ring] if not inner_ring else [outer_ring, inner_ring]
        features.append(
            {
                "type": "Feature",
                "id": ring,
                "properties": {"name": ring},
                "geometry": {"type": "Polygon", "coordinates": coordinates},
            }
        )

    return {"type": "FeatureCollection", "features": features}


def _format_value(metric_key: str, value: float) -> str:
    if metric_key == "order_count":
        return f"{int(round(value)):,}".replace(",", " ")
    return f"{value:.1f}"


def _colorscale(scale: Iterable[str]) -> List[List[float | str]]:
    palette = list(scale)
    steps = max(len(palette) - 1, 1)
    return [[index / steps, color] for index, color in enumerate(palette)]


def build_metric_figure(aggregated: pd.DataFrame, metric: MetricConfig) -> go.Figure:
    geojson = build_ring_geojson()
    times = (
        aggregated[["time_key", "time_label", "order_day_of_week", "order_hour"]]
        .drop_duplicates()
        .sort_values(["order_day_of_week", "order_hour"])
        .reset_index(drop=True)
    )
    if times.empty:
        raise ValueError("No rows to visualize after aggregation")

    metric_max = float(aggregated[metric.key].max())
    zmax = 1.0 if metric_max <= 0 else metric_max

    def frame_payload(time_key: str) -> Tuple[List[float], List[List[str]]]:
        frame_rows = (
            aggregated[aggregated["time_key"] == time_key]
            .set_index("ring")
            .reindex(RING_ORDER)
            .fillna(0)
        )
        z_values = frame_rows[metric.key].astype(float).tolist()
        customdata = []
        for ring, row in frame_rows.iterrows():
            customdata.append(
                [
                    ring,
                    _format_value("late_rate", float(row["late_rate"])),
                    _format_value("delivery_time", float(row["delivery_time"])),
                    _format_value("order_count", float(row["order_count"])),
                    _format_value(metric.key, float(row[metric.key])),
                ]
            )
        return z_values, customdata

    first_key = times.loc[0, "time_key"]
    initial_z, initial_custom = frame_payload(first_key)

    figure = go.Figure(
        data=[
            go.Choroplethmapbox(
                geojson=geojson,
                featureidkey="id",
                locations=RING_ORDER,
                z=initial_z,
                zmin=0,
                zmax=zmax,
                colorscale=_colorscale(metric.colorscale),
                marker={"opacity": 0.74, "line": {"color": "#F8FAFC", "width": 1.5}},
                colorbar={"title": metric.title},
                customdata=initial_custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + f"{metric.title}: "
                    + "%{customdata[4]} "
                    + metric.unit
                    + "<br>Late rate: %{customdata[1]} %<br>"
                    + "Delivery time: %{customdata[2]} min<br>"
                    + "Orders: %{customdata[3]}<extra></extra>"
                ),
            )
        ]
    )

    frames = []
    slider_steps = []
    for _, time_row in times.iterrows():
        time_key = time_row["time_key"]
        time_label = time_row["time_label"]
        z_values, customdata = frame_payload(time_key)
        frames.append(
            go.Frame(
                name=time_key,
                data=[
                    go.Choroplethmapbox(
                        z=z_values,
                        customdata=customdata,
                    )
                ],
                traces=[0],
            )
        )
        slider_steps.append(
            {
                "label": time_label,
                "method": "animate",
                "args": [
                    [time_key],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )
    figure.frames = frames

    figure.update_layout(
        title=f"{metric.title} by Time Slice",
        mapbox={
            "style": "carto-darkmatter",
            "center": {"lat": KREMLIN_LAT, "lon": KREMLIN_LON},
            "zoom": 8.7,
        },
        paper_bgcolor="#0B1324",
        plot_bgcolor="#0B1324",
        font={"color": "#E2E8F0", "family": "Segoe UI"},
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.03,
                "xanchor": "left",
                "yanchor": "bottom",
                "direction": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 550, "redraw": True},
                                "transition": {"duration": 220},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "len": 0.9,
                "x": 0.05,
                "y": 0.01,
                "pad": {"t": 24, "b": 4},
                "currentvalue": {"prefix": "Time: ", "visible": True},
                "steps": slider_steps,
            }
        ],
    )
    return figure


def generate_metric_html(metric: MetricConfig) -> Path:
    raw = read_dataset()
    aggregated = prepare_aggregates(raw)
    figure = build_metric_figure(aggregated, metric)
    destination = output_path(metric.output_name)
    figure.write_html(destination, include_plotlyjs="cdn", full_html=True)
    return destination
