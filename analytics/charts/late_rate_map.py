import math
from pathlib import Path

import pandas as pd
import plotly.express as px


def create_ring_geojson() -> dict:
    kremlin_lat, kremlin_lon = 55.7522, 37.6175
    features = []
    mapping = {
        "< 5 km": (0, 5),
        "5-10 km": (5, 10),
        "10-15 km": (10, 15),
        "15-20 km": (15, 20),
        "> 20 km": (20, 30),
    }

    def get_coords(radius_km: float, n: int = 64) -> list[list[float]]:
        points = []
        for i in range(n + 1):
            angle = math.radians(i * 360 / n)
            dx = radius_km * math.cos(angle) / (111.32 * math.cos(math.radians(kremlin_lat)))
            dy = radius_km * math.sin(angle) / 110.57
            points.append([kremlin_lon + dx, kremlin_lat + dy])
        return points

    for bin_name, (r_in, r_out) in mapping.items():
        outer_ring = get_coords(r_out)
        inner_ring = get_coords(r_in)[::-1] if r_in > 0 else []
        features.append(
            {
                "type": "Feature",
                "id": bin_name,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [outer_ring, inner_ring] if inner_ring else [outer_ring],
                },
                "properties": {"name": bin_name},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "magnit_data.csv"
    out_path = root / "output" / "late_rate.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка и подготовка данных
    df = pd.read_csv(data_path)

    # --- УМНАЯ ЛОГИКА ОПОЗДАНИЙ ---
    cte_map = {
        "< 15": 15,
        "15-20": 20,
        "20-30": 30,
        "30-45": 45,
        "45-60": 60,
        "60-90": 90,
        "> 90": 120,
    }
    promised_map = {
        "< 30": 30,
        "30-45": 45,
        "45-60": 60,
        "60-90": 90,
        "> 90": 120,
    }

    df["cte_numeric"] = df["cte_bin"].map(cte_map)
    df["promised_numeric"] = df["promised_time_bin"].map(promised_map)
    df = df.dropna(subset=["cte_numeric", "promised_numeric", "distance_to_kremlin_bin"]).copy()

    # Заказ считается опоздавшим, если факт превысил обещание
    df["is_late"] = (df["cte_numeric"] > df["promised_numeric"]).astype(int)

    # --- СОРТИРОВКА ВРЕМЕНИ ---
    df = df.sort_values(["order_day_of_week", "order_hour"])
    df["time_label"] = (
        "День " + df["order_day_of_week"].astype(str) + ", час " + df["order_hour"].astype(str)
    )

    stats = (
        df.groupby(
            [
                "time_label",
                "distance_to_kremlin_bin",
                "order_day_of_week",
                "order_hour",
            ],
            as_index=False,
        )
        .agg(
            late_rate=("is_late", "mean"),
            order_count=("order_id", "count"),
        )
        .sort_values(["order_day_of_week", "order_hour"])
    )
    stats["late_rate"] *= 100

    rings_geojson = create_ring_geojson()

    # 3. Визуализация
    fig = px.choropleth_mapbox(
        stats,
        geojson=rings_geojson,
        locations="distance_to_kremlin_bin",
        color="late_rate",
        animation_frame="time_label",
        color_continuous_scale="Reds",
        range_color=[0, 40],
        mapbox_style="carto-darkmatter",
        center={"lat": 55.7522, "lon": 37.6175},
        zoom=9,
        opacity=0.6,
        labels={"late_rate": "% Опозданий", "order_count": "Заказов"},
        hover_data={"order_count": True},
    )

    fig.update_layout(
        title="Пульс опозданий: Факт vs Обещание (по кольцам Москвы)",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )

    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
