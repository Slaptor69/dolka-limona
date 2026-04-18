from common import MetricConfig, generate_metric_html


def main() -> None:
    metric = MetricConfig(
        key="delivery_time",
        title="Delivery Time",
        unit="min",
        colorscale=["#eef6ff", "#9ad1ff", "#3f8efc", "#1939b7"],
        output_name="delivery_time.html",
    )
    output = generate_metric_html(metric)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
