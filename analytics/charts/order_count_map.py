from common import MetricConfig, generate_metric_html


def main() -> None:
    metric = MetricConfig(
        key="order_count",
        title="Order Load",
        unit="orders",
        colorscale=["#f3ffe5", "#b8e986", "#4caf50", "#1f6b2c"],
        output_name="order_count.html",
    )
    output = generate_metric_html(metric)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
