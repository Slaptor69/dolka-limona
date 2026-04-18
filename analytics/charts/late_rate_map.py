from common import MetricConfig, generate_metric_html


def main() -> None:
    metric = MetricConfig(
        key="late_rate",
        title="Courier Delays",
        unit="%",
        colorscale=["#fff4e6", "#ffbf69", "#ff7b54", "#d7263d"],
        output_name="late_rate.html",
    )
    output = generate_metric_html(metric)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
