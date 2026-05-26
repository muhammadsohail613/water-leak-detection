import pandas as pd
import numpy as np

np.random.seed(42)


def generate_flow_data(n_points=1000):
    timestamps = pd.date_range(start="2024-01-01", periods=n_points, freq="1min")

    # Normal flow: ~50 L/min with small noise
    flow_rate = np.random.normal(loc=50, scale=2, size=n_points)
    pressure = np.random.normal(loc=4.0, scale=0.1, size=n_points)
    velocity = np.random.normal(loc=1.2, scale=0.05, size=n_points)

    labels = np.zeros(n_points, dtype=int)  # 0 = normal

    # Inject leak events (sudden flow drop + pressure drop)
    leak_windows = [(200, 230), (500, 540), (780, 810)]
    for start, end in leak_windows:
        flow_rate[start:end] -= np.random.uniform(10, 20)  # flow drops
        pressure[start:end] -= np.random.uniform(0.5, 1.2)  # pressure drops
        velocity[start:end] -= np.random.uniform(0.2, 0.4)
        labels[start:end] = 1  # 1 = leak

    df = pd.DataFrame({
        "timestamp": timestamps,
        "flow_rate_lpm": np.round(flow_rate, 3),
        "pressure_bar": np.round(pressure, 3),
        "velocity_ms": np.round(velocity, 3),
        "label": labels
    })

    df.to_csv("data/sensor_data.csv", index=False)
    print(f"✅ Generated {n_points} rows — {labels.sum()} leak points injected")
    print(df.head(10))


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    generate_flow_data()