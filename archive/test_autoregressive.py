import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("GOT HERE 1: Generating sample data.")
np.random.seed(42)
n = 100  # Number of data points (rows)
time = np.arange(n)
temperature = np.sin(time / 10) + 0.1 * np.random.randn(n)
pressure = np.cos(time / 10) + 0.1 * np.random.randn(n)
humidity = np.sin(time / 5) + 0.1 * np.random.randn(n)

df = pd.DataFrame(
    {
        "Time": time,
        "Temperature": temperature,
        "Pressure": pressure,
        "Humidity": humidity,
    }
)
print("GOT HERE 2: Dataframe created.")

print("GOT HERE 3: Applying AR(1) model to each feature.")


def apply_ar1_model(series):
    X = series[:-1].values.reshape(-1, 1)
    y = series[1:].values
    model = LinearRegression()
    model.fit(X, y)
    return model


temperature_model = apply_ar1_model(df["Temperature"])
pressure_model = apply_ar1_model(df["Pressure"])
humidity_model = apply_ar1_model(df["Humidity"])
print("GOT HERE 4: AR(1) models fitted for all features.")


def generate_ar1_sequence(model, last_value, n_future):
    generated_values = [last_value]
    for _ in range(n_future):
        next_value = model.predict(np.array(generated_values[-1]).reshape(-1, 1))
        generated_values.append(next_value[0])
    return generated_values


n_future = 10
temperature_future = generate_ar1_sequence(
    temperature_model, df["Temperature"].iloc[-1], n_future
)
pressure_future = generate_ar1_sequence(
    pressure_model, df["Pressure"].iloc[-1], n_future
)
humidity_future = generate_ar1_sequence(
    humidity_model, df["Humidity"].iloc[-1], n_future
)
print("GOT HERE 5: Generated future values for all features.")

print("GOT HERE 6: Plotting the original and generated data.")
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(df["Time"], df["Temperature"], label="Original Temperature")
plt.plot(
    np.arange(n, n + n_future),
    temperature_future[-n_future:],
    label="Generated Temperature",
    linestyle="--",
)
plt.title("Temperature Time Series")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df["Time"], df["Pressure"], label="Original Pressure")
plt.plot(
    np.arange(n, n + n_future),
    pressure_future[-n_future:],
    label="Generated Pressure",
    linestyle="--",
)
plt.title("Pressure Time Series")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df["Time"], df["Humidity"], label="Original Humidity")
plt.plot(
    np.arange(n, n + n_future),
    humidity_future[-n_future:],
    label="Generated Humidity",
    linestyle="--",
)
plt.title("Humidity Time Series")
plt.legend()

plt.tight_layout()
plt.show()

print("GOT HERE 7: Outputting the generated future values.")
print("Generated Future Temperature:", temperature_future[-n_future:])
print("Generated Future Pressure:", pressure_future[-n_future:])
print("Generated Future Humidity:", humidity_future[-n_future:])

print("GOT HERE 8: Creating and printing the table of original and generated values.")

future_time = np.arange(n, n + n_future)
future_df = pd.DataFrame(
    {
        "Time": future_time,
        "Generated Temperature": temperature_future[-n_future:],
        "Generated Pressure": pressure_future[-n_future:],
        "Generated Humidity": humidity_future[-n_future:],
    }
)

print("\nOriginal Data (Last 10 Rows) and Generated Future Data:")
combined_df = pd.concat(
    [df[["Time", "Temperature", "Pressure", "Humidity"]].tail(10), future_df],
    ignore_index=False,
)
print(combined_df)
