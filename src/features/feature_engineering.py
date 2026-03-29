import pandas as pd

def select_features(df: pd.DataFrame):
    features = [
        "avg_wind_speed",
        "avg_rotor_rpm",
        "avg_temp",
        "wind_speed_std",
        "hour",
        "day_of_week",
        "month",
        "lag_1h_wind",
        "lag_2h_wind",
        "lag_1h_power",
        "rolling_mean_wind_3h",
        "rolling_mean_power_3h"
    ]
    target = "target_power"
    return features, df[features], df[target]