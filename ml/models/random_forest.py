from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_random_forest_model(random_state: int = 42) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    return pipeline