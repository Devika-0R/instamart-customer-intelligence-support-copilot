
"""Part B: Customer segmentation and leakage-aware lapse prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01")
LAPSE_CUTOFF = pd.Timestamp("2025-09-01")
LAPSE_END = pd.Timestamp("2025-10-01")


def load_data():
    """Load cleaned customer, order and product datasets."""

    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    products = pd.read_csv(CLEAN / "products.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"]
    )

    return customers, orders, products


# --------------------------------------------------
# 2. RFM CUSTOMER SEGMENTATION
# --------------------------------------------------

def rfm_and_segments(k=4):
    """
    Calculate RFM metrics and identify customer segments.

    RFM:
    Recency  - Days since last completed order.
    Frequency - Number of completed orders.
    Monetary - Total completed order value.
    """

    customers, orders, products = load_data()

    # Use completed orders before the snapshot date.
    completed = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] < SNAPSHOT_DATE)
    ].copy()

    # Calculate RFM values.
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order"]
    ).dt.days

    rfm = rfm.drop(columns="last_order")

    # Standardize RFM values.
    scaler = StandardScaler()

    rfm_features = ["recency", "frequency", "monetary"]

    X_scaled = scaler.fit_transform(rfm[rfm_features])

    # Compare different cluster counts.
    diagnostics = []

    for n in range(2, 9):

        model = KMeans(
            n_clusters=n,
            n_init=10,
            random_state=42,
        )

        labels = model.fit_predict(X_scaled)

        diagnostics.append({
            "k": n,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(
                X_scaled, labels
            ),
        })

    diagnostics = pd.DataFrame(diagnostics)

    # Fit final K-Means model.
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )

    rfm["segment_id"] = kmeans.fit_predict(X_scaled)

    # Create segment profile.
    profile = (
        rfm.groupby("segment_id")
        .agg(
            size=("customer_id", "size"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    # Identify Premium segment using highest mean monetary.
    premium_id = profile.loc[
        profile["monetary"].idxmax(), "segment_id"
    ]

    # Assign descriptive segment names.
    other_segments = profile[
        profile["segment_id"] != premium_id
    ].copy()

    # Among remaining segments, identify the most frequent.
    frequent_id = other_segments.loc[
        other_segments["frequency"].idxmax(),
        "segment_id"
    ]

    # Identify the least recent/least frequent group.
    remaining = other_segments[
        other_segments["segment_id"] != frequent_id
    ].copy()

    if len(remaining) > 0:
        occasional_id = remaining.loc[
            remaining["recency"].idxmax(),
            "segment_id"
        ]
    else:
        occasional_id = None

    name_map = {}

    for segment_id in profile["segment_id"]:

        if segment_id == premium_id:
            name_map[segment_id] = "Premium Customers"

        elif segment_id == frequent_id:
            name_map[segment_id] = "Loyal Active Customers"

        elif segment_id == occasional_id:
            name_map[segment_id] = "Occasional Customers"

        else:
            name_map[segment_id] = "Regular Customers"

    rfm["segment_name"] = rfm["segment_id"].map(name_map)

    rfm["is_premium"] = (
        rfm["segment_id"] == premium_id
    )

    # PCA for two-dimensional visualization.
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_

    # Save customer segment results.
    OUT.mkdir(parents=True, exist_ok=True)

    rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium",
        ]
    ].to_csv(
        OUT / "customer_segments.csv",
        index=False,
    )

    return (
        rfm,
        diagnostics,
        profile,
        explained_variance,
    )


# --------------------------------------------------
# 3. LAPSE PREDICTION
# --------------------------------------------------

def build_lapse_model():
    """
    Build and evaluate customer lapse prediction models.

    Features use information before September 1, 2025.
    Lapse label uses September 2025 activity.
    """

    customers, orders, products = load_data()

    # Historical orders before the cutoff.
    historical = orders[
        orders["order_ts"] < LAPSE_CUTOFF
    ].copy()

    completed = historical[
        historical["status"] == "completed"
    ].copy()

    # Customers with at least one historical completed order.
    cohort = (
        completed.groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )

    cohort["recency"] = (
        LAPSE_CUTOFF - cohort["last_order"]
    ).dt.days

    cohort = cohort.drop(columns="last_order")

    # Calculate cancellation rate using historical final orders.
    final_orders = historical[
        historical["status"].isin(["completed", "cancelled"])
    ]

    cancellation = (
        final_orders.groupby("customer_id")
        .agg(
            total_orders=("order_id", "nunique"),
            cancelled_orders=(
                "status",
                lambda x: (x == "cancelled").sum(),
            ),
        )
        .reset_index()
    )

    cancellation["cancellation_rate"] = (
        cancellation["cancelled_orders"]
        / cancellation["total_orders"]
    )

    cohort = cohort.merge(
        cancellation[
            ["customer_id", "cancellation_rate"]
        ],
        on="customer_id",
        how="left",
    )

    # Find each customer's favourite product category.
    category_data = completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )

    category_counts = (
        category_data.groupby(
            ["customer_id", "category"]
        )
        .size()
        .reset_index(name="order_count")
    )

    favourite = (
        category_counts.sort_values(
            ["customer_id", "order_count", "category"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        .rename(
            columns={"category": "favorite_category"}
        )
        [["customer_id", "favorite_category"]]
    )

    cohort = cohort.merge(
        favourite,
        on="customer_id",
        how="left",
    )

    # Add customer information.
    cohort = cohort.merge(
        customers,
        on="customer_id",
        how="left",
    )

    cohort["tenure_days"] = (
        LAPSE_CUTOFF - cohort["signup_date"]
    ).dt.days

    # Create September lapse labels.
    september_completed = orders[
        (orders["order_ts"] >= LAPSE_CUTOFF)
        & (orders["order_ts"] < LAPSE_END)
        & (orders["status"] == "completed")
    ]

    active_customers = set(
        september_completed["customer_id"].unique()
    )

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(active_customers)
    ).astype(int)

    # Separate customer ID and target from model features.
    customer_ids = cohort["customer_id"].copy()
    y = cohort["lapsed"].copy()

    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days",
    ]

    categorical_features = [
        "favorite_category",
        "city",
        "gender",
    ]

    X = cohort[
        numeric_features + categorical_features
    ].copy()

    # Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Preprocessing: fit imputers and encoders on training data only.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    # Model definitions and hyperparameter grids.
    model_settings = [
        (
            "logistic",
            LogisticRegression(max_iter=1000),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        (
            "forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [5, 10, 20],
            },
        ),
    ]

    models = {}

    for name, estimator, parameter_grid in model_settings:

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        search = GridSearchCV(
            pipeline,
            parameter_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        search.fit(X_train, y_train)

        predictions = search.predict(X_test)

        models[name] = {
            "search": search,
            "best_params": search.best_params_,
            "cv_f1": search.best_score_,
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
        }

    # Select Logistic Regression to match the notebook analysis.
    selected_search = models["logistic"]["search"]
    selected_model = selected_search.best_estimator_

    # Save selected model.
    OUT.mkdir(parents=True, exist_ok=True)

    model_path = OUT / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)

    # Confirm saved model gives the same predictions.
    loaded_model = joblib.load(model_path)

    assert np.array_equal(
        selected_model.predict(X_test),
        loaded_model.predict(X_test),
    )

    # Baseline: predict every customer as not lapsed.
    baseline_predictions = np.zeros(
        len(y_test), dtype=int
    )

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    # Build Premium retention list.
    segments, _, _, _ = rfm_and_segments(k=4)

    premium_ids = set(
        segments.loc[
            segments["is_premium"],
            "customer_id",
        ]
    )

    lapse_probabilities = selected_model.predict_proba(
        X_test
    )[:, list(selected_model.classes_).index(1)]

    retention = pd.DataFrame({
        "customer_id": customer_ids.loc[X_test.index].values,
        "lapse_probability": lapse_probabilities,
        "actually_lapsed": y_test.values,
    })

    retention = (
        retention[
            retention["customer_id"].isin(premium_ids)
        ]
        .sort_values(
            "lapse_probability",
            ascending=False,
        )
        .head(10)
    )

    retention.to_csv(
        OUT / "retention_list.csv",
        index=False,
    )

    # Return results for notebook display.
    model_summary = {
        name: {
            key: value
            for key, value in details.items()
            if key != "search"
        }
        for name, details in models.items()
    }

    return {
        "cohort_rows": len(cohort),
        "lapsed": int(y.sum()),
        "baseline_accuracy": baseline_accuracy,
        "models": model_summary,
        "retention_list": retention,
    }
