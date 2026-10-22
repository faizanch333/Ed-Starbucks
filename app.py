import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Starbucks EDA & Decision Tree",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Starbucks Data Analysis")
st.write("Exploratory Data Analysis and Decision Tree Classification")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("starbucks.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "starbucks.csv not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()

st.success("Dataset loaded successfully!")

# -----------------------------------
# Dataset Overview
# -----------------------------------
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric("Duplicate Rows", int(df.duplicated().sum()))

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("Descriptive Statistics"):
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

with st.expander("Column Information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

# -----------------------------------
# EDA
# -----------------------------------
st.header("2. Exploratory Data Analysis")

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

category_cols = df.select_dtypes(
    exclude=np.number
).columns.tolist()

tab1, tab2, tab3, tab4 = st.tabs([
    "Category Distribution",
    "Calories",
    "Correlations",
    "Missing Values"
])

with tab1:
    category_col = next(
        (
            c for c in df.columns
            if "category" in c.lower()
        ),
        None
    )

    if category_col:
        counts = df[category_col].value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Distribution of Starbucks Categories")
        ax.set_xlabel(category_col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No category column found.")

with tab2:
    calorie_col = next(
        (
            c for c in numeric_cols
            if "calorie" in c.lower()
        ),
        None
    )

    if calorie_col:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            data=df,
            x=calorie_col,
            kde=True,
            ax=ax
        )
        ax.set_title("Calorie Distribution")
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 3))
        sns.boxplot(data=df, x=calorie_col, ax=ax)
        ax.set_title("Calories Box Plot")
        st.pyplot(fig)
        plt.close(fig)

        highest = df.loc[df[calorie_col].idxmax()]

        drink_col = next(
            (
                c for c in df.columns
                if any(
                    word in c.lower()
                    for word in [
                        "beverage", "drink", "name", "product"
                    ]
                )
            ),
            None
        )

        if drink_col:
            st.subheader("Highest-Calorie Drink")
            st.write(
                f"**{highest[drink_col]}** — "
                f"{highest[calorie_col]} calories"
            )
    else:
        st.info("No numerical calorie column found.")

with tab3:
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Nutritional Correlation Matrix")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numerical columns are required.")

with tab4:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if len(missing):
        st.bar_chart(missing)
    else:
        st.success("No missing values found.")

# -----------------------------------
# Decision Tree Model
# -----------------------------------
st.header("3. Decision Tree Classification")

# Find the classification target
target_col = next(
    (
        c for c in df.columns
        if c.lower().replace("_", " ").strip()
        in [
            "category",
            "beverage category",
            "drink category"
        ]
    ),
    None
)

if target_col is None:
    st.warning(
        "No category target was found. "
        "Check your dataset's target column."
    )
    st.stop()

# Remove rows without target labels
model_df = df.dropna(subset=[target_col]).copy()

X = model_df.drop(columns=[target_col])
y = model_df[target_col].astype(str)

# Remove columns with no useful variation
constant_cols = [
    c for c in X.columns
    if X[c].nunique(dropna=False) <= 1
]
X = X.drop(columns=constant_cols)

# Identify numerical and categorical features
num_features = X.select_dtypes(
    include=np.number
).columns.tolist()

cat_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

# Prepare missing values and encode categories
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    ("numeric", num_pipeline, num_features),
    ("categorical", cat_pipeline, cat_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    )
])

# Split data
class_counts = y.value_counts()
stratify_target = (
    y if len(class_counts) > 1
    and class_counts.min() >= 2
    else None
)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_target
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

except ValueError as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

accuracy = accuracy_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("Model Accuracy", f"{accuracy:.2%}")
m2.metric("Training Samples", len(X_train))
m3.metric("Testing Samples", len(X_test))

st.caption(
    f"Target column: {target_col} | "
    "Decision Tree maximum depth: 5"
)

# -----------------------------------
# Confusion Matrix
# -----------------------------------
st.subheader("Confusion Matrix")

labels = model.named_steps[
    "classifier"
].classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)
ax.set_xlabel("Predicted Category")
ax.set_ylabel("Actual Category")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# -----------------------------------
# Classification Report
# -----------------------------------
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

st.dataframe(
    pd.DataFrame(report).T,
    use_container_width=True
)

# -----------------------------------
# Feature Importance
# -----------------------------------
st.subheader("Top 15 Important Features")

fitted_preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    importance["Feature"].iloc[::-1],
    importance["Importance"].iloc[::-1]
)
ax.set_title("Decision Tree Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("View Feature Importance Table"):
    st.dataframe(importance, use_container_width=True)

# -----------------------------------
# Decision Tree Visualization
# -----------------------------------
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(22, 10))

plot_tree(
    classifier,
    feature_names=feature_names,
    class_names=[str(c) for c in classifier.classes_],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7,
    ax=ax
)

ax.set_title("Decision Tree — First Three Levels")
st.pyplot(fig)
plt.close(fig)

st.divider()
st.caption(
    "Starbucks EDA and Decision Tree project "
    "— built with Python and Streamlit."
)