# TMDB Movie Rating Prediction

Predict movie ratings using machine learning based on popularity metrics and release information.

## Dataset

- **Source**: TMDB Movies Dataset (`tmdb_movies_dataset.csv`)
- **Size**: 4000 movies
- **Features**: movie_id, title, release_date, vote_count, popularity, rating

## Libraries Used

| Library        | Purpose                       |
| -------------- | ----------------------------- |
| `pandas`       | Data loading and manipulation |
| `numpy`        | Numerical operations          |
| `matplotlib`   | Static visualizations         |
| `seaborn`      | Statistical plots (heatmaps)  |
| `scikit-learn` | ML models and preprocessing   |
| `joblib`       | Model serialization           |
| `streamlit`    | Web app deployment            |

## Workflow

### 1. Data Loading & Cleaning

- Load CSV and strip column whitespace
- Convert `release_date` to datetime
- Extract `release_year`, `release_month`, `release_day` as features
- Drop rows with missing values

### 2. Exploratory Data Analysis

- Rating distribution histogram
- Popularity vs Rating scatter plot
- Correlation heatmap between features
- Average rating by year (bar chart)

### 3. Feature Engineering

Features used for prediction:

- `vote_count` - Number of votes
- `popularity` - TMDB popularity score
- `release_year` - Year of release
- `release_month` - Month of release
- `release_day` - Day of release

Target: `rating`

### 4. Data Preprocessing

- **Train-Test Split**: 80% train, 20% test
- **StandardScaler**: Normalize features to zero mean and unit variance (important for linear models)

### 5. Models Trained

| Model             | Why Used                                      |
| ----------------- | --------------------------------------------- |
| Linear Regression | Baseline, simple relationship                 |
| Ridge Regression  | Handles multicollinearity                     |
| Lasso Regression  | Feature selection via L1 regularization       |
| Random Forest     | Captures non-linear patterns, ensemble method |
| Gradient Boosting | Sequential learning, often best performer     |

### 6. Evaluation Metrics

- **MSE** (Mean Squared Error) - Penalizes large errors
- **RMSE** (Root MSE) - Same unit as target
- **MAE** (Mean Absolute Error) - Average error magnitude
- **R² Score** - Proportion of variance explained (higher = better)

### 7. Model Selection & Saving

Best model (highest R²) saved as:

- `movie_rating_model.joblib`
- `scaler.joblib`

## Run the Notebook

```bash
cd labs/tmdb
# Open main.ipynb and run all cells
```

## Run the Web App

```bash
cd labs/tmdb
streamlit run deploy.py
```

## App Features

- **Predict Rating**: Enter movie metrics to predict rating
- **Explore Data**: View dataset statistics and distributions
- **Top Movies**: Browse highest-rated movies with filters
