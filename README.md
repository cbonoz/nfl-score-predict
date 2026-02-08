# NFL Score Predictor

A machine learning system that predicts NFL game scores using scikit-learn's Random Forest and Gradient Boosting models, trained on historical NFL game data from 1999-2025.

## Features

- **Multi-target predictions**: Predicts away team score, home team score, and total score
- **Multiple models**: Uses Random Forest and Gradient Boosting regressors for comparison
- **Feature engineering**: Incorporates team identity, season, week, rest days, temperature, wind, stadium conditions
- **Model evaluation**: Provides MAE and R² metrics on test data
- **Feature importance analysis**: Shows which factors most impact scoring predictions

## Data Source

Historical NFL game data (7,275 games from 1999-2025):
- Source: https://raw.githubusercontent.com/nflverse/nfldata/refs/heads/master/data/games.csv
- Stored in: `data/games.csv`

## Installation

```bash
uv pip install numpy pandas scikit-learn scipy
```

Or using the provided pyproject.toml:
```bash
uv sync
```

## Usage

Open `predict.ipynb` and run the cells in order:

1. **Cell 1**: Import libraries
2. **Cell 2**: Load and explore the dataset
3. **Cell 3**: Preprocess data and engineer features
4. **Cell 4**: Create train/test split (80/20)
5. **Cell 5**: Train Random Forest and Gradient Boosting models for each target
6. **Cell 6**: Prepare features for the target game
7. **Cell 7**: Generate predictions and display feature importance

## Model Performance

Tested on 1,455 games (20% of data):

| Target | Random Forest MAE | RF R² | GB MAE | GB R² |
|--------|------------------|-------|--------|-------|
| Away Score | 8.02 | 0.009 | 7.85 | 0.047 |
| Home Score | 8.28 | 0.022 | 8.14 | 0.049 |
| Total Score | 10.88 | 0.009 | 10.74 | 0.044 |

## Example Prediction

For game `2025_22_SEA_NE` (Seattle Seahawks @ New England Patriots, Week 22):

**Random Forest Predictions:**
- Seattle (Away): **18.8 points**
- New England (Home): **19.5 points**
- Total: **44.7 points**

**Gradient Boosting Predictions:**
- Seattle (Away): **19.2 points**
- New England (Home): **25.1 points**
- Total: **45.9 points**

## Key Insights

Top predictive features:
1. Away team identity (17-18% importance)
2. Season (15-16% importance)
3. Week number (14-15% importance)
4. Temperature (13-14% importance)
5. Home team identity (11-12% importance)

Lower R² values reflect the inherent unpredictability of NFL outcomes due to numerous unmeasured variables (player injuries, play-calling, coaching changes, etc.).

## Project Structure

```
nfl-score-predict/
├── predict.ipynb          # Main prediction notebook
├── data/
│   └── games.csv          # Historical NFL game data
├── README.md              # This file
└── pyproject.toml         # Python dependencies
```
