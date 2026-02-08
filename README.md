# NFL Score Predictor

A machine learning system that predicts NFL game scores using scikit-learn's Random Forest and Gradient Boosting models, trained on historical NFL game data from 1999-2025.

## Features

- **Dual target predictions**: Predicts away team score and home team score independently
- **Derived total**: Total score calculated as Away + Home for mathematical consistency
- **Multiple models**: Uses Random Forest and Gradient Boosting regressors for comparison
- **Expanded features** (21 total):
  - **Temporal**: Season, week, day of week
  - **Conditions**: Temperature, wind, roof, surface
  - **Teams**: Away/home team identity, QB names, coaches
  - **Context**: Game type (regular/playoff), division game flag
  - **Market**: Over/under odds, spread line
  - **Rest**: Away/home rest days
- **Model evaluation**: Provides MAE and R² metrics with 5-fold cross-validation
- **Feature importance analysis**: Shows which factors most impact scoring predictions

## Data Source

Historical NFL game data (7,275+ games from 1999-2025):
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
3. **Cell 3**: Preprocess data, fill missing values, engineer features
4. **Cell 4**: Summary of features used in model
5. **Cell 5**: Create train/test split (80/20)
6. **Cell 6**: Train Random Forest and Gradient Boosting models for away and home scores
7. **Cell 7**: 5-fold cross-validation evaluation
8. **Cell 8**: Make predictions for target game (2025_22_SEA_NE)
9. **Cell 9**: Display model comparison and ensemble predictions
10. **Cell 10**: Show feature importance for top factors

## Model Performance

5-Fold Cross-Validation Results:

| Target | Random Forest CV MAE | GB CV MAE | GB Improvement |
|--------|---------------------|-----------|-----------------|
| Away Score | 8.37 ± 0.29 | 8.11 ± 0.25 | -3.1% |
| Home Score | 8.41 ± 0.16 | 8.15 ± 0.14 | -3.1% |

**Note**: Gradient Boosting outperforms Random Forest on both targets. Total Score is calculated as Away + Home predictions (no separate model), ensuring mathematical consistency.

## Example Prediction

For upcoming game `2025_22_SEA_NE` (Seattle Seahawks @ New England Patriots, Week 22):

**Ensemble Predictions** (average of RF & GB with expanded features):
- Away Team (SEA): **21.9 points**
- Home Team (NE): **23.3 points**
- Total Score: **45.2 points** (Away + Home)

*Features synthesized from recent games (2023+) since this is an upcoming matchup*

**Model Breakdown:**
- Random Forest: SEA 20.7, NE 23.2, Total 43.9
- Gradient Boosting: SEA 23.1, NE 23.4, Total 46.5

## Key Insights

Top predictive features (by importance):
1. Away team identity (17-18%)
2. Season (15-16%)
3. Week number (14-15%)
4. Temperature (13-14%)
5. Home team identity (11-12%)

**Why GB > QB importance is relatively low**: Teams have established scoring patterns; individual rest days matter less than seasonal trends and team composition.

**Total Score Calculation**: Total = Away Prediction + Home Prediction (ensures mathematical consistency and eliminates independent model bias).

Lower R² values (0.03-0.05) reflect the inherent unpredictability of NFL outcomes due to numerous unmeasured variables (player injuries, play-calling adjustments, coaching changes, weather conditions on game day, etc.).

## Project Structure

```
nfl-score-predict/
├── predict.ipynb          # Main prediction notebook
├── data/
│   └── games.csv          # Historical NFL game data
├── README.md              # This file
└── pyproject.toml         # Python dependencies
```

## Version History

- **v2** (Current): Expanded features (+QB/coach/game_type/weekday/betting_odds), total score = Away + Home
- **v1**: Core features only (season, week, rest, temp, wind, teams, roof, surface)
