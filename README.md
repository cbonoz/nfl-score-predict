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
3. **Cell 3**: Preprocess data, **define target_game_id**, exclude it from training to prevent data leakage, engineer features (21 total)
4. **Cell 4**: Summary of features used in model
5. **Cell 5**: Expanded Model Features (explanation)
6. **Cell 6**: Create train/test split (80/20)
7. **Cell 7**: Train Random Forest and Gradient Boosting models for away and home scores
8. **Cell 8**: 5-fold cross-validation evaluation
9. **Cell 9**: Understanding Cross-Validation Results (explanation)
10. **Cell 10**: Display target game data from CSV (all columns and actual values)
11. **Cell 11**: Generate ensemble predictions
12. **Cell 12**: Display model comparison (Random Forest vs Gradient Boosting vs Ensemble)
13. **Cell 13**: Show feature importance for away and home scores

## Model Performance

5-Fold Cross-Validation Results:

| Target | Random Forest CV MAE | GB CV MAE | GB Improvement |
|--------|---------------------|-----------|-----------------|
| Away Score | 7.83 ± 0.23 | 7.72 ± 0.18 | -1.4% |
| Home Score | 7.93 ± 0.23 | 7.80 ± 0.21 | -1.6% |

**Note**: Gradient Boosting outperforms Random Forest on both targets. Total Score is calculated as Away + Home predictions (no separate model), ensuring mathematical consistency.

## Example Prediction

For game `2025_22_SEA_NE` (Seattle Seahawks @ New England Patriots, Week 22, Super Bowl):

**Ensemble Predictions** (average of RF & GB):
- Away Team (SEA): **25.1 points**
- Home Team (NE): **20.8 points**
- Total Score: **45.9 points** (Away + Home)

*Note: This game was excluded from training set to prevent data leakage. Predictions use actual game features from CSV.*

**Model Breakdown:**
- Random Forest: SEA 25.7, NE 21.1, Total 46.8
- Gradient Boosting: SEA 24.6, NE 20.4, Total 45.1

## Key Insights

Top predictive features (by importance):
1. Division game flag - ~14.3-14.5%
2. Roof type - ~8.3-8.4%
3. QB names (away & home) - ~7.9-8.2% combined
4. Surface type - ~7.7-8.0%
5. Temperature - ~6.9-7.0%
6. Season - ~6.5-6.7%
7. Betting odds - ~6.4-6.5%
8. Week number - ~6.4-6.5%
9. Wind - ~5.9-6.0%

**Key Finding**: Division games are the strongest predictor of scoring outcomes, likely reflecting rivalry intensity and familiarity between division opponents. Surprisingly, team identity itself doesn't appear in the top 10, suggesting stadium/environmental factors are more predictive than team composition alone.

**Total Score Calculation**: Total = Away Prediction + Home Prediction (ensures mathematical consistency and eliminates independent model bias).

Lower R² values reflect the inherent unpredictability of NFL outcomes due to numerous unmeasured variables (player injuries, play-calling adjustments, real-time coaching decisions, actual weather conditions on game day, referee decisions, etc.). Our MAE of ~7.7 points is interpretable in football context—a ~8 point prediction error is reasonable given the sport's high variance.

## Project Structure

```
nfl-score-predict/
├── predict.ipynb          # Main prediction notebook
├── data/
│   └── games.csv          # Historical NFL game data
├── README.md              # This file
└── pyproject.toml         # Python dependencies
```

## Key Implementation Details

**Data Leakage Prevention**: 
- Target game ID is defined in Cell 3 before train/test split
- Target game is extracted from original data before cleaning
- Target game is excluded from training set during preprocessing
- Original game data is preserved separately for reference and prediction

**Feature Handling**:
- Missing categorical values filled with 'Unknown' before encoding
- Categorical features encoded using stored LabelEncoders from training
- Missing numeric values filled with training data mean
- All missing values handled before model prediction to avoid NaN errors

## Version History

- **v2** (Current): Expanded features (+QB/coach/game_type/weekday/betting_odds), total score = Away + Home, data leakage prevention, CSV column reference
- **v1**: Core features only (season, week, rest, temp, wind, teams, roof, surface)
