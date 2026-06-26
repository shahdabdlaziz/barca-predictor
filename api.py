import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# CONFIGURATION & STARTUP
DATA_FILE  = 'FCB_Football_Matches.xlsx'
MODEL_FILE = 'barca_prediction_model.joblib'
BARCA_NAMES = {'barcelona', 'barca', 'fc barcelona'}
REVERSE_LABEL_MAP = {0: 'L', 1: 'D', 2: 'W'}

app = FastAPI(title="FC Barcelona Match Predictor API", version="1.0")

# Enable CORS so your front-end (React, Vue, HTML/JS) can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend's actual URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Data into memory when the server starts
if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
    raise RuntimeError("Missing model or data files. Run your training script first.")

print("Loading model and historical data...")
_api_model = joblib.load(MODEL_FILE)
_api_df    = pd.read_excel(DATA_FILE)
_api_df['Date']          = pd.to_datetime(_api_df['Date'])
_api_df                  = _api_df.sort_values('Date').reset_index(drop=True)
_api_df['Result_Points'] = _api_df['Result'].map({'W':3, 'D':1, 'L':0})
print("✓ API Ready")


# DATA MODELS 
class MatchInput(BaseModel):
    home_team:    str
    away_team:    str
    referee_name: str = "Other"
    competition:  str = "La Liga"

def apply_custom_thresholds(prob_dict, win_min=50.0, loss_min=20.0):
    """Custom risk logic for final prediction."""
    w = prob_dict.get('W', 0.0)
    l = prob_dict.get('L', 0.0)
    
    if l >= loss_min: return 'L'
    elif w < win_min: return 'D'
    else: return 'W'

def _build_feature_row(opponent, venue, referee, competition, source_df):
    """Generates features for the API request."""
    row = {
        'Venue': [venue], 'Opponent': [opponent], 
        'Competition': [competition], 'Referee': [referee]
    }
    row['Barca_xG_Form']   = [source_df['Barca_xG'].tail(3).mean()]
    row['Barca_Goal_Form'] = [source_df['Barca_Goals'].tail(3).mean()]
    row['Points_Last5']    = [source_df['Result_Points'].tail(5).mean()]
    
    h2h_matches = source_df[source_df['Opponent'] == opponent]['Result_Points']
    row['H2H_Points'] = [h2h_matches.mean() if len(h2h_matches) > 0 else source_df['Result_Points'].mean()]
    
    last_date = pd.to_datetime(source_df['Date'].iloc[-1])
    row['DaysRest'] = [min(30, max(1, (pd.Timestamp.now() - last_date).days))]
    
    return pd.DataFrame(row)


# ENDPOINTS
@app.get("/")
def health_check():
    return {"status": "online", "message": "FC Barcelona Predictor API is running."}

@app.post("/predict")
def predict_match(match: MatchInput):
    try:
        # Determine venue
        if match.home_team.lower() in BARCA_NAMES:
            venue, opponent, other_team = 'Home', match.away_team, match.away_team
        else:
            venue, opponent, other_team = 'Away', match.home_team, match.home_team

        # Normalize specific categories to match training data
        ref_map  = _api_df['Referee'].value_counts()
        comp_map = _api_df['Competition'].value_counts()
        norm_ref  = match.referee_name if ref_map.get(match.referee_name, 0) >= 3 else 'Other'
        norm_comp = match.competition  if comp_map.get(match.competition, 0) >= 3 else 'Other'

        # Generate features and predict
        input_data    = _build_feature_row(opponent, venue, norm_ref, norm_comp, _api_df)
        probabilities = _api_model.predict_proba(input_data)[0]
        classes       = _api_model.named_steps['classifier'].classes_
        
        raw_probs = {
            REVERSE_LABEL_MAP[int(classes[i])]: round(probabilities[i] * 100, 1)
            for i in range(len(classes))
        }

        # Apply logic
        barca_result = apply_custom_thresholds(raw_probs, win_min=50.0, loss_min=20.0)
        predicted_winner = (
            "Barcelona" if barca_result == 'W' else
            other_team  if barca_result == 'L' else "Draw"
        )

        # Return clean JSON 
        return {
            "match_details": {
                "home": match.home_team,
                "away": match.away_team,
                "referee": match.referee_name,
                "competition": match.competition
            },
            "prediction": {
                "winner_text": predicted_winner,
                "barca_result_code": barca_result
            },
            "probabilities": {
                "barca_win": raw_probs.get('W', 0.0),
                "draw": raw_probs.get('D', 0.0),
                "opponent_win": raw_probs.get('L', 0.0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))