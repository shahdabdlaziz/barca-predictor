import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

DATA_FILE = 'FCB_Football_Matches.xlsx'
MODEL_FILE = 'barca_prediction_model.joblib'
LABEL_MAP = {'L': 0, 'D': 1, 'W': 2}

print("🔄 Retraining model to match your current library version...")

df = pd.read_excel(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

ref_counts = df['Referee'].value_counts()
df['Referee'] = df['Referee'].apply(lambda x: 'Other' if ref_counts[x] < 3 else x)
comp_counts = df['Competition'].value_counts()
df['Competition'] = df['Competition'].apply(lambda x: 'Other' if comp_counts[x] < 3 else x)

df['Result_Points'] = df['Result'].map({'W': 3, 'D': 1, 'L': 0})
df['Barca_xG_Form']   = df['Barca_xG'].shift(1).rolling(3, min_periods=1).mean().fillna(df['Barca_xG'].mean())
df['Barca_Goal_Form'] = df['Barca_Goals'].shift(1).rolling(3, min_periods=1).mean().fillna(df['Barca_Goals'].mean())
df['Points_Last5']    = df['Result_Points'].shift(1).rolling(5, min_periods=1).mean().fillna(df['Result_Points'].mean())
df['H2H_Points'] = df.groupby('Opponent')['Result_Points'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean()).fillna(df['Result_Points'].mean())
df['DaysRest'] = df['Date'].diff().dt.days.fillna(7).clip(1, 30)

rolling_num = ['Barca_xG_Form', 'Barca_Goal_Form', 'Points_Last5', 'H2H_Points', 'DaysRest']
categorical_features = ['Venue', 'Opponent', 'Competition', 'Referee']
all_features = categorical_features + rolling_num

X = df[all_features]
y = df['Result'].map(LABEL_MAP)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, stratify=y, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), rolling_num)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=500, max_depth=5, class_weight='balanced', random_state=42))
])

model.fit(X_train, y_train)
joblib.dump(model, MODEL_FILE)
print("✨ Fresh, compatible model saved successfully!")