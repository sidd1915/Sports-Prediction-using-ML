from flask import Flask , request , render_template
import pickle
import pandas as pd

app = Flask(__name__)

xgb_clf = pickle.load(open('xgb_model.pkl','rb'))
label_encoder = pickle.load(open('label_encoder.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

def load_venue_mapping():
    venue_df = pd.read_csv('venues.csv')
    venue_map = dict(zip(venue_df['venue_name'],venue_df['venue_id']))
    return venue_map

venue_mapping = load_venue_mapping()

def predict_winner(home_team,away_team,venue_name,toss_won,decision,nrr):
    home_team_id = label_encoder.transform([home_team])[0]
    away_team_id = label_encoder.transform([away_team])[0]
    toss_won_id = label_encoder.transform([toss_won])[0]


    decision = 0 if decision.lower() == 'bat' else 1

    venue_id = venue_mapping[venue_name]

    input_data = {
        'home_team': home_team_id,
        'away_team': away_team_id,
        'venue_id': venue_id,
        'toss_won': toss_won_id,
        'decision':decision,
        'nrr':nrr
    }

    input_df = pd.DataFrame([input_data])

    input_features = scaler.transform(input_df)

    pred_winner = xgb_clf.predict(input_features)[0]

    pred_winner_team = label_encoder.inverse_transform([pred_winner])[0]

    return pred_winner_team


@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        venue_name = request.form['venue_name']
        toss_won_team = request.form['toss_won']
        decision = request.form['decision']
        nrr = request.form['nrr']

        winner = predict_winner(home_team, away_team, venue_name, toss_won_team, decision, nrr)

        return render_template('index.html', winner=winner, venue_mapping=venue_mapping.keys())
    
    
    return render_template('index.html', venue_mapping=venue_mapping.keys())


if __name__ == '__main__':
    app.run(debug=True)