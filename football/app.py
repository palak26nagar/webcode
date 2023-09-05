from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained MLP model (with error handling)
#try:
from joblib import load
mlp_model = load('mlp_model.joblib')

#except FileNotFoundError:
   # print("Model file not found")
    #mlp_model = None
#except Exception as e:
    #print(f"An error occurred while loading the model: {e}")
   # mlp_model = None


# Dummy scores for all teams (replace with actual scores)
team_scores = {
    'Arsenal': 17.1252348821881,
    'Aston Villa': 17.1687895550603,
    'Bournemouth': 17.6924359596587,
    'Brentford': 17.5528471328973,
    'Brighton': 17.3514934041518,
    'Burnley': 14.7844903605243,
    'Chelsea': 18.2857464834588,
    'Crystal Palace': 16.2150294659922,
    'Everton': 16.8391790911814,
    'Fulham': 14.7100877479451,
    'Leeds United': 19.7998512067696,
    'Leicester City': 21.6256982924356,
    'Liverpool': 25.9951026333797,
    'Manchester City': 25.603573072,
    'Manchester Utd': 16.1538295521982,
    'Newcastle Utd': 15.8309960048378,
    'Norwich City': 15.007190286197,
    'Nottingham Forest': 17.8685602060204,
    'Sheffield Utd': 14.7814646575064,
    'Southampton': 20.1994614105463,
    'Tottenham': 19.8164338977403,
    'Watford': 13.7950995752172,
    'West Brom': 15.4068660829936,
    'West Ham': 21.5783244324313,
    'Wolves': 15.5394040481932
}


def preprocess_data(day, opponent, venue, time, score):
    # Initialize the input vector with zeros
    input_vector = [0] * 36

    # Encode the selected day
    days = ['Fri', 'Mon', 'Sat', 'Sun', 'Thu', 'Tue', 'Wed']
    input_vector[days.index(day)] = 1

    # Encode the selected opponent (Team 2)
    opponents = [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
        'Leeds United', 'Leicester City', 'Liverpool', 'Manchester City',
        'Manchester Utd', 'Newcastle Utd', 'Norwich City', 'Nottingham Forest',
        'Sheffield Utd', 'Southampton', 'Tottenham', 'Watford', 'West Brom',
        'West Ham', 'Wolves'
    ]
    input_vector[opponents.index(opponent) + 7] = 1

    # Encode the venue (0 for Away, 1 for Home)
    input_vector[32] = 1 if venue == 'Home' else 0

    # Encode the time (Afternoon or Evening)
    if time == 'Afternoon':
        input_vector[33] = 1
    elif time == 'Evening':
        input_vector[34] = 1

    # Include the score of Team 1
    input_vector[35] = score

    # Create a DataFrame with the correct feature names
    feature_names = [
        'Day_Fri', 'Day_Mon', 'Day_Sat', 'Day_Sun', 'Day_Thu', 'Day_Tue', 'Day_Wed',
        'Opponent_Arsenal', 'Opponent_Aston Villa', 'Opponent_Bournemouth',
        'Opponent_Brentford', 'Opponent_Brighton', 'Opponent_Burnley', 'Opponent_Chelsea',
        'Opponent_Crystal Palace', 'Opponent_Everton', 'Opponent_Fulham', 'Opponent_Leeds United',
        'Opponent_Leicester City', 'Opponent_Liverpool', 'Opponent_Manchester City',
        'Opponent_Manchester Utd', 'Opponent_Newcastle Utd', 'Opponent_Norwich City',
        "Opponent_Nott'ham Forest", 'Opponent_Sheffield Utd', 'Opponent_Southampton',
        'Opponent_Tottenham', 'Opponent_Watford', 'Opponent_West Brom', 'Opponent_West Ham',
        'Opponent_Wolves', 'Venue_Encoded', 'Time_Afternoon', 'Time_Evening', 'Score'
    ]
    input_df = pd.DataFrame([input_vector], columns=feature_names)

    return input_df

teams_with_images = [
    ('Arsenal', 'https://iconplanet.app/preview/png/256/arsenal-fc-logo--32049.png'),
    ('Aston Villa', 'https://iconplanet.app/preview/png/64/aston-villa-logo--32055.png'),
    ('Bournemouth', 'https://iconplanet.app/preview/png/64/afc-bournemouth-fc-logo--32059.png'),
    ('Brentford', 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2a/Brentford_FC_crest.svg/1200px-Brentford_FC_crest.svg.png'),
    ('Brighton', 'https://upload.wikimedia.org/wikipedia/en/thumb/f/fd/Brighton_%26_Hove_Albion_logo.svg/800px-Brighton_%26_Hove_Albion_logo.svg.png'),
    ('Burnley', 'https://upload.wikimedia.org/wikipedia/en/thumb/6/6d/Burnley_FC_Logo.svg/1200px-Burnley_FC_Logo.svg.png'),
    ('Chelsea', 'https://iconplanet.app/preview/png/64/chelsea-fc-logo--32050.png'),
    ('Crystal Palace', 'https://iconplanet.app/preview/png/64/crystal-palace-fc-logo--32057.png'),
    ('Everton', 'https://iconplanet.app/preview/png/64/everton-fc-logo--32054.png'),
    ('Fulham', 'https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg'),
    ('Leeds United', 'https://iconplanet.app/preview/png/64/leeds-united-fc-logo--32058.png'),
    ('Leicester City', 'https://iconplanet.app/preview/png/64/leicester-city-fc-logo--32056.png'),
    ('Liverpool', 'https://iconplanet.app/preview/png/64/liverpool-fc-logo--32052.png'),
    ('Manchester City', 'https://iconplanet.app/preview/png/64/manchester-city-fc-logo--32051.png'),
    ('Manchester Utd', 'https://iconplanet.app/preview/png/64/manchester-united-fc-logo--32048.png'),
    ('Newcastle Utd', 'https://iconplanet.app/preview/png/64/newcastle-united-logo--32071.png'),
    ('Norwich City', 'https://iconplanet.app/preview/png/64/norwich-fc-logo--32061.png'),
    ('Nottingham Forest', 'https://upload.wikimedia.org/wikipedia/en/thumb/e/e5/Nottingham_Forest_F.C._logo.svg/1200px-Nottingham_Forest_F.C._logo.svg.png'),
    ('Sheffield Utd', 'https://iconplanet.app/preview/png/64/sheffield-united-fc-logo--32062.png'),
    ('Southampton', 'https://iconplanet.app/preview/png/64/southampton-fc-logo--32063.png'),
    ('Tottenham', 'https://iconplanet.app/preview/png/64/tottenham-hotspur-logo--32053.png'),
    ('Watford', 'https://iconplanet.app/preview/png/64/watford-fc-logo--32067.png'),
    ('West Brom', 'https://iconplanet.app/preview/png/64/west-bromwich-albion-fc-logo--32068.png'),
    ('West Ham', 'https://iconplanet.app/preview/png/64/west-ham-united-fc-logo--32069.png'),
    ('Wolves', 'https://upload.wikimedia.org/wikipedia/en/thumb/f/fc/Wolverhampton_Wanderers.svg/1200px-Wolverhampton_Wanderers.svg.png')
]


@app.route('/')
def index():
    return render_template('index.html', teams_with_images=teams_with_images)



@app.route('/')
def home():
    teams = [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds United',
        'Leicester City', 'Liverpool', 'Manchester City', 'Manchester Utd',
        'Newcastle Utd', 'Norwich City', 'Nottingham Forest', 'Sheffield Utd',
        'Southampton', 'Tottenham', 'Watford', 'West Brom', 'West Ham', 'Wolves'
    ]
    return render_template('index.html', teams=teams)


from flask import request, jsonify

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    team1 = data['team1']
    team2 = data['team2']
    venue = data['venue']
    time = data['time']
    day = data['day']
    team1_score = team_scores[team1]

    # Preprocess the data
    input_df = preprocess_data(day, team2, venue, time, team1_score)
    column_names = input_df.head()#columns.tolist()
    print(column_names)
    # Make a prediction using the loaded MLP model
    prediction_result = mlp_model.predict(input_df)
    print("prediction :", prediction_result[0])
    # Convert the prediction result to a meaningful response (e.g., "Win", "Loss")
    prediction = "Loss!!" if prediction_result[0] == 0 else "Win!!"  # Adjust as needed

    return jsonify({'prediction': prediction})



if __name__ == '__main__':
    app.run(debug=True, port = 8000)
