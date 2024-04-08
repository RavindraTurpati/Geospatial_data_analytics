from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap
app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv(r"C:/datasets/vizag_lat_lon.csv")
df2 = df.dropna()
dummies = pd.get_dummies(df2['location'])
dummies = dummies.astype(int)
df4 = pd.concat((df2, dummies), axis=1)
#df4 = df3.drop('location', axis=1)
df5 = df4.copy()
df5.columns = df5.columns.str.replace(' ', '_')
x = df5.drop(['price', 'Latitude', 'Longitude','location'], axis=1)
y = df5['price']

# Train the linear regression model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
model = LinearRegression()
model.fit(x_train, y_train)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Function to get predictions by location
def get_predictions(area, bhk, bath):
    locations = x.columns[3:]
    predictions_by_location = {}
    for location in locations:
        p = np.zeros(len(x.columns))
        p[0] = area
        p[1] = bhk
        p[2] = bath
        loc_index = x.columns.get_loc(location)
        p[loc_index] = 1
        prediction = model.predict([p])[0]
        predictions_by_location[location] = prediction
    return predictions_by_location

# Load the HTML template
@app.route('/xxx')
def home():
    return render_template('index.html')

# Handle form submission
@app.route('/map143', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    # Get predictions
    predicted_prices = get_predictions(area, bhk, bath)

    # Create DataFrame with predicted prices
    df6 = pd.DataFrame(list(predicted_prices.values()), columns=['Predicted_Price'])
    df6 = pd.DataFrame(df6.values.repeat(5, axis=0), columns=df6.columns)
    df7 = pd.concat((df6, df5), axis=1)

    # KMeans clustering
    kmeans = KMeans(n_clusters=5)
    df7['cluster'] = kmeans.fit_predict(df7[['Predicted_Price']])

    # Create map
    map_vishakapatnam = folium.Map(location=[17.6868, 83.2185], zoom_start=12)

    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df7.iterrows()]
    HeatMap(heat_data).add_to(map_vishakapatnam)
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Add markers to the map
    for index, row in df7.iterrows():
        cluster_index = int(row['cluster'] % len(cluster_colors))
        cluster_color = cluster_colors[cluster_index]
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['location']}<br>{row['Predicted_Price']}<br>Cluster: {row['cluster']}",
            icon=folium.Icon(color=cluster_color)
        ).add_to(map_vishakapatnam)
    cluster_counts = df7['cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        cluster_data = df7[df7['cluster'] == cluster]
        cluster_center = [cluster_data['Latitude'].mean(), cluster_data['Longitude'].mean()]
        min_price = cluster_data['Predicted_Price'].min()
        max_price = cluster_data['Predicted_Price'].max()
        price_range_text = f"Price Range: {min_price:.2f} - {max_price:.2f}"
        popup_text = f"Cluster: {cluster}<br>Number of Data Points: {count}<br>{price_range_text}"

        # Create a custom marker with cluster number inside it
        folium.Marker(

            location=cluster_center,
            popup=popup_text,
            icon=folium.Icon(color='white')  # Set marker color to white
        ).add_to(map_vishakapatnam)

    # Save the map
    map_vishakapatnam.save("templates/map143.html")

    return render_template('map143.html', map_created=True)

if __name__ == '__main__':
    app.run(debug=True,port=5003)
