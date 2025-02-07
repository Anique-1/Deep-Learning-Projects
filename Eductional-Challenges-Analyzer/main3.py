import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load pre-trained sentiment analyzer for analyzing education feedback
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize geocoder
geolocator = Nominatim(user_agent="edu_analysis_app")

class EducationChallengeAnalyzer:
    def __init__(self):
        # Pre-defined challenges and solutions database
        self.challenges_db = {
            'infrastructure': {
                'indicators': ['internet', 'computers', 'electricity', 'building'],
                'solutions': [
                    'Implement solar power systems for reliable electricity',
                    'Establish mobile computer labs',
                    'Partner with ISPs for internet connectivity',
                    'Develop offline learning resources'
                ]
            },
            'teaching_quality': {
                'indicators': ['teacher', 'training', 'qualification', 'methodology'],
                'solutions': [
                    'Provide online teacher training programs',
                    'Implement mentor-mentee systems',
                    'Create professional development workshops',
                    'Establish teacher resource centers'
                ]
            },
            'accessibility': {
                'indicators': ['distance', 'transportation', 'remote', 'access'],
                'solutions': [
                    'Develop mobile education units',
                    'Implement distance learning programs',
                    'Establish community learning centers',
                    'Create transportation support systems'
                ]
            }
        }
    
    def analyze_text(self, text):
        """Analyze education-related text to identify challenges and sentiment"""
        sentiment = sentiment_analyzer(text)[0]
        
        identified_challenges = {}
        for challenge, data in self.challenges_db.items():
            score = sum(1 for indicator in data['indicators'] if indicator.lower() in text.lower())
            if score > 0:
                identified_challenges[challenge] = {
                    'score': score,
                    'solutions': data['solutions']
                }
        
        return identified_challenges, sentiment

def get_location_data(location_name):
    """Get geographical coordinates for a location"""
    try:
        location = geolocator.geocode(location_name)
        return location.latitude, location.longitude
    except:
        return None, None

def create_map(locations_data):
    """Create an interactive map with education challenges"""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    for loc in locations_data:
        if loc['lat'] and loc['lon']:
            folium.Marker(
                [loc['lat'], loc['lon']],
                popup=f"Location: {loc['name']}<br>Main Challenge: {loc['main_challenge']}"
            ).add_to(m)
    
    return m

def main():
    st.title("Global Educational Challenges Analyzer")
    
    # Initialize analyzer
    analyzer = EducationChallengeAnalyzer()
    
    # Sidebar for location input
    st.sidebar.header("Location Information")
    location = st.sidebar.text_input("Enter Location (City/Country):")
    
    # Main content area
    st.header("Educational Challenge Analysis")
    
    # Text input for educational situation description
    situation_text = st.text_area(
        "Describe the educational situation and challenges in your area:",
        height=150
    )
    
    if st.button("Analyze Challenges"):
        if location and situation_text:
            # Get location coordinates
            lat, lon = get_location_data(location)
            
            if lat and lon:
                # Analyze challenges
                challenges, sentiment = analyzer.analyze_text(situation_text)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Sentiment analysis
                st.write("Overall Situation Assessment:", 
                        "🟢 Positive" if sentiment['label'] == 'POSITIVE' else "🔴 Negative")
                
                # Display challenges and solutions
                if challenges:
                    for challenge, details in challenges.items():
                        st.write(f"\n**Challenge Area: {challenge.replace('_', ' ').title()}**")
                        st.write("Recommended Solutions:")
                        for solution in details['solutions']:
                            st.write(f"- {solution}")
                
                    # Create and display map
                    locations_data = [{
                        'name': location,
                        'lat': lat,
                        'lon': lon,
                        'main_challenge': max(challenges.items(), key=lambda x: x[1]['score'])[0]
                    }]
                    
                    st.subheader("Geographic Visualization")
                    map_data = create_map(locations_data)
                    folium_static(map_data)
                    
                else:
                    st.warning("No specific challenges identified. Please provide more detailed information.")
            else:
                st.error("Could not find the specified location. Please check the location name.")
        else:
            st.warning("Please provide both location and situation description.")

if __name__ == "__main__":
    main()