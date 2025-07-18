import streamlit as st
import plotly.express as px
import pandas as pd
from utils import (
    load_aqi_data,
    load_weather_data,
    get_day_data,
    is_daytime,
    get_color_for_category,
    classify_aqi
)

st.set_page_config(page_title="Karachi AQI Forecast", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background:
            linear-gradient(135deg, rgba(12,27,51,0.85), rgba(26,58,106,0.85)),
            url("https://wallpapers.com/images/featured/karachi-pictures-05x5929jya0yamcv.jpg") no-repeat center center fixed;
        background-size: cover;
        color: white;
    }
    .hourly-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hourly-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 15px 10px;
        margin: 5px 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-width: 100px;
        text-align: center;
        color: white;
    }
    .hour-time { font-size: 0.9rem; font-weight: 500; margin-bottom: 8px; opacity: 0.9; }
    .hour-icon { font-size: 1.8rem; margin-bottom: 8px; }
    .aqi-value { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
    .aqi-category { font-size: 0.7rem; margin-bottom: 10px; opacity: 0.8; }
    .weather-info {
        border-top: 1px solid rgba(255,255,255,0.2);
        padding-top: 8px;
        margin-top: 8px;
    }
    .weather-item {
        font-size: 0.75rem;
        margin-bottom: 3px;
        opacity: 0.9;
    }
            
            
    .aqi-good { color: #00E676; }
    .aqi-moderate { color: #FFEB3B; }
    .aqi-sensitive { color: #FF9800; }
    .aqi-unhealthy { color: #F44336; }
    .aqi-very-unhealthy { color: #9C27B0; }
    .aqi-hazardous { color: #795548; }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
try:
    aqi_df = load_aqi_data()
    weather_df = load_weather_data()

    if aqi_df.empty:
        st.error("No AQI data available")
        st.stop()
    if weather_df.empty:
        st.warning("No weather data available - showing AQI only")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# --- Title ---
st.markdown("<h1 style='text-align:center;'>Karachi AQI Forecast</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>Get a 3-day hourly forecast of Air Quality Index (AQI).</h2>", unsafe_allow_html=True)

# --- Tabs by Date ---
unique_days = sorted(aqi_df["date"].unique())
tab_labels = [day.strftime("%A, %b %d") for day in unique_days]
tabs = st.tabs(tab_labels)

def get_aqi_class(aqi_value):
    category = classify_aqi(aqi_value)
    return {
        "Good": "aqi-good",
        "Moderate": "aqi-moderate",
        "Unhealthy for Sensitive Groups": "aqi-sensitive",
        "Unhealthy": "aqi-unhealthy",
        "Very Unhealthy": "aqi-very-unhealthy",
        "Hazardous": "aqi-hazardous"
    }.get(category, "")

# --- Main Tab Loop ---
for i, day in enumerate(unique_days):
    with tabs[i]:
        day_data = get_day_data(aqi_df, day)
        weather_day_data = get_day_data(weather_df, day) if not weather_df.empty else pd.DataFrame()

        if day_data.empty:
            st.error(f"No data available for {day.strftime('%A, %B %d')}")
            continue

        st.markdown('<div class="hourly-container">', unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-bottom: 20px;'>📅 {day.strftime('%A, %B %d, %Y')}</h3>", unsafe_allow_html=True)

        # Layout in groups of 6
        hour_blocks = [day_data.iloc[i:i+6] for i in range(0, len(day_data), 6)]
        for block in hour_blocks:
            cols = st.columns(len(block))
            for col, (_, row) in zip(cols, block.iterrows()):
                hour_label = row["timestamp"].strftime("%I %p").replace("AM", "am").replace("PM", "pm")
                category = row.get("aqi_category", classify_aqi(row["aqi"]))
                aqi_value = round(row["aqi"])
                icon = "☀️" if is_daytime(row["hour"]) else "🌙"
                

                # # Match weather
                # weather_row = weather_day_data[weather_day_data['timestamp'] == row['timestamp']]
                # weather_info = ""
                # if not weather_row.empty:
                #     w = weather_row.iloc[0]
                #     temp = w.get("temp_C", "N/A")
                #     wind = w.get("windspeed_kph", "N/A")
                #     humidity = w.get("humidity_%", "N/A")
                #     precip = w.get("precip_mm", "N/A")
                #     weather_info = f"""
                #         <div class="weather-info">
                #             <div class="weather-item">🌡️ {temp}°C</div>
                #             <div class="weather-item">💨 {wind} km/h</div>
                #             <div class="weather-item">💧 {humidity}%</div>
                #             <div class="weather-item">☔ {precip} mm</div>
                #         </div>
                #     """

                with col:
                    st.markdown(f"""
                        <div class="hourly-card">
                            <div class="hour-time">{hour_label}</div>
                            <div class="hour-icon">{icon}</div>
                            <div class="aqi-value {get_aqi_class(aqi_value)}">{aqi_value}</div>
                            <div class="aqi-category">{category}</div>
                            
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Line Chart
        st.subheader("24-Hour AQI Trend")
        chart_data = day_data.copy()
        chart_data["hour"] = chart_data["timestamp"].dt.hour
        chart_data["aqi"] = chart_data["aqi"].round()

        shapes = [
            dict(type="rect", xref="paper", yref="y", x0=0, y0=0, x1=1, y1=50, fillcolor="#009966", opacity=0.2, layer="below", line_width=0),
            dict(type="rect", xref="paper", yref="y", x0=0, y0=51, x1=1, y1=100, fillcolor="#FFDE33", opacity=0.2, layer="below", line_width=0),
            dict(type="rect", xref="paper", yref="y", x0=0, y0=101, x1=1, y1=150, fillcolor="#FF9933", opacity=0.2, layer="below", line_width=0),
            dict(type="rect", xref="paper", yref="y", x0=0, y0=151, x1=1, y1=200, fillcolor="#CC0033", opacity=0.2, layer="below", line_width=0),
            dict(type="rect", xref="paper", yref="y", x0=0, y0=201, x1=1, y1=300, fillcolor="#660099", opacity=0.2, layer="below", line_width=0),
            dict(type="rect", xref="paper", yref="y", x0=0, y0=301, x1=1, y1=500, fillcolor="#7E0023", opacity=0.2, layer="below", line_width=0)
        ]

        fig = px.line(chart_data, x="hour", y="aqi", markers=True, line_shape="spline")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            shapes=shapes,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickmode='linear'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, max(200, chart_data["aqi"].max() + 50)]),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

        # AQI Legend
        st.subheader("AQI Categories")
        legend_cols = st.columns(4)
        legend_items = [
            ("Good", "#009966", "0–50"),
            ("Moderate", "#FFDE33", "51–100"),
            ("Sensitive", "#FF9933", "101–150"),
            ("Unhealthy", "#CC0033", "151–200")
        ]
        for idx, (label, color, range_) in enumerate(legend_items):
            with legend_cols[idx]:
                st.markdown(f"""
                    <div style="background:{color};padding:10px;border-radius:8px;text-align:center;">
                        <strong>{label}</strong><br>
                        <span style="font-size:0.8rem;">{range_}</span>
                    </div>
                """, unsafe_allow_html=True)
