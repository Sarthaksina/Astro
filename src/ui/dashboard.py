#!/usr/bin/env python
"""
Cosmic Market Oracle - Main Streamlit Dashboard

This is the main web interface for the Cosmic Market Oracle, providing
interactive visualization, real-time predictions, and user management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List, Optional, Any
# import logging # Removed
from pathlib import Path
import sys
from src.utils.logger import get_logger # Added
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.ui.auth import AuthManager
from src.ui.websocket_client import WebSocketClient
from src.ui.user_preferences import UserPreferences
from src.ui.feedback_system import FeedbackSystem
from src.api.app import app as fastapi_app
from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.trading.strategy_framework import VedicAstrologyStrategy
from src.evaluation.visualization import PerformanceVisualizer
from src.llm_integration.conversation import ConversationalInterface

# Configure logging
# logging.basicConfig(level=logging.INFO) # Removed
logger = get_logger(__name__) # Changed

# Page configuration
st.set_page_config(
    page_title="Cosmic Market Oracle",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .alert-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: #e8f5e8;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CosmicMarketDashboard:
    """Main dashboard class for the Cosmic Market Oracle."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.auth_manager = AuthManager()
        self.user_preferences = UserPreferences()
        self.feedback_system = FeedbackSystem()
        self.websocket_client = WebSocketClient()
        self.planetary_calculator = PlanetaryCalculator()
        self.conversation_interface = ConversationalInterface()
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def run(self):
        """Run the main dashboard."""
        # Authentication check
        if not st.session_state.authenticated:
            self.show_login_page()
            return
        
        # Main dashboard
        self.show_main_dashboard()
    
    def show_login_page(self):
        """Show the login/registration page."""
        st.markdown("""
        <div class="main-header">
            <h1>🌟 Cosmic Market Oracle</h1>
            <p>Advanced AI-Powered Financial Astrology Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    if self.auth_manager.authenticate(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        with tab2:
            st.subheader("Create New Account")
            
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_email = st.text_input("Email Address")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")
                
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif self.auth_manager.register_user(new_username, new_email, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed. Username may already exist.")
    
    def show_main_dashboard(self):
        """Show the main dashboard interface."""
        # Header
        st.markdown(f"""
        <div class="main-header">
            <h1>🌟 Cosmic Market Oracle</h1>
            <p>Welcome back, {st.session_state.username}!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        self.show_sidebar()
        
        # Main content area
        page = st.session_state.get('current_page', 'Dashboard')
        
        if page == 'Dashboard':
            self.show_dashboard_page()
        elif page == 'Predictions':
            self.show_predictions_page()
        elif page == 'Analysis':
            self.show_analysis_page()
        elif page == 'Chat':
            self.show_chat_page()
        elif page == 'Alerts':
            self.show_alerts_page()
        elif page == 'Settings':
            self.show_settings_page()
        elif page == 'Feedback':
            self.show_feedback_page()
    
    def show_sidebar(self):
        """Show the sidebar navigation."""
        with st.sidebar:
            st.image("https://via.placeholder.com/200x100/1e3c72/ffffff?text=Cosmic+Oracle", 
                    use_column_width=True)
            
            # Navigation menu

        # In show_sidebar method:
            selected = st.selectbox(
            "Navigation",
            ["Dashboard", "Predictions", "Analysis", "Chat", "Alerts", "Settings", "Feedback"],
            index=0
            )
            st.session_state.current_page = selected
            
            # Real-time alerts
            st.subheader("🔔 Live Alerts")
            alerts = self.websocket_client.get_recent_alerts()
            
            if alerts:
                for alert in alerts[-3:]:  # Show last 3 alerts
                    st.markdown(f"""
                    <div class="alert-card">
                        <strong>{alert['type']}</strong><br>
                        {alert['message']}<br>
                        <small>{alert['timestamp']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent alerts")
            
            # Quick stats
            st.subheader("📊 Quick Stats")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Active Strategies", "3", "↑1")
            
            with col2:
                st.metric("Accuracy", "78.5%", "↑2.1%")
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.rerun()
    
    def show_dashboard_page(self):
        """Show the main dashboard page."""
        st.header("📊 Market Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="S&P 500",
                value="4,567.89",
                delta="12.34 (0.27%)",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="Prediction Accuracy",
                value="78.5%",
                delta="2.1%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                label="Active Signals",
                value="7",
                delta="2",
                delta_color="normal"
            )
        
        with col4:
            st.metric(
                label="Portfolio Return",
                value="15.2%",
                delta="3.4%",
                delta_color="normal"
            )
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Market Trend with Astrological Overlay")
            
            # Generate sample data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            prices = 4000 + np.cumsum(np.random.randn(len(dates)) * 10)
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name='S&P 500',
                line=dict(color='blue', width=2)
            ))
            
            # Add astrological events
            event_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='30D')
            event_prices = np.interp(event_dates.astype(int), dates.astype(int), prices)
            
            fig.add_trace(go.Scatter(
                x=event_dates,
                y=event_prices,
                mode='markers',
                name='Astrological Events',
                marker=dict(color='red', size=10, symbol='star')
            ))
            
            fig.update_layout(
                title="Market Trend with Astrological Events",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌟 Current Planetary Positions")
            
            # Get current planetary positions
            current_date = datetime.now()
            planets_data = self.planetary_calculator.get_all_planets(current_date)
            
            # Create planetary positions chart
            planet_names = list(planets_data.keys())
            longitudes = [planets_data[planet]['longitude'] for planet in planet_names]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=[1] * len(planet_names),
                theta=longitudes,
                mode='markers+text',
                text=planet_names,
                textposition="middle center",
                marker=dict(size=15, color=px.colors.qualitative.Set1[:len(planet_names)])
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 1]),
                    angularaxis=dict(direction="clockwise", period=360)
                ),
                title="Current Planetary Positions",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions table
        st.subheader("🔮 Recent Predictions")
        
        # Sample predictions data
        predictions_data = {
            'Date': ['2024-01-15', '2024-01-14', '2024-01-13', '2024-01-12'],
            'Symbol': ['SPY', 'QQQ', 'SPY', 'IWM'],
            'Prediction': ['Bullish', 'Bearish', 'Neutral', 'Bullish'],
            'Confidence': ['85%', '72%', '68%', '91%'],
            'Actual': ['✅ Correct', '✅ Correct', '❌ Wrong', '✅ Correct'],
            'Astrological Factor': ['Jupiter Trine', 'Mars Square', 'Mercury Retrograde', 'Venus Conjunction']
        }
        
        df = pd.DataFrame(predictions_data)
        st.dataframe(df, use_container_width=True)
    
    def show_predictions_page(self):
        """Show the predictions page."""
        st.header("🔮 Market Predictions")
        
        # Prediction controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Select Symbol", ["SPY", "QQQ", "IWM", "DIA"])
        
        with col2:
            timeframe = st.selectbox("Timeframe", ["1D", "1W", "1M", "3M"])
        
        with col3:
            if st.button("Generate Prediction", type="primary"):
                st.success("Prediction generated!")
        
        # Current prediction
        st.subheader(f"📊 Current Prediction for {symbol}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prediction chart
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            actual_prices = 450 + np.cumsum(np.random.randn(15) * 2)
            predicted_prices = actual_prices[-1] + np.cumsum(np.random.randn(15) * 2)
            
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(go.Scatter(
                x=dates[:15],
                y=actual_prices,
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Predicted prices
            fig.add_trace(go.Scatter(
                x=dates[14:],
                y=predicted_prices,
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="prediction-card">
                <h4>🎯 Prediction Summary</h4>
                <p><strong>Direction:</strong> Bullish ↗️</p>
                <p><strong>Confidence:</strong> 85%</p>
                <p><strong>Target Price:</strong> $465.50</p>
                <p><strong>Time Horizon:</strong> 2 weeks</p>
                <p><strong>Key Factor:</strong> Jupiter Trine Venus</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h5>🌟 Astrological Factors</h5>
                <ul>
                    <li>Jupiter in Taurus (Bullish)</li>
                    <li>Venus Conjunction (Moderate)</li>
                    <li>Mars Sextile (Supportive)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def show_analysis_page(self):
        """Show the analysis page."""
        st.header("📊 Market Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Technical Analysis", "Astrological Analysis", "Performance"])
        
        with tab1:
            st.subheader("📈 Technical Indicators")
            
            # Technical analysis chart
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            prices = 450 + np.cumsum(np.random.randn(100) * 2)
            sma_20 = pd.Series(prices).rolling(20).mean()
            sma_50 = pd.Series(prices).rolling(50).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=dates, y=prices, name='Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dates, y=sma_20, name='SMA 20', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dates, y=sma_50, name='SMA 50', line=dict(color='red')))
            
            fig.update_layout(title="Technical Analysis", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🌟 Astrological Correlations")
            
            # Astrological correlation matrix
            aspects = ['Conjunction', 'Opposition', 'Trine', 'Square', 'Sextile']
            correlations = np.random.rand(5, 5)
            
            fig = px.imshow(
                correlations,
                x=aspects,
                y=aspects,
                color_continuous_scale='RdBu',
                title="Aspect Correlation Matrix"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📊 Strategy Performance")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Return", "15.2%", "3.4%")
            
            with col2:
                st.metric("Sharpe Ratio", "1.85", "0.12")
            
            with col3:
                st.metric("Max Drawdown", "-5.8%", "1.2%")
            
            # Performance chart
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            returns = np.random.randn(100) * 0.02 + 0.001
            cumulative_returns = (1 + pd.Series(returns)).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Strategy Returns',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_chat_page(self):
        """Show the AI chat interface page."""
        st.header("💬 AI Assistant Chat")
        
        # Chat interface
        st.subheader("Ask the Cosmic Oracle")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>🌟 Oracle:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask about market predictions, astrological analysis, or trading strategies...",
                    placeholder="e.g., What does Jupiter's position mean for the market this week?"
                )
            
            with col2:
                submitted = st.form_submit_button("Send", type="primary")
            
            if submitted and user_input:
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Generate AI response (mock for now)
                ai_response = self.generate_ai_response(user_input)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': datetime.now().isoformat()
                })
                
                st.rerun()
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Market Summary", use_container_width=True):
                self.add_quick_response("Here's today's market summary with astrological insights...")
        
        with col2:
            if st.button("🔮 Weekly Forecast", use_container_width=True):
                self.add_quick_response("Based on planetary movements, here's the weekly forecast...")
        
        with col3:
            if st.button("⚠️ Risk Analysis", use_container_width=True):
                self.add_quick_response("Current risk factors based on astrological indicators...")
    
    def show_alerts_page(self):
        """Show the alerts and notifications page."""
        st.header("🔔 Alerts & Notifications")
        
        # Alert settings
        st.subheader("⚙️ Alert Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Market Movement Alerts", value=True)
            st.checkbox("Astrological Event Alerts", value=True)
            st.checkbox("Strategy Performance Alerts", value=False)
        
        with col2:
            st.selectbox("Alert Frequency", ["Real-time", "Hourly", "Daily"])
            st.selectbox("Delivery Method", ["Dashboard", "Email", "Both"])
        
        # Active alerts
        st.subheader("📢 Active Alerts")
        
        alerts_data = [
            {
                "Time": "10:30 AM",
                "Type": "Astrological",
                "Message": "Jupiter entering Taurus - Bullish signal for financial markets",
                "Priority": "High",
                "Status": "Active"
            },
            {
                "Time": "09:15 AM",
                "Type": "Market",
                "Message": "S&P 500 approaching resistance level at 4,580",
                "Priority": "Medium",
                "Status": "Active"
            },
            {
                "Time": "08:45 AM",
                "Type": "Strategy",
                "Message": "Vedic Strategy #1 generated BUY signal for QQQ",
                "Priority": "High",
                "Status": "Executed"
            }
        ]
        
        for alert in alerts_data:
            priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[alert["Priority"]]
            status_color = {"Active": "🟦", "Executed": "✅", "Expired": "⚫"}[alert["Status"]]
            
            st.markdown(f"""
            <div class="alert-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{priority_color} {alert['Type']} Alert</strong><br>
                        {alert['Message']}<br>
                        <small>⏰ {alert['Time']}</small>
                    </div>
                    <div>
                        {status_color} {alert['Status']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert history
        st.subheader("📜 Alert History")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("To Date", value=datetime.now())
        
        # Filter options
        alert_types = st.multiselect(
            "Filter by Type",
            ["Market", "Astrological", "Strategy", "System"],
            default=["Market", "Astrological"]
        )
        
        # Mock alert history data
        history_data = {
            'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
            'Type': np.random.choice(['Market', 'Astrological', 'Strategy'], size=len(pd.date_range(start=start_date, end=end_date, freq='D'))),
            'Count': np.random.randint(1, 10, size=len(pd.date_range(start=start_date, end=end_date, freq='D')))
        }
        
        df_history = pd.DataFrame(history_data)
        
        # Alert frequency chart
        fig = px.bar(df_history, x='Date', y='Count', color='Type', title="Alert Frequency Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    def show_settings_page(self):
        """Show the user settings and preferences page."""
        st.header("⚙️ Settings & Preferences")
        
        # User preferences
        user_prefs = self.user_preferences.get_preferences(st.session_state.username)
        
        tab1, tab2, tab3, tab4 = st.tabs(["General", "Trading", "Notifications", "Display"])
        
        with tab1:
            st.subheader("👤 General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                timezone = st.selectbox(
                    "Timezone",
                    ["UTC", "EST", "PST", "GMT"],
                    index=user_prefs.get('timezone_index', 0)
                )
                
                language = st.selectbox(
                    "Language",
                    ["English", "Spanish", "French", "German"],
                    index=user_prefs.get('language_index', 0)
                )
            
            with col2:
                auto_refresh = st.checkbox(
                    "Auto-refresh data",
                    value=user_prefs.get('auto_refresh', True)
                )
                
                refresh_interval = st.slider(
                    "Refresh interval (seconds)",
                    min_value=10,
                    max_value=300,
                    value=user_prefs.get('refresh_interval', 60)
                )
        
        with tab2:
            st.subheader("📈 Trading Preferences")
            
            col1, col2 = st.columns(2)
            
            with col1:
                default_symbol = st.selectbox(
                    "Default Symbol",
                    ["SPY", "QQQ", "IWM", "DIA"],
                    index=user_prefs.get('default_symbol_index', 0)
                )
                
                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    ["Conservative", "Moderate", "Aggressive"],
                    index=user_prefs.get('risk_tolerance_index', 1)
                )
            
            with col2:
                default_timeframe = st.selectbox(
                    "Default Timeframe",
                    ["1D", "1W", "1M", "3M"],
                    index=user_prefs.get('default_timeframe_index', 0)
                )
                
                auto_execute = st.checkbox(
                    "Auto-execute signals",
                    value=user_prefs.get('auto_execute', False)
                )
        
        with tab3:
            st.subheader("🔔 Notification Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                email_notifications = st.checkbox(
                    "Email Notifications",
                    value=user_prefs.get('email_notifications', True)
                )
                
                push_notifications = st.checkbox(
                    "Push Notifications",
                    value=user_prefs.get('push_notifications', True)
                )
            
            with col2:
                notification_frequency = st.selectbox(
                    "Notification Frequency",
                    ["Real-time", "Hourly", "Daily"],
                    index=user_prefs.get('notification_frequency_index', 0)
                )
                
                quiet_hours = st.checkbox(
                    "Enable Quiet Hours",
                    value=user_prefs.get('quiet_hours', False)
                )
        
        with tab4:
            st.subheader("🎨 Display Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                theme = st.selectbox(
                    "Theme",
                    ["Light", "Dark", "Auto"],
                    index=user_prefs.get('theme_index', 0)
                )
                
                chart_style = st.selectbox(
                    "Chart Style",
                    ["Modern", "Classic", "Minimal"],
                    index=user_prefs.get('chart_style_index', 0)
                )
            
            with col2:
                show_tooltips = st.checkbox(
                    "Show Tooltips",
                    value=user_prefs.get('show_tooltips', True)
                )
                
                compact_view = st.checkbox(
                    "Compact View",
                    value=user_prefs.get('compact_view', False)
                )
        
        # Save settings button
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            # Collect all preferences
            new_prefs = {
                'timezone_index': ["UTC", "EST", "PST", "GMT"].index(timezone),
                'language_index': ["English", "Spanish", "French", "German"].index(language),
                'auto_refresh': auto_refresh,
                'refresh_interval': refresh_interval,
                'default_symbol_index': ["SPY", "QQQ", "IWM", "DIA"].index(default_symbol),
                'risk_tolerance_index': ["Conservative", "Moderate", "Aggressive"].index(risk_tolerance),
                'default_timeframe_index': ["1D", "1W", "1M", "3M"].index(default_timeframe),
                'auto_execute': auto_execute,
                'email_notifications': email_notifications,
                'push_notifications': push_notifications,
                'notification_frequency_index': ["Real-time", "Hourly", "Daily"].index(notification_frequency),
                'quiet_hours': quiet_hours,
                'theme_index': ["Light", "Dark", "Auto"].index(theme),
                'chart_style_index': ["Modern", "Classic", "Minimal"].index(chart_style),
                'show_tooltips': show_tooltips,
                'compact_view': compact_view
            }
            
            # Save preferences
            self.user_preferences.save_preferences(st.session_state.username, new_prefs)
            st.success("Settings saved successfully!")
    
    def show_feedback_page(self):
        """Show the feedback and rating page."""
        st.header("⭐ Feedback & Ratings")
        
        # Overall satisfaction
        st.subheader("📊 Overall Experience")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            overall_rating = st.slider(
                "Overall Satisfaction",
                min_value=1,
                max_value=5,
                value=4,
                help="Rate your overall experience with the Cosmic Market Oracle"
            )
        
        with col2:
            st.markdown(f"""
            <div style="padding: 20px; text-align: center;">
                <h2>{'⭐' * overall_rating}{'☆' * (5 - overall_rating)}</h2>
                <p><strong>{['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'][overall_rating-1]}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature ratings
        st.subheader("🔧 Feature Ratings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_accuracy = st.slider("Prediction Accuracy", 1, 5, 4)
            ui_usability = st.slider("User Interface", 1, 5, 4)
            response_time = st.slider("Response Time", 1, 5, 3)
        
        with col2:
            astrological_insights = st.slider("Astrological Insights", 1, 5, 5)
            alert_system = st.slider("Alert System", 1, 5, 4)
            documentation = st.slider("Documentation", 1, 5, 3)
        
        # Written feedback
        st.subheader("💬 Written Feedback")
        
        feedback_type = st.selectbox(
            "Feedback Type",
            ["General Feedback", "Bug Report", "Feature Request", "Improvement Suggestion"]
        )
        
        feedback_text = st.text_area(
            "Your Feedback",
            placeholder="Please share your thoughts, suggestions, or report any issues...",
            height=150
        )
        
        # Contact information (optional)
        st.subheader("📧 Contact Information (Optional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            contact_email = st.text_input("Email Address (for follow-up)")
        
        with col2:
            contact_preference = st.selectbox(
                "Contact Preference",
                ["No follow-up needed", "Email me updates", "Email me when resolved"]
            )
        
        # Submit feedback
        if st.button("📤 Submit Feedback", type="primary", use_container_width=True):
            feedback_data = {
                'username': st.session_state.username,
                'timestamp': datetime.now().isoformat(),
                'overall_rating': overall_rating,
                'feature_ratings': {
                    'prediction_accuracy': prediction_accuracy,
                    'ui_usability': ui_usability,
                    'response_time': response_time,
                    'astrological_insights': astrological_insights,
                    'alert_system': alert_system,
                    'documentation': documentation
                },
                'feedback_type': feedback_type,
                'feedback_text': feedback_text,
                'contact_email': contact_email,
                'contact_preference': contact_preference
            }
            
            # Save feedback
            self.feedback_system.submit_feedback(feedback_data)
            st.success("Thank you for your feedback! We appreciate your input.")
        
        # Recent feedback summary
        st.subheader("📈 Community Feedback Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Rating", "4.2/5", "↑0.1")
        
        with col2:
            st.metric("Total Feedback", "1,247", "↑23")
        
        with col3:
            st.metric("Response Rate", "94%", "↑2%")
        
        # Feedback trends
        feedback_dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        feedback_counts = np.random.randint(10, 50, size=30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=feedback_dates,
            y=feedback_counts,
            mode='lines+markers',
            name='Daily Feedback',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Feedback Trends (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Feedback Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_ai_response(self, user_input: str) -> str:
        """Generate AI response to user input."""
        # Mock AI responses based on keywords
        user_input_lower = user_input.lower()
        
        if 'jupiter' in user_input_lower:
            return "Jupiter's current position in Taurus suggests a bullish influence on financial markets. This transit typically brings stability and growth opportunities, especially in sectors related to banking, real estate, and commodities. The energy is favorable for long-term investments."
        
        elif 'market' in user_input_lower and 'prediction' in user_input_lower:
            return "Based on current astrological configurations, I see a moderately bullish trend for the next 2 weeks. Key factors include Venus in a favorable aspect to Mars, suggesting increased market confidence. However, watch for potential volatility around the upcoming Mercury retrograde period."
        
        elif 'risk' in user_input_lower:
            return "Current risk factors include Mars approaching a square aspect with Saturn, which could create market tension in the next 5-7 days. I recommend reducing position sizes and maintaining higher cash levels during this period. The risk should subside once Mars moves past this aspect."
        
        elif 'strategy' in user_input_lower:
            return "For the current astrological climate, I recommend a balanced approach: 60% equities focused on stable sectors (utilities, consumer staples), 30% in growth sectors that benefit from current planetary transits, and 10% cash for opportunities. The Vedic Nakshatra analysis suggests favorable timing for entries on Monday and Thursday."
        
        else:
            return "I understand you're asking about market conditions. Based on current planetary positions and market data, I can provide insights on timing, risk assessment, and strategic recommendations. Could you be more specific about what aspect you'd like me to analyze?"