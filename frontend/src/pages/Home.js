// src/pages/Home.js
import React, { useState } from 'react';
import UserForm from '../components/UserForm';
import '../styles/Home.css';

function Home() {
  const [userData, setUserData] = useState(null);

  return (
    <div className="home-container">
      <div className="hero-section">
        <h1>LeetCode Question Recommender</h1>
        <p>Get personalized problem recommendations based on your LeetCode history</p>
      </div>

      <div className="features-section">
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Smart Analysis</h3>
          <p>Analyzes your solved problems to find patterns and areas for improvement</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🔍</div>
          <h3>Custom Recommendations</h3>
          <p>Choose between description-based or solution-based recommendations</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🧠</div>
          <h3>AI-Powered</h3>
          <p>Uses AI to curate the most relevant problems for your skill level</p>
        </div>
      </div>

      <div className="form-section">
        <h2>Enter your LeetCode username to get started</h2>
        <UserForm setUserData={setUserData} />
      </div>
    </div>
  );
}

export default Home;