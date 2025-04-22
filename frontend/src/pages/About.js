// src/pages/About.js
import React from 'react';
import '../styles/About.css';

function About() {
  return (
    <div className="about-container">
      <h1>About LeetCode Recommender</h1>
      
      <section className="about-section">
        <h2>How It Works</h2>
        <p>
          LeetCode Recommender is a tool that helps programmers find the most relevant 
          LeetCode problems to solve based on their past performance and learning needs.
        </p>
        <p>
          The system analyzes your solved problems and uses advanced machine learning 
          algorithms to recommend problems that will help you improve your skills.
        </p>
      </section>
      
      <section className="about-section">
        <h2>Recommendation Types</h2>
        <div className="info-cards">
          <div className="info-card">
            <h3>Description-Based</h3>
            <p>
              Uses TF-IDF vectorization of problem descriptions to find similar problems
              that match your solving patterns and introduce new concepts.
            </p>
          </div>
          <div className="info-card">
            <h3>Solution-Based</h3>
            <p>
              Uses a graph-based approach to find problems with similar solution patterns,
              focusing on algorithmic techniques and data structures.
            </p>
          </div>
        </div>
      </section>
      
      <section className="about-section">
        <h2>AI-Powered Curation</h2>
        <p>
          Our system uses advanced AI to analyze your problem-solving history and curate 
          the top 5 problems that will be most beneficial for your learning journey.
        </p>
        <p>
          The AI considers factors like:
        </p>
        <ul>
          <li>Patterns from your solved problems</li>
          <li>Progressive difficulty levels</li>
          <li>Coverage of important data structures and algorithms</li>
          <li>Frequency in technical interviews</li>
        </ul>
      </section>
    </div>
  );
}

export default About;