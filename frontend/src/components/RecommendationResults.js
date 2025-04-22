// src/components/RecommendationResults.js
import React from 'react';
import '../styles/RecommendationResults.css';

function RecommendationResults({ results, isLoading }) {
  const renderTopRecommendations = () => {
    if (!results.top_recommendations) return null;
    
    return (
      <div className="top-recommendations">
        <h3>Top 5 AI-Curated Recommendations</h3>
        <div className="ai-analysis">
          <pre>{results.top_recommendations}</pre>
        </div>
      </div>
    );
  };

  const renderAllRecommendations = () => {
    if (!results.recommendations || results.recommendations.length === 0) {
      return <p>No recommendations available.</p>;
    }
    
    return (
      <div className="all-recommendations">
        <h3>All 15 Recommendations</h3>
        <div className="recommendation-grid">
          {results.recommendations.map((rec, index) => (
            <div key={index} className="recommendation-card">
              <h4>{rec.title}</h4>
              <div className="rec-details">
                <span className={`difficulty ${rec.difficulty.toLowerCase()}`}>
                  {rec.difficulty}
                </span>
                <span className="similarity">
                  Match: {(rec.similarity * 100).toFixed(1)}%
                </span>
              </div>
              <p className="rec-description">
                {rec.content.substring(0, 100)}...
              </p>
              <a 
                href={`https://leetcode.com/problems/${rec.slug || ''}`}
                target="_blank" 
                rel="noopener noreferrer"
                className="view-problem"
              >
                View Problem
              </a>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="recommendation-results">
      {isLoading ? (
        <div className="loading-results">
          <div className="spinner"></div>
          <p>Generating recommendations...</p>
        </div>
      ) : results.error ? (
        <div className="error-container">
          <h3>Error</h3>
          <p>{results.error}</p>
        </div>
      ) : (
        <>
          {renderTopRecommendations()}
          {renderAllRecommendations()}
        </>
      )}
    </div>
  );
}

export default RecommendationResults;