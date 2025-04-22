// src/components/RecommendationOptions.js
import React from 'react';
import '../styles/RecommendationOptions.css';

function RecommendationOptions({ onOptionsChange, isLoading }) {
  const [recommendationType, setRecommendationType] = React.useState('description');
  const [useGemini, setUseGemini] = React.useState(true);

  const handleTypeChange = (e) => {
    const newType = e.target.value;
    setRecommendationType(newType);
    onOptionsChange({ recommendationType: newType, useGemini });
  };

  const handleGeminiChange = (e) => {
    const newValue = e.target.checked;
    setUseGemini(newValue);
    onOptionsChange({ recommendationType, useGemini: newValue });
  };

  return (
    <div className="recommendation-options">
      <h3>Recommendation Options</h3>
      <div className="options-container">
        <div className="option-group">
          <label>Recommendation Type:</label>
          <div className="radio-group">
            <label>
              <input
                type="radio"
                value="description"
                checked={recommendationType === 'description'}
                onChange={handleTypeChange}
                disabled={isLoading}
              />
              Based on Question Descriptions
            </label>
            <label>
              <input
                type="radio"
                value="solution"
                checked={recommendationType === 'solution'}
                onChange={handleTypeChange}
                disabled={isLoading}
              />
              Based on Solutions
            </label>
          </div>
        </div>

        <div className="option-group">
          <label>
            <input
              type="checkbox"
              checked={useGemini}
              onChange={handleGeminiChange}
              disabled={isLoading}
            />
            Use AI to select the best 5 recommendations
          </label>
        </div>
      </div>
    </div>
  );
}

export default RecommendationOptions;