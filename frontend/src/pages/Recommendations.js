
// src/pages/Recommendations.js
import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import UserStats from '../components/UserStats';
import RecommendationOptions from '../components/RecommendationOptions';
import RecommendationResults from '../components/RecommendationResults';
import '../styles/Recommendations.css';

function Recommendations() {
  const location = useLocation();
  const navigate = useNavigate();
  const [userData, setUserData] = useState(null);
  const [options, setOptions] = useState({
    recommendationType: 'description',
    useGemini: true
  });
  const [results, setResults] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Load user data from location state or redirect to home
  useEffect(() => {
    if (location.state?.userData) {
      setUserData(location.state.userData);
      // Generate recommendations immediately with default settings
      generateRecommendations(location.state.userData, options);
    } else {
      navigate('/');
    }
  }, [location.state, navigate]);

  const handleOptionsChange = (newOptions) => {
    setOptions({
      ...options,
      ...newOptions
    });
    
    if (userData) {
      generateRecommendations(userData, {
        ...options,
        ...newOptions
      });
    }
  };

 
  const generateRecommendations = async (user, optionsToUse) => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/recommend', {
        username: user.username,
        solved_slugs: user.solved_problems,
        recommendation_type: optionsToUse.recommendationType,
        use_gemini: optionsToUse.useGemini
      });
      
      setResults(response.data);
    } catch (error) {
      console.error('Error generating recommendations:', error);
      setError(error.response?.data?.error || 'Failed to generate recommendations');
      setResults({ error: error.response?.data?.error || 'Failed to generate recommendations' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="recommendations-container">
      <h1>Your LeetCode Recommendations</h1>
      
      {userData ? (
        <>
          <UserStats userData={userData} />
          <RecommendationOptions 
            onOptionsChange={handleOptionsChange} 
            isLoading={isLoading} 
          />
          <RecommendationResults 
            results={results} 
            isLoading={isLoading} 
          />
        </>
      ) : (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading user data...</p>
        </div>
      )}
    </div>
  );
}

export default Recommendations;
