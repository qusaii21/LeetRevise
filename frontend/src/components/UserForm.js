// src/components/UserForm.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/UserForm.css';

function UserForm({ setUserData }) {
  const [username, setUsername] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim()) {
      setError('Please enter a valid username');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await axios.get(`/fetch-user?username=${username}`);
      
      if (response.data.total_solved === 0) {
        setError('No solved problems found for this user');
        setIsLoading(false);
        return;
      }

      setUserData(response.data);
      // Redirect to recommendations page
      navigate('/recommendations', { 
        state: { 
          userData: response.data
        }
      });
    } catch (error) {
      console.error('Error fetching user data:', error);
      setError(error.response?.data?.error || 'Failed to fetch user data');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="user-form-container">
      <form className="user-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username">LeetCode Username</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter your LeetCode username"
            required
          />
        </div>
        
        <button type="submit" className="submit-btn" disabled={isLoading}>
          {isLoading ? 'Loading...' : 'Get Recommendations'}
        </button>
        
        {error && <div className="error-message">{error}</div>}
      </form>
    </div>
  );
}

export default UserForm;