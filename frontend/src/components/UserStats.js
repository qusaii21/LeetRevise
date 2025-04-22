// src/components/UserStats.js
import React from 'react';
import '../styles/UserStats.css';

function UserStats({ userData }) {
  if (!userData || !userData.username) {
    return null;
  }

  return (
    <div className="user-stats">
      <h3>User Statistics</h3>
      <div className="stats-container">
        <div className="stat-item">
          <span className="stat-label">Username:</span>
          <span className="stat-value">{userData.username}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total Solved:</span>
          <span className="stat-value">{userData.total_solved}</span>
        </div>
      </div>
    </div>
  );
}

export default UserStats;