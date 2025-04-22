// src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Recommendations from './pages/Recommendations';
import About from './pages/About';
import './App.css';

// Set base URL for axios
axios.defaults.baseURL = 'http://localhost:5000/api';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [serverStatus, setServerStatus] = useState({});

  useEffect(() => {
    // Check server health when app loads
    const checkServerHealth = async () => {
      try {
        const response = await axios.get('/health');
        setServerStatus(response.data);
        setIsLoading(false);
      } catch (error) {
        console.error('Server health check failed:', error);
        setServerStatus({ status: 'error', message: 'Cannot connect to server' });
        setIsLoading(false);
      }
    };

    checkServerHealth();
  }, []);

  return (
    <Router>
      <div className="App">
        <Navbar />
        
        {isLoading ? (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Loading application...</p>
          </div>
        ) : !serverStatus.status || serverStatus.status !== 'ok' ? (
          <div className="error-container">
            <h2>Server Connection Error</h2>
            <p>Cannot connect to the backend server. Please make sure it's running.</p>
            <code>{JSON.stringify(serverStatus)}</code>
          </div>
        ) : (
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/recommendations" element={<Recommendations />} />
            <Route path="/about" element={<About />} />
          </Routes>
        )}
      </div>
    </Router>
  );
}

export default App;