import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { predictionService } from '../api/predictionService';
import LoadingSpinner from '../components/LoadingSpinner';
import './DashboardPage.css';

function DashboardPage() {
  const { user } = useAuth();
  const [latestPrediction, setLatestPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadLatestPrediction();
  }, []);

  const loadLatestPrediction = async () => {
    try {
      const response = await predictionService.getLatest();
      if (response.status === 'success' && response.data.prediction) {
        setLatestPrediction(response.data.prediction);
      }
    } catch (error) {
      console.error('Failed to load prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'mild': return '#4caf50';
      case 'moderate': return '#ff9800';
      case 'severe': return '#f44336';
      default: return '#666';
    }
  };

  const getRecommendations = (severity) => {
    switch (severity) {
      case 'mild':
        return [
          '🥗 Focus on anti-inflammatory foods',
          '🏃 30 minutes walking daily',
          '😴 Get 7-8 hours of sleep',
          '🧘 Practice stress management',
        ];
      case 'moderate':
        return [
          '🥗 Strict Mediterranean diet',
          '🏊 Water aerobics 4-5 days/week',
          '🌡️ Use hot/cold therapy',
          '⚖️ Maintain healthy weight',
        ];
      case 'severe':
        return [
          '🥗 Plant-based anti-inflammatory diet',
          '🏥 Regular rheumatologist visits',
          '💊 Follow medication strictly',
          '😴 Prioritize rest (8-9 hours)',
        ];
      default:
        return [];
    }
  };

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="loading-container">
          <LoadingSpinner />
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        <div className="dashboard-header">
          <div>
            <h1>Welcome back, {user?.username}! 👋</h1>
            <p>Here's your RA health overview</p>
          </div>
        </div>

        <div className="dashboard-grid">
          {latestPrediction ? (
            <>
              <div className="card latest-result">
                <h2>Latest RA Detection</h2>
                <div className="result-display">
                  <div 
                    className="percentage-circle"
                    style={{ borderColor: getSeverityColor(latestPrediction.severity_level) }}
                  >
                    <span className="percentage">{latestPrediction.result_percentage}%</span>
                    <span className="label">RA Score</span>
                  </div>
                  <div className="result-info">
                    <div className="severity-badge" style={{ background: getSeverityColor(latestPrediction.severity_level) }}>
                      {latestPrediction.severity_level.toUpperCase()}
                    </div>
                    <p className="result-date">
                      Analyzed on {new Date(latestPrediction.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>

              <div className="card recommendations">
                <h2>Personalized Recommendations</h2>
                <ul className="recommendation-list">
                  {getRecommendations(latestPrediction.severity_level).map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
                <button 
                  onClick={() => navigate('/chatbot')}
                  className="btn-primary"
                >
                  Get More Recommendations
                </button>
              </div>
            </>
          ) : (
            <div className="card no-data">
              <div className="no-data-content">
                <div className="no-data-icon">📊</div>
                <h2>No Data Yet</h2>
                <p>Upload your first X-ray to get started with RA detection and personalized recommendations</p>
                <button 
                  onClick={() => navigate('/upload')}
                  className="btn-primary"
                >
                  Upload X-Ray Now
                </button>
              </div>
            </div>
          )}

          <div className="card quick-actions">
            <h2>Quick Actions</h2>
            <div className="actions-grid">
              <button 
                onClick={() => navigate('/upload')}
                className="action-btn"
              >
                <span className="action-icon">📤</span>
                <span>Upload X-Ray</span>
              </button>
              <button 
                onClick={() => navigate('/chatbot')}
                className="action-btn"
              >
                <span className="action-icon">💬</span>
                <span>AI Assistant</span>
              </button>
              <button 
                onClick={() => navigate('/history')}
                className="action-btn"
              >
                <span className="action-icon">📊</span>
                <span>View History</span>
              </button>
            </div>
          </div>

          <div className="card health-tips">
            <h2>Daily Health Tip</h2>
            <div className="tip-content">
              <div className="tip-icon">💡</div>
              <p>
                <strong>Stay Hydrated!</strong> Drinking 8 glasses of water daily helps reduce 
                inflammation and keeps your joints lubricated. Add lemon for extra anti-inflammatory benefits.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
