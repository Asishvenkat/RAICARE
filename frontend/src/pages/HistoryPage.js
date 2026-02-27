import React, { useState, useEffect } from 'react';
import { predictionService } from '../api/predictionService';
import LoadingSpinner from '../components/LoadingSpinner';
import './HistoryPage.css';

function HistoryPage() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await predictionService.getHistory(20);
      if (response.status === 'success') {
        setPredictions(response.data.predictions);
      }
    } catch (error) {
      console.error('Failed to load history:', error);
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

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="history-page">
        <div className="loading-container">
          <LoadingSpinner />
        </div>
      </div>
    );
  }

  return (
    <div className="history-page">
      <div className="history-container">
        <div className="history-header">
          <h1>Prediction History</h1>
          <p>Track your RA detection results over time</p>
        </div>

        {predictions.length === 0 ? (
          <div className="no-history">
            <div className="no-history-icon">📊</div>
            <h2>No History Yet</h2>
            <p>Your prediction history will appear here once you upload X-rays</p>
          </div>
        ) : (
          <>
            <div className="stats-overview">
              <div className="stat-card">
                <div className="stat-number">{predictions.length}</div>
                <div className="stat-label">Total Scans</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">
                  {predictions[0]?.result_percentage.toFixed(1)}%
                </div>
                <div className="stat-label">Latest Score</div>
              </div>
              <div className="stat-card">
                <div 
                  className="stat-number"
                  style={{ color: getSeverityColor(predictions[0]?.severity_level) }}
                >
                  {predictions[0]?.severity_level.toUpperCase()}
                </div>
                <div className="stat-label">Current Severity</div>
              </div>
            </div>

            <div className="timeline">
              {predictions.map((prediction, idx) => (
                <div key={prediction.id} className="timeline-item">
                  <div className="timeline-marker" style={{ background: getSeverityColor(prediction.severity_level) }}></div>
                  <div className="timeline-content">
                    <div className="timeline-header">
                      <h3>{formatDate(prediction.timestamp)}</h3>
                      <div 
                        className="severity-badge"
                        style={{ background: getSeverityColor(prediction.severity_level) }}
                      >
                        {prediction.severity_level}
                      </div>
                    </div>
                    <div className="timeline-body">
                      <div className="prediction-details">
                        <div className="detail-item">
                          <span className="detail-label">RA Score:</span>
                          <span className="detail-value">{prediction.result_percentage}%</span>
                        </div>
                        <div className="xray-preview">
                          <img src={prediction.image_url} alt={`X-ray ${idx + 1}`} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default HistoryPage;
