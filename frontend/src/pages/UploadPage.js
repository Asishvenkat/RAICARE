import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { predictionService } from '../api/predictionService';
import LoadingSpinner from '../components/LoadingSpinner';
import './UploadPage.css';

function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  
  const navigate = useNavigate();

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setError('');
      setPrediction(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setError('');

    try {
      // Upload and get AI prediction from backend
      const response = await predictionService.uploadPrediction(selectedFile);

      if (response.status === 'success') {
        setPrediction(response.data.prediction);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'none': return '#4caf50';
      case 'mild': return '#ff9800';
      case 'moderate': return '#ff9800';
      case 'severe': return '#f44336';
      default: return '#666';
    }
  };

  const getSeverityMessage = (severity, percentage) => {
    switch (severity) {
      case 'none':
        return 'Great news! No signs of Rheumatoid Arthritis detected in your X-ray. Continue maintaining a healthy lifestyle.';
      case 'mild':
        return 'Your results show mild RA indicators. Early intervention with lifestyle changes can be very effective.';
      case 'moderate':
        return 'Your results indicate moderate RA. We recommend consulting with a rheumatologist and following our personalized care plan.';
      case 'severe':
        return 'Your results show severe RA indicators. Please consult a healthcare professional immediately for proper treatment.';
      default:
        return '';
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <h1>X-Ray Analysis</h1>
        <p className="subtitle">Upload your hand X-ray for RA detection</p>

        {error && <div className="error-message">{error}</div>}

        <div className="upload-section">
          {!preview ? (
            <div className="upload-area">
              <input
                type="file"
                id="file-input"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-input" className="upload-label">
                <div className="upload-icon">📁</div>
                <h3>Choose X-Ray Image</h3>
                <p>Click to browse or drag and drop</p>
                <p className="file-info">Supports: JPG, PNG</p>
              </label>
            </div>
          ) : (
            <div className="preview-section">
              <div className="image-preview">
                <img src={preview} alt="X-ray preview" />
              </div>
              
              {!prediction && (
                <div className="action-buttons">
                  <button 
                    onClick={() => {
                      setSelectedFile(null);
                      setPreview(null);
                      setPrediction(null);
                    }}
                    className="btn-secondary"
                  >
                    Change Image
                  </button>
                  <button 
                    onClick={handleAnalyze}
                    className="btn-primary"
                    disabled={analyzing}
                  >
                    {analyzing ? 'Analyzing...' : 'Analyze X-Ray'}
                  </button>
                </div>
              )}

              {analyzing && (
                <div className="loading-section">
                  <LoadingSpinner />
                  <p>Analyzing X-ray with AI...</p>
                </div>
              )}

              {prediction && (
                <div className="results-section">
                  <h2>Analysis Results</h2>
                  
                  <div className="result-card">
                    <div className="result-header">
                      <h3>RA Detection Score</h3>
                      <div 
                        className="percentage-badge"
                        style={{ background: getSeverityColor(prediction.severity_level) }}
                      >
                        {prediction.result_percentage}%
                      </div>
                    </div>
                    
                    <div className="severity-info">
                      <span className="severity-label">Severity Level:</span>
                      <span 
                        className="severity-value"
                        style={{ color: getSeverityColor(prediction.severity_level) }}
                      >
                        {prediction.severity_level.toUpperCase()}
                      </span>
                    </div>

                    <p className="severity-message">
                      {getSeverityMessage(prediction.severity_level, prediction.result_percentage)}
                    </p>

                    <div className="result-actions">
                      <button 
                        onClick={() => navigate('/chatbot')}
                        className="btn-primary"
                      >
                        Get Personalized Recommendations
                      </button>
                      <button 
                        onClick={() => navigate('/history')}
                        className="btn-secondary"
                      >
                        View History
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default UploadPage;
