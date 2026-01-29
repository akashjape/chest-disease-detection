import React from "react";
import { FiTrendingUp, FiAlertCircle } from "react-icons/fi";
import "./Results.css";

const Results = ({ predictions }) => {
  const topPredictions = predictions?.top_predictions || [];
  const allPredictions = predictions?.predictions || {};
  const isValidUpload = predictions?.success !== false;
  const modelStatus = predictions?.model_status || "";
  const isMockMode = predictions?.is_mock === true;

  // Check if this is an invalid upload (non-xray image)
  const isInvalidUpload =
    !isValidUpload ||
    modelStatus.toLowerCase().includes("invalid") ||
    modelStatus.toLowerCase().includes("does not appear") ||
    modelStatus.toLowerCase().includes("not a") ||
    modelStatus.toLowerCase().includes("frontal chest");

  if (isInvalidUpload) {
    return (
      <div className="results-error">
        <FiAlertCircle className="error-icon" />
        <h3 className="error-title">❌ Not a Chest X-Ray</h3>
        <p className="error-message">
          {modelStatus || "The uploaded image is not a chest X-ray radiograph."}
        </p>
        <p className="error-hint">
          📋 Please upload a valid frontal chest radiograph (PA or AP view)
          <br />
          ✓ Accepted: Medical X-ray images only
          <br />✗ Not accepted: Documents, photos, screenshots, or other images
        </p>
      </div>
    );
  }

  return (
    <div className="results-container">
      {/* MODEL STATUS BADGE */}
      {isMockMode && (
        <div className="status-badge demo">
          <span>📊 Demo Mode</span>
          <p className="badge-text">Mock predictions for testing</p>
        </div>
      )}
      {!isMockMode && (
        <div className="status-badge live">
          <span>✅ Live Model</span>
          <p className="badge-text">Real predictions from trained model</p>
        </div>
      )}

      {/* TOP 3 PREDICTIONS */}
      <div className="predictions-section">
        <div className="section-header">
          <FiTrendingUp className="section-icon" />
          <h2>Top Predictions</h2>
        </div>

        <div className="predictions-grid">
          {topPredictions.slice(0, 3).map((pred, index) => (
            <div key={index} className={`prediction-card rank-${index + 1}`}>
              <div className="rank-badge">{index + 1}</div>
              <div className="card-content">
                <h4 className="disease-name">{pred.disease}</h4>
                <div className="confidence-display">
                  <span className="confidence-value">
                    {(pred.probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.min(pred.probability * 100, 100)}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ALL PREDICTIONS TABLE */}
      <div className="all-predictions-section">
        <div className="section-header">
          <h3>Complete Analysis</h3>
        </div>
        <div className="predictions-table">
          {Object.entries(allPredictions)
            .sort(([, a], [, b]) => b - a)
            .map(([disease, probability]) => (
              <div key={disease} className="table-row">
                <span className="disease-col">{disease}</span>
                <div className="bar-col">
                  <div className="mini-bar">
                    <div
                      className="mini-fill"
                      style={{ width: `${Math.min(probability * 100, 100)}%` }}
                    ></div>
                  </div>
                </div>
                <span className="percentage-col">
                  {(probability * 100).toFixed(2)}%
                </span>
              </div>
            ))}
        </div>
      </div>

      {/* DISCLAIMER */}
      <div className="results-disclaimer">
        <FiAlertCircle className="disclaimer-icon" />
        <div className="disclaimer-content">
          <p className="disclaimer-title">Clinical Disclaimer</p>
          <p className="disclaimer-text">
            These predictions are for research and educational purposes only. Do
            not use for clinical diagnosis. Always consult qualified healthcare
            professionals.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Results;
