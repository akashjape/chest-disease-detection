import React from "react";
import { FiLoader } from "react-icons/fi";
import "./LoadingSpinner.css";

const LoadingSpinner = () => {
  return (
    <div className="loading-spinner-container">
      <div className="spinner-content">
        <div className="spinner-animation">
          <FiLoader className="spinner-icon" />
        </div>
        <p className="spinner-title">Analyzing X-ray...</p>
        <p className="spinner-subtitle">Processing medical imaging data</p>
        <div className="progress-bar">
          <div className="progress-fill"></div>
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner;
