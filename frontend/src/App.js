import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { FiUpload, FiX } from "react-icons/fi";
import Results from "./components/Results";
import LoadingSpinner from "./components/LoadingSpinner";
import "./App.css";

let API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
// Ensure the API URL is browser-accessible (not 0.0.0.0)
if (API_URL.includes("0.0.0.0")) {
  API_URL = API_URL.replace("0.0.0.0", "127.0.0.1");
}
if (API_URL.includes("127.0.0.1") && !API_URL.includes("localhost")) {
  API_URL = API_URL.replace("127.0.0.1", "localhost");
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [responsePayload, setResponsePayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isMockMode, setIsMockMode] = useState(false);

  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const res = await axios.get(`${API_URL}/health`);
        setIsMockMode(!res.data.model_loaded);
      } catch {
        setIsMockMode(true);
      }
    };
    checkModelStatus();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    if (!acceptedFiles.length) return;
    const file = acceptedFiles[0];
    setSelectedFile(file);
    setError(null);
    setResponsePayload(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    maxFiles: 1,
  });

  const handlePredict = async () => {
    if (!selectedFile) return;
    try {
      setLoading(true);
      setError(null);
      const formData = new FormData();
      formData.append("file", selectedFile);
      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResponsePayload(res.data || null);
    } catch (err) {
      const errorDetail =
        err?.response?.data?.detail || err?.message || "Analysis failed";
      setError(errorDetail);
      setResponsePayload({
        success: false,
        model_status: errorDetail,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setResponsePayload(null);
    setError(null);
  };

  return (
    <div className="xray-app">
      {/* HEADER */}
      <header className="xray-header">
        <div className="header-content">
          <div className="header-title">
            <div className="logo-icon">🫁</div>
            <div>
              <h1>Chest X-Ray Analysis</h1>
              <p>AI-Powered Diagnostic Assistant</p>
            </div>
          </div>
          <div className={`model-status ${isMockMode ? "demo" : "live"}`}>
            {isMockMode ? "📊 Demo Mode" : "✅ Live Model"}
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="xray-main">
        {/* LEFT PANEL - UPLOAD & CONTROL */}
        <section className="upload-section">
          <div className="upload-card">
            <h2>Upload Scan</h2>
            <p className="section-desc">
              Select a frontal chest X-ray (PA/AP view)
            </p>

            {/* DROPZONE */}
            <div
              {...getRootProps()}
              className={`medical-dropzone ${isDragActive ? "active" : ""} ${preview ? "has-preview" : ""}`}
            >
              <input {...getInputProps()} />
              {!preview ? (
                <>
                  <FiUpload className="dropzone-icon" />
                  <p className="dropzone-text">
                    {isDragActive ? "Drop image here" : "Drag & drop or click"}
                  </p>
                  <span className="dropzone-hint">JPG, PNG • max 10MB</span>
                </>
              ) : (
                <div className="preview-container">
                  <img
                    src={preview}
                    alt="scan preview"
                    className="preview-image"
                  />
                  <div className="preview-overlay">
                    <p className="filename">{selectedFile?.name}</p>
                  </div>
                </div>
              )}
            </div>

            {/* ERROR MESSAGE */}
            {error && (
              <div className="alert alert-error">
                <FiX className="alert-icon" />
                <div className="alert-content">
                  <p className="alert-title">Upload Error</p>
                  <p className="alert-msg">{error}</p>
                </div>
              </div>
            )}

            {/* ACTION BUTTONS */}
            <div className="button-group">
              <button
                onClick={handlePredict}
                disabled={!selectedFile || loading}
                className="btn-primary"
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>
              <button onClick={handleClear} className="btn-secondary">
                Clear
              </button>
            </div>

            {/* DISCLAIMER */}
            <div className="disclaimer">
              <p>⚠️ For research & evaluation. Not for clinical diagnosis.</p>
            </div>
          </div>
        </section>

        {/* RIGHT PANEL - RESULTS */}
        <section className="results-section">
          <div className="results-card">
            {loading && <LoadingSpinner />}
            {!loading && responsePayload && (
              <Results predictions={responsePayload} />
            )}
            {!loading && !responsePayload && (
              <div className="results-empty">
                <div className="empty-icon">📋</div>
                <p>No analysis yet</p>
                <span>Upload an X-ray to begin</span>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="xray-footer">
        <p>Clinical-grade AI for research purposes • Version 1.0</p>
      </footer>
    </div>
  );
}

export default App;
