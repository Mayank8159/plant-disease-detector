"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type PredictionResponse = {
  prediction: string;
  confidence: number;
  threshold: number;
};

type HistoryItem = {
  id: string;
  prediction: string;
  confidence: number;
  threshold: number;
  fileName: string;
  createdAt: string;
  imageDataUrl?: string;
  mimeType?: string;
  sizeBytes?: number;
  lastModified?: number;
};

type HistoryFilter = "all" | "strong" | "inconclusive";

type PredictionDetails = {
  plant: string;
  condition: string;
  isHealthy: boolean;
  note: string;
};

type NextStepAdvice = {
  action: "Monitor" | "Isolate" | "Retake image";
  detail: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.trim() ||
  (process.env.NODE_ENV === "production"
    ? "https://plant-disease-detector-tmla.onrender.com"
    : "http://127.0.0.1:8000");
const HISTORY_STORAGE_KEY = "plant-detector-history";
const HISTORY_LIMIT = 10;
const OFFLINE_WAKE_CHECK_SECONDS = 60;
const ONLINE_HEALTH_POLL_MS = 50_000;

function toReadableLabel(value: string): string {
  return value
    .replace(/___/g, " - ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parsePredictionDetails(prediction: string): PredictionDetails {
  if (prediction === "Inconclusive") {
    return {
      plant: "Unknown",
      condition: "Inconclusive",
      isHealthy: false,
      note: "Confidence is below threshold. Try a clearer single-leaf image with better lighting.",
    };
  }

  const [plantPart = "Unknown", conditionPart = "Unknown"] = prediction.split("___");
  const plant = toReadableLabel(plantPart);
  const condition = toReadableLabel(conditionPart);
  const isHealthy = /healthy/i.test(conditionPart);

  return {
    plant,
    condition,
    isHealthy,
    note: isHealthy
      ? "Leaf appears healthy for this plant category based on the uploaded image."
      : "Disease-like visual patterns detected. Confirm with an agronomy expert for treatment steps.",
  };
}

function getNextStepAdvice(result: PredictionResponse, details: PredictionDetails): NextStepAdvice {
  if (result.confidence < result.threshold || details.condition === "Inconclusive") {
    return {
      action: "Retake image",
      detail: "Capture one leaf in daylight, avoid blur, and keep background uncluttered.",
    };
  }

  if (details.isHealthy) {
    return {
      action: "Monitor",
      detail: "No immediate action needed. Recheck leaves in 3 to 5 days for new spots.",
    };
  }

  return {
    action: "Isolate",
    detail: "Separate the affected plant if possible and inspect nearby leaves to reduce spread.",
  };
}

function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString([], {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "short",
  });
}

function toConfidencePercent(value: number): number {
  return Math.max(0, Math.min(100, value * 100));
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error("Could not read selected image."));
    reader.readAsDataURL(file);
  });
}

async function dataUrlToFile(dataUrl: string, fileName: string, mimeType = "image/jpeg"): Promise<File> {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return new File([blob], fileName, {
    type: blob.type || mimeType,
    lastModified: Date.now(),
  });
}

export default function Home() {
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [wakeCountdown, setWakeCountdown] = useState<number | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyFilter, setHistoryFilter] = useState<HistoryFilter>("all");
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw) as HistoryItem[];
      if (Array.isArray(parsed)) {
        setHistory(parsed.slice(0, HISTORY_LIMIT));
      }
    } catch {
      setHistory([]);
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // Ignore storage failures (private mode/quota) and keep session history in memory.
    }
  }, [history]);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
      if (response.ok) {
        setHealth("online");
        setWakeCountdown(null);
        return;
      }
      setHealth("offline");
      setWakeCountdown((previous) => previous ?? OFFLINE_WAKE_CHECK_SECONDS);
    } catch {
      setHealth("offline");
      setWakeCountdown((previous) => previous ?? OFFLINE_WAKE_CHECK_SECONDS);
    }
  }, []);

  useEffect(() => {
    const intervalId = setInterval(() => {
      void checkHealth();
    }, ONLINE_HEALTH_POLL_MS);

    void checkHealth();

    return () => {
      clearInterval(intervalId);
    };
  }, [checkHealth]);

  useEffect(() => {
    if (health !== "offline") {
      return;
    }

    const countdownId = setInterval(() => {
      setWakeCountdown((previous) => {
        if (previous === null) {
          return OFFLINE_WAKE_CHECK_SECONDS;
        }

        if (previous <= 1) {
          void checkHealth();
          return OFFLINE_WAKE_CHECK_SECONDS;
        }

        return previous - 1;
      });
    }, 1000);

    return () => {
      clearInterval(countdownId);
    };
  }, [checkHealth, health]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setResult(null);
    setError("");

    if (previewUrl) {
      if (previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
    }

    if (!selectedFile) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    if (!selectedFile.type.startsWith("image/")) {
      setFile(null);
      setPreviewUrl(null);
      setError("Please select an image file.");
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  }

  async function runPrediction(targetFile: File, sourceImageDataUrl?: string) {
    setError("");
    setResult(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", targetFile);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload?.detail || "Prediction request failed.");
      }

      const prediction = payload as PredictionResponse;
      setResult(prediction);

      const encodedImage = sourceImageDataUrl ?? (await fileToDataUrl(targetFile));

      setHistory((previous) => [
        {
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          prediction: prediction.prediction,
          confidence: prediction.confidence,
          threshold: prediction.threshold,
          fileName: targetFile.name,
          createdAt: new Date().toISOString(),
          imageDataUrl: encodedImage,
          mimeType: targetFile.type,
          sizeBytes: targetFile.size,
          lastModified: targetFile.lastModified,
        },
        ...previous,
      ].slice(0, HISTORY_LIMIT));
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Could not reach backend server.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  async function handlePredict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setError("Upload an image to run prediction.");
      return;
    }

    await runPrediction(file);
  }

  async function handleRerun(item: HistoryItem) {
    if (!item.imageDataUrl) {
      setError("No cached image available for this history entry.");
      return;
    }

    try {
      const rebuiltFile = await dataUrlToFile(
        item.imageDataUrl,
        item.fileName,
        item.mimeType,
      );
      setFile(rebuiltFile);
      setPreviewUrl(item.imageDataUrl);
      await runPrediction(rebuiltFile, item.imageDataUrl);
    } catch {
      setError("Could not restore this history item for re-run.");
    }
  }

  function clearHistory() {
    setHistory([]);
    setHistoryFilter("all");
    try {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // Keep UI responsive even if storage is unavailable.
    }
  }

  const confidencePercent = useMemo(() => {
    if (!result) {
      return null;
    }
    return toConfidencePercent(result.confidence);
  }, [result]);

  const filteredHistory = useMemo(() => {
    if (historyFilter === "all") {
      return history;
    }

    return history.filter((item) => {
      const isStrong = item.confidence >= item.threshold;
      if (historyFilter === "strong") {
        return isStrong;
      }
      return !isStrong;
    });
  }, [history, historyFilter]);

  const predictionDetails = useMemo(() => {
    if (!result) {
      return null;
    }
    return parsePredictionDetails(result.prediction);
  }, [result]);

  const nextStep = useMemo(() => {
    if (!result || !predictionDetails) {
      return null;
    }
    return getNextStepAdvice(result, predictionDetails);
  }, [result, predictionDetails]);

  return (
    <div className="app-shell">
      <main className="console-card">
        <header className="console-topbar">
          <div>
            <p className="eyebrow">Plant Intelligence Console</p>
            <h1 className="title">Plant Disease Detector</h1>
          </div>
          <div className={`status-pill ${health}`}>
            <span className="status-dot" />
            {health === "checking" && "Checking backend"}
            {health === "online" && "Backend online"}
            {health === "offline" && "Backend sleeping"}
          </div>
        </header>

        {health === "offline" && wakeCountdown !== null && (
          <p className="wake-timer" aria-live="polite">
            Wake check in {wakeCountdown}s. Render free web services can take around 30-90 seconds to wake.
          </p>
        )}

        <section className="console-grid">
          <form className="panel panel-lifted" onSubmit={handlePredict}>
            <h2>Upload Leaf Image</h2>
            <p className="subtext">Use a clear photo of one leaf for best detection quality.</p>

            <label className="file-drop" htmlFor="leaf-image">
              <input id="leaf-image" type="file" accept="image/*" onChange={handleFileChange} />
              <span>Tap to select image</span>
              <small>JPG, JPEG, PNG</small>
            </label>

            <button type="submit" disabled={isLoading || !file || health !== "online"}>
              {isLoading ? "Running analysis..." : "Analyze Plant"}
            </button>

            <p className="api-text">API: {API_BASE_URL}</p>
          </form>

          <section className="panel panel-inset">
            <h2>Preview</h2>
            <div className="preview-box">
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={previewUrl} alt="Uploaded leaf" className="preview-image" />
              ) : (
                <p className="placeholder">Uploaded image preview appears here.</p>
              )}
            </div>

            {error && <p className="error-message">{error}</p>}
          </section>
        </section>

        <section className="result-card">
          <h2>Model Result</h2>

          {!result && <p className="placeholder">Run analysis to view prediction and confidence.</p>}

          {result && (
            <>
              <div className="result-layout">
                <div>
                  <p className="result-label">Disease Name</p>
                  <p className="result-value">{toReadableLabel(result.prediction)}</p>
                </div>

                <div>
                  <p className="result-label">Confidence</p>
                  <p className="result-value">{confidencePercent?.toFixed(2)}%</p>
                  <div className="meter-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={confidencePercent ?? 0}>
                    <span className="meter-fill" style={{ width: `${confidencePercent ?? 0}%` }} />
                  </div>
                  <p className="threshold-text">Threshold: {(result.threshold * 100).toFixed(0)}%</p>
                </div>
              </div>

              {predictionDetails && (
                <div className="diagnosis-card">
                  <div className="diagnosis-grid">
                    <div>
                      <p className="result-label">Plant Type</p>
                      <p className="diagnosis-value">{predictionDetails.plant}</p>
                    </div>
                    <div>
                      <p className="result-label">Leaf Status</p>
                      <p className={`diagnosis-value ${predictionDetails.isHealthy ? "status-healthy" : "status-disease"}`}>
                        {predictionDetails.condition}
                      </p>
                    </div>
                  </div>
                  <p className="diagnosis-note">{predictionDetails.note}</p>

                  {nextStep && (
                    <div className="next-step-card" aria-live="polite">
                      <p className="result-label">Recommended next step</p>
                      <p className="next-step-action">{nextStep.action}</p>
                      <p className="next-step-detail">{nextStep.detail}</p>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </section>

        <section className="history-card">
          <div className="history-header">
            <h2>Recent Predictions</h2>
            <div className="history-actions">
              <span className="history-count">Last {HISTORY_LIMIT}</span>
              <button
                type="button"
                className="clear-history-btn"
                onClick={clearHistory}
                disabled={history.length === 0}
              >
                Clear History
              </button>
            </div>
          </div>

          <div className="filter-chips" role="group" aria-label="History filters">
            <button
              type="button"
              className={`chip ${historyFilter === "all" ? "active" : ""}`}
              onClick={() => setHistoryFilter("all")}
            >
              All
            </button>
            <button
              type="button"
              className={`chip ${historyFilter === "strong" ? "active" : ""}`}
              onClick={() => setHistoryFilter("strong")}
            >
              Strong
            </button>
            <button
              type="button"
              className={`chip ${historyFilter === "inconclusive" ? "active" : ""}`}
              onClick={() => setHistoryFilter("inconclusive")}
            >
              Inconclusive
            </button>
          </div>

          {filteredHistory.length === 0 && (
            <p className="placeholder">No predictions yet. Run analysis to populate history.</p>
          )}

          {filteredHistory.length > 0 && (
            <ul className="history-list" aria-label="Prediction history list">
              {filteredHistory.map((item) => {
                const confidenceValue = toConfidencePercent(item.confidence);
                const badgeTone = item.confidence >= item.threshold ? "strong" : "soft";
                return (
                  <li key={item.id} className="history-item">
                    <div className="history-text">
                      <p className="history-prediction">{toReadableLabel(item.prediction)}</p>
                      <p className="history-meta">
                        {item.fileName} | {formatTimestamp(item.createdAt)}
                        {item.sizeBytes ? ` | ${(item.sizeBytes / 1024).toFixed(1)} KB` : ""}
                      </p>
                    </div>
                    <div className="history-side">
                      <span className={`confidence-badge ${badgeTone}`}>
                        {confidenceValue.toFixed(1)}%
                      </span>
                      <button
                        type="button"
                        className="rerun-btn"
                        onClick={() => void handleRerun(item)}
                        disabled={isLoading || !item.imageDataUrl || health !== "online"}
                      >
                        Re-run
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </main>
    </div>
  );
}
