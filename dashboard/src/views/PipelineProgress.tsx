import { useState } from "react";
import { runPipeline } from "../lib/api";
import type { PipelineResult } from "../lib/api";

export function PipelineProgress() {
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  const handleRunPipeline = async () => {
    setLoading(true);
    const { data, fromMock: mock } = await runPipeline();
    setResult(data);
    setFromMock(mock);
    setLoading(false);
  };

  const svgWidth = 400;
  const svgHeight = 200;

  const pathData =
    result && result.rounds.length > 0
      ? result.rounds
          .map((round, idx) => {
            const x = (idx / (result.rounds.length - 1 || 1)) * (svgWidth - 40) + 20;
            const y = svgHeight - (round.score / 100) * (svgHeight - 40) - 20;
            return `${idx === 0 ? "M" : "L"} ${x} ${y}`;
          })
          .join(" ")
      : "";

  const convergenceThreshold = 85;
  const thresholdY = svgHeight - (convergenceThreshold / 100) * (svgHeight - 40) - 20;

  return (
    <div className="space-y-3">
      <header>
        <h2
          className="text-konjo-display text-konjo-fg"
          style={{ fontSize: 20, fontWeight: 600 }}
        >
          Hardening Pipeline
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Multi-round adversarial optimization ·{" "}
          <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <button
          onClick={handleRunPipeline}
          disabled={loading}
          className={[
            "w-full px-3 py-2 rounded-konjo text-[12px] font-mono uppercase transition-colors",
            loading
              ? "bg-konjo-surface/40 text-konjo-fg-muted cursor-not-allowed"
              : "bg-konjo-accent text-konjo-bg hover:bg-konjo-accent/90",
          ].join(" ")}
        >
          {loading ? "Running…" : "Run Full Pipeline"}
        </button>

        {result && (
          <div className="space-y-4">
            {/* Progress info */}
            <div className="flex justify-between items-center text-[13px]">
              <div>
                <div className="text-konjo-fg-muted">Round</div>
                <div className="text-konjo-fg font-mono">
                  {result.total_rounds}
                </div>
              </div>
              <div>
                <div className="text-konjo-fg-muted">Final Score</div>
                <div className="text-konjo-fg font-mono">
                  {result.final_score.toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-konjo-fg-muted">Status</div>
                <div
                  className="font-mono"
                  style={{
                    color: result.converged
                      ? "var(--color-konjo-good)"
                      : "var(--color-konjo-warm)",
                  }}
                >
                  {result.converged ? "CONVERGED" : "RUNNING"}
                </div>
              </div>
            </div>

            {/* Chart */}
            <svg
              width="100%"
              height="240"
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="bg-konjo-surface/40 rounded border border-konjo-line/60"
            >
              {/* Grid */}
              <line
                x1="20"
                y1={svgHeight - 20}
                x2={svgWidth - 20}
                y2={svgHeight - 20}
                stroke="var(--color-konjo-line)"
                strokeWidth="1"
                opacity="0.3"
              />
              <line
                x1="20"
                y1="20"
                x2="20"
                y2={svgHeight - 20}
                stroke="var(--color-konjo-line)"
                strokeWidth="1"
                opacity="0.3"
              />

              {/* Convergence threshold */}
              <line
                x1="20"
                y1={thresholdY}
                x2={svgWidth - 20}
                y2={thresholdY}
                stroke="var(--color-konjo-good)"
                strokeWidth="1"
                strokeDasharray="4 4"
                opacity="0.4"
              />
              <text
                x={svgWidth - 25}
                y={thresholdY - 3}
                textAnchor="end"
                className="text-konjo-mono text-[9px]"
                fill="var(--color-konjo-fg-muted)"
              >
                85%
              </text>

              {/* Score curve */}
              {pathData && (
                <path
                  d={pathData}
                  fill="none"
                  stroke="var(--color-konjo-accent)"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Data points */}
              {result.rounds.map((round, idx) => {
                const x =
                  (idx / (result.rounds.length - 1 || 1)) *
                    (svgWidth - 40) +
                  20;
                const y =
                  svgHeight - (round.score / 100) * (svgHeight - 40) - 20;
                return (
                  <circle
                    key={idx}
                    cx={x}
                    cy={y}
                    r="3"
                    fill="var(--color-konjo-accent)"
                    opacity="0.8"
                  />
                );
              })}

              {/* Labels */}
              <text
                x={25}
                y={15}
                className="text-konjo-mono text-[9px]"
                fill="var(--color-konjo-fg-muted)"
              >
                Score
              </text>
              <text
                x={svgWidth - 25}
                y={svgHeight - 5}
                textAnchor="end"
                className="text-konjo-mono text-[9px]"
                fill="var(--color-konjo-fg-muted)"
              >
                Round
              </text>
            </svg>

            {/* Round summary */}
            <div className="bg-konjo-surface/60 rounded p-3 space-y-2 max-h-48 overflow-y-auto">
              <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
                Per-Round Scores
              </div>
              <div className="space-y-1 text-[11px]">
                {result.rounds.map((round) => (
                  <div
                    key={round.round}
                    className="flex justify-between items-center"
                  >
                    <span className="text-konjo-fg-muted">Round {round.round}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-konjo-line/30 rounded overflow-hidden">
                        <div
                          className="h-full bg-konjo-accent"
                          style={{
                            width: `${Math.min(100, (round.score / 100) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-konjo-fg font-mono min-w-[40px] text-right">
                        {round.score.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {loading && (
          <div className="text-konjo-mono text-[11px] text-konjo-fg-muted animate-pulse">
            running hardening pipeline…
          </div>
        )}
      </div>
    </div>
  );
}
