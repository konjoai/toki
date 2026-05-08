import { useState } from "react";
import { runRound } from "../lib/api";
import type { RoundResult } from "../lib/api";

export function AttackBoard() {
  const [roundResult, setRoundResult] = useState<RoundResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  const handleRunRound = async () => {
    setLoading(true);
    const { data, fromMock: mock } = await runRound();
    setRoundResult(data);
    setFromMock(mock);
    setLoading(false);
  };

  return (
    <div className="space-y-3">
      <header>
        <h2
          className="text-konjo-display text-konjo-fg"
          style={{ fontSize: 20, fontWeight: 600 }}
        >
          Attack Board
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Real-time adversarial results ·{" "}
          <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <button
          onClick={handleRunRound}
          disabled={loading}
          className={[
            "w-full px-3 py-2 rounded-konjo text-[12px] font-mono uppercase transition-colors",
            loading
              ? "bg-konjo-surface/40 text-konjo-fg-muted cursor-not-allowed"
              : "bg-konjo-accent text-konjo-bg hover:bg-konjo-accent/90",
          ].join(" ")}
        >
          {loading ? "Running…" : "Run Attack Round"}
        </button>

        {roundResult && (
          <div className="space-y-3">
            <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
              Round {roundResult.round} · Score: {roundResult.score.toFixed(1)}%
            </div>

            <div className="space-y-2">
              {roundResult.attack_results.map((result, idx) => (
                <div
                  key={idx}
                  className="bg-konjo-surface/60 rounded p-3 space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="text-konjo-fg text-[12px] truncate">
                        {result.prompt}
                      </div>
                      <div className="text-konjo-mono text-[10px] text-konjo-fg-muted mt-1">
                        {result.category}
                      </div>
                    </div>
                    <div
                      className={[
                        "px-2 py-1 rounded text-[10px] font-mono uppercase whitespace-nowrap",
                        result.blocked
                          ? "bg-konjo-good/20 text-konjo-good"
                          : "bg-konjo-warm/20 text-konjo-warm",
                      ].join(" ")}
                    >
                      {result.blocked ? "BLOCKED" : "BYPASSED"}
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-konjo-fg-muted">
                      {result.result.slice(0, 40)}
                      {result.result.length > 40 ? "…" : ""}
                    </span>
                    <span
                      className="text-konjo-accent font-mono"
                      style={{ marginLeft: "auto" }}
                    >
                      {result.score.toFixed(0)}%
                    </span>
                  </div>

                  <div className="h-1.5 bg-konjo-line/30 rounded overflow-hidden">
                    <div
                      className={
                        result.blocked
                          ? "h-full bg-konjo-good"
                          : "h-full bg-konjo-warm"
                      }
                      style={{ width: `${result.score}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="text-konjo-mono text-[11px] text-konjo-fg-muted animate-pulse">
            running adversarial attack…
          </div>
        )}
      </div>
    </div>
  );
}
