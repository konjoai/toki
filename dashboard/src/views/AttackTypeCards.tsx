import { useEffect, useState } from "react";
import { fetchAttacks } from "../lib/api";
import type { AttackType } from "../lib/api";

export function AttackTypeCards() {
  const [attackTypes, setAttackTypes] = useState<AttackType[]>([]);
  const [loading, setLoading] = useState(true);
  const [fromMock, setFromMock] = useState(false);

  useEffect(() => {
    (async () => {
      const { data, fromMock: mock } = await fetchAttacks();
      setAttackTypes(data.attack_types);
      setFromMock(mock);
      setLoading(false);
    })();
  }, []);

  if (loading) {
    return (
      <div className="space-y-3">
        <header>
          <h2
            className="text-konjo-display text-konjo-fg"
            style={{ fontSize: 20, fontWeight: 600 }}
          >
            Attack Types
          </h2>
        </header>
        <div className="text-konjo-fg-muted text-konjo-mono text-[12px]">
          loading attack types…
        </div>
      </div>
    );
  }

  const colors: Record<string, string> = {
    jailbreak: "var(--color-konjo-warm)",
    injection: "var(--color-konjo-accent)",
    edge_case: "var(--color-konjo-good)",
    boundary: "var(--color-konjo-fg-muted)",
  };

  return (
    <div className="space-y-3">
      <header>
        <h2
          className="text-konjo-display text-konjo-fg"
          style={{ fontSize: 20, fontWeight: 600 }}
        >
          Attack Types
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          4 adversarial strategies ·{" "}
          <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
        </p>
      </header>

      <div className="grid sm:grid-cols-2 gap-3">
        {attackTypes.map((attack) => (
          <div
            key={attack.category}
            className="glass-konjo rounded-konjo-lg p-4 space-y-2"
          >
            <div
              className="text-konjo-mono uppercase tracking-[0.16em] text-[11px] font-bold"
              style={{ color: colors[attack.category] }}
            >
              {attack.category}
            </div>
            <p className="text-konjo-fg text-[13px] leading-relaxed">
              {attack.description}
            </p>
            <div className="space-y-1 pt-2 border-t border-konjo-line/40">
              <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">
                Sample Prompts
              </div>
              {attack.sample_prompts.map((prompt: string, i: number) => (
                <div
                  key={i}
                  className="text-konjo-fg text-[11px] italic opacity-75"
                >
                  "{prompt.slice(0, 50)}{prompt.length > 50 ? "…" : ""}"
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
