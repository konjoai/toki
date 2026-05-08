import { KonjoApp } from "@konjoai/ui";
import { AttackTypeCards } from "./views/AttackTypeCards";
import { AttackBoard } from "./views/AttackBoard";
import { PipelineProgress } from "./views/PipelineProgress";
import { ComparePanel } from "./views/ComparePanel";
import { MetaInspector } from "./views/MetaInspector";

export default function App() {
  return (
    <KonjoApp
      product="toki"
      tagline="Adversarial Hardening · Red Team. Blue Team. Convergence."
      status={{ label: "ready", severity: "ok" }}
    >
      <Hero />

      <div className="space-y-6 mt-10">
        <AttackTypeCards />

        <AttackBoard />

        <PipelineProgress />

        <ComparePanel />

        <MetaInspector />

        <Footer />
      </div>
    </KonjoApp>
  );
}

function Hero() {
  return (
    <section className="text-center pt-6 pb-2">
      <p
        className="text-konjo-mono uppercase tracking-[0.32em] text-konjo-violet"
        style={{ fontSize: 11 }}
      >
        toki · 時 · adversarial · 赤
      </p>
      <h1
        className="text-konjo-display text-konjo-fg mt-4 mx-auto"
        style={{
          fontSize: 52,
          fontWeight: 600,
          letterSpacing: "-0.025em",
          maxWidth: 920,
          lineHeight: 1.05,
        }}
      >
        Models,{" "}
        <span style={{ color: "var(--color-konjo-accent)" }}>hardened</span>.
      </h1>
      <p
        className="text-konjo-fg-muted mt-5 mx-auto"
        style={{ fontSize: 16, maxWidth: 640, lineHeight: 1.55 }}
      >
        Adversarial hardening pipeline. Red-team your language model with
        jailbreaks, injections, edge cases, and boundary tests. Watch the score
        converge as your model learns to refuse unsafe requests.
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer
      className="mt-16 pt-8 border-t border-konjo-line/60 text-konjo-fg-muted text-konjo-mono"
      style={{ fontSize: 12 }}
    >
      <div className="flex flex-wrap gap-4 justify-between items-baseline">
        <span>
          built on{" "}
          <span className="text-konjo-fg">@konjoai/ui</span>
          {" · "}
          <span className="text-konjo-fg">/api/attacks</span>
          {" · "}
          <span className="text-konjo-fg">/api/run-round</span>
          {" · "}
          <span className="text-konjo-fg">/api/run-pipeline</span>
        </span>
        <span className="text-konjo-fg-faint">
          part of the KonjoAI portfolio · squish · kyro · miru · kohaku · kairu
          · toki · squash
        </span>
      </div>
    </footer>
  );
}
