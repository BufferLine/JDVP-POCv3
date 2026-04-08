"use client";

import { useState } from "react";
import { I18nProvider, useI18n, LanguageToggle } from "@/components/I18nProvider";
import { AxisChart } from "@/components/AxisChart";
import { TrendBadge } from "@/components/TrendBadge";
import { Summary } from "@/components/Summary";
import { InflectionPoints } from "@/components/InflectionPoints";

type TurnScore = {
  turn_number: number;
  human_input_preview: string;
  scores: Record<string, number>;
};

type AnalyzeResult = {
  turns: TurnScore[];
  trends: {
    score: Record<string, string>;
    bucket: Record<string, string>;
    slopes: Record<string, number>;
  };
  summary: {
    turn_count: number;
    avg_da: number;
    max_da_turn: number;
    max_da_value: number;
    overall_trend: string;
  };
};

function AnalyzerApp() {
  const { t } = useI18n();
  const [mode, setMode] = useState<"link" | "paste">("link");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body = mode === "link" ? { mode, url } : { mode, text };
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Analysis failed");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex-1 flex flex-col items-center px-4 py-8">
      <div className="w-full max-w-3xl space-y-8">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold tracking-tight">
              <span style={{ color: "var(--accent-jdvp-light)" }}>JDVP</span>{" "}
              Analyzer
            </h1>
            <p className="text-sm" style={{ color: "var(--surface-muted)" }}>
              {t.header.subtitle}
            </p>
          </div>
          <LanguageToggle />
        </div>

        {/* Description */}
        <p className="text-sm leading-relaxed" style={{ color: "var(--surface-muted)" }}>
          {t.header.description}
        </p>

        {/* Input Card */}
        <div
          className="rounded-xl p-6 space-y-4"
          style={{
            background: "var(--surface-card)",
            border: "1px solid var(--surface-border)",
          }}
        >
          <div className="flex gap-2">
            {(["link", "paste"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                style={{
                  background: mode === m ? "var(--accent-jdvp)" : "var(--surface-elevated)",
                  color: mode === m ? "#fff" : "var(--surface-muted)",
                }}
              >
                {m === "link" ? t.input.tabLink : t.input.tabPaste}
              </button>
            ))}
          </div>

          {mode === "link" ? (
            <div className="space-y-1">
              <input
                type="url"
                placeholder={t.input.linkPlaceholder}
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="w-full px-4 py-3 rounded-lg text-sm outline-none"
                style={{
                  background: "var(--surface-elevated)",
                  border: "1px solid var(--surface-border)",
                  color: "var(--surface-fg)",
                }}
              />
              <p className="text-xs px-1" style={{ color: "var(--surface-muted)" }}>
                {t.input.linkHint}
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              <textarea
                placeholder={t.input.pastePlaceholder}
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={8}
                className="w-full px-4 py-3 rounded-lg text-sm outline-none resize-y"
                style={{
                  background: "var(--surface-elevated)",
                  border: "1px solid var(--surface-border)",
                  color: "var(--surface-fg)",
                }}
              />
              <p className="text-xs px-1" style={{ color: "var(--surface-muted)" }}>
                {t.input.pasteHint}
              </p>
            </div>
          )}

          <button
            onClick={handleSubmit}
            disabled={loading || (mode === "link" ? !url : !text)}
            className="w-full py-3 rounded-lg font-medium text-sm transition-opacity disabled:opacity-40"
            style={{ background: "var(--accent-jdvp)", color: "#fff" }}
          >
            {loading ? t.input.analyzing : t.input.analyze}
          </button>

          {error && (
            <p className="text-sm" style={{ color: "#ef4444" }}>
              {error}
            </p>
          )}
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-6">
            <Summary summary={result.summary} />

            {/* Chart */}
            <div
              className="rounded-xl p-6"
              style={{
                background: "var(--surface-card)",
                border: "1px solid var(--surface-border)",
              }}
            >
              <h2 className="text-lg font-semibold mb-1">{t.chart.title}</h2>
              <p className="text-xs mb-4" style={{ color: "var(--surface-muted)" }}>
                {t.chart.subtitle}
              </p>
              <AxisChart turns={result.turns} />
            </div>

            {/* Inflection Points */}
            <InflectionPoints turns={result.turns} />

            {/* Trends */}
            <div
              className="rounded-xl p-6"
              style={{
                background: "var(--surface-card)",
                border: "1px solid var(--surface-border)",
              }}
            >
              <h2 className="text-lg font-semibold mb-4">{t.trends.title}</h2>
              <div className="grid grid-cols-2 gap-3">
                {(["judgment_delegation", "cognitive_passivity", "information_dependency", "da_derived"] as const).map(
                  (field) => (
                    <TrendBadge
                      key={field}
                      field={field}
                      scoreTrend={result.trends.score[field]}
                      bucketTrend={result.trends.bucket[field]}
                      slope={result.trends.slopes[field]}
                    />
                  )
                )}
              </div>
            </div>

            {/* Turn Details */}
            <div
              className="rounded-xl p-6"
              style={{
                background: "var(--surface-card)",
                border: "1px solid var(--surface-border)",
              }}
            >
              <h2 className="text-lg font-semibold mb-4">{t.details.title}</h2>
              <div className="space-y-2 text-sm">
                {result.turns.map((turn) => (
                  <div
                    key={turn.turn_number}
                    className="flex items-start gap-3 py-2"
                    style={{ borderBottom: "1px solid var(--surface-border)" }}
                  >
                    <span
                      className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-mono"
                      style={{
                        background: "var(--surface-elevated)",
                        color: "var(--surface-muted)",
                      }}
                    >
                      {turn.turn_number + 1}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="truncate" style={{ color: "var(--surface-fg)" }}>
                        {turn.human_input_preview}
                      </p>
                      <div
                        className="flex gap-3 mt-1 text-xs font-mono"
                        style={{ color: "var(--surface-muted)" }}
                      >
                        <span style={{ color: "var(--color-jh)" }}>
                          JH {turn.scores.judgment_delegation}
                        </span>
                        <span style={{ color: "var(--color-cp)" }}>
                          CP {turn.scores.cognitive_passivity}
                        </span>
                        <span style={{ color: "var(--color-id)" }}>
                          ID {turn.scores.information_dependency}
                        </span>
                        <span style={{ color: "var(--color-da)" }}>
                          DA {turn.scores.da_derived}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Disclaimer + Footer */}
        <div className="space-y-4 pt-4">
          <div
            className="rounded-lg p-4 text-xs"
            style={{
              background: "var(--surface-elevated)",
              color: "var(--surface-muted)",
              border: "1px solid var(--surface-border)",
            }}
          >
            {t.disclaimer.text}
          </div>
          <div className="text-center text-xs" style={{ color: "var(--surface-muted)" }}>
            {t.footer.poweredBy} &middot; {t.footer.bufferline}
          </div>
        </div>
      </div>
    </main>
  );
}

export default function Home() {
  return (
    <I18nProvider>
      <AnalyzerApp />
    </I18nProvider>
  );
}
