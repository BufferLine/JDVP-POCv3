"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useI18n } from "./I18nProvider";
import { detectInflections } from "./InflectionPoints";

const AXIS_KEYS = [
  "judgment_delegation",
  "cognitive_passivity",
  "information_dependency",
  "da_derived",
] as const;

const AXIS_COLORS: Record<string, string> = {
  judgment_delegation: "#f59e0b",
  cognitive_passivity: "#ef4444",
  information_dependency: "#8b5cf6",
  da_derived: "#06b6d4",
};

type Turn = {
  turn_number: number;
  human_input_preview: string;
  scores: Record<string, number>;
};

function CustomTooltip({ active, payload, label, t }: any) {
  if (!active || !payload?.length) return null;

  const turn = payload[0]?.payload;
  return (
    <div
      className="rounded-lg p-3 text-xs max-w-xs"
      style={{
        background: "var(--surface-card)",
        border: "1px solid var(--surface-border)",
        boxShadow: "var(--shadow-lg)",
      }}
    >
      <p className="font-medium mb-1" style={{ color: "var(--surface-fg)" }}>
        {t.chart.turnLabel} {label}: {turn?.preview}
      </p>
      <div className="space-y-0.5">
        {payload.map((entry: any) => {
          const axisKey = entry.dataKey;
          const axisInfo = t.axes[axisKey as keyof typeof t.axes];
          return (
            <div key={axisKey} className="flex justify-between gap-4">
              <span style={{ color: entry.color }}>
                {axisInfo?.label || axisKey}
              </span>
              <span className="font-mono" style={{ color: "var(--surface-fg)" }}>
                {entry.value}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function AxisChart({ turns }: { turns: Turn[] }) {
  const { t } = useI18n();

  const data = turns.map((turn) => ({
    turn: turn.turn_number + 1,
    preview: turn.human_input_preview,
    ...turn.scores,
  }));

  // Find inflection points for reference lines
  const inflections = detectInflections(turns, 1.5);
  const inflectionTurns = [...new Set(inflections.slice(0, 3).map((p) => p.turn + 1))];

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
        <XAxis
          dataKey="turn"
          stroke="var(--surface-muted)"
          fontSize={12}
          label={{
            value: t.chart.turnLabel,
            position: "insideBottom",
            offset: -2,
            fill: "var(--surface-muted)",
          }}
        />
        <YAxis
          domain={[0, 10]}
          stroke="var(--surface-muted)"
          fontSize={12}
          ticks={[0, 2, 4, 6, 8, 10]}
        />
        <Tooltip content={<CustomTooltip t={t} />} />
        <Legend
          formatter={(value: string) => {
            const axisInfo = t.axes[value as keyof typeof t.axes];
            return axisInfo?.label || value;
          }}
          wrapperStyle={{ fontSize: 12 }}
        />
        {inflectionTurns.map((turn) => (
          <ReferenceLine
            key={turn}
            x={turn}
            stroke="var(--surface-muted)"
            strokeDasharray="3 3"
            strokeOpacity={0.5}
          />
        ))}
        {AXIS_KEYS.map((key) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            name={key}
            stroke={AXIS_COLORS[key]}
            strokeWidth={key === "da_derived" ? 3 : 1.5}
            dot={{ r: key === "da_derived" ? 4 : 2.5 }}
            activeDot={{ r: 6 }}
            opacity={key === "da_derived" ? 1 : 0.7}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
