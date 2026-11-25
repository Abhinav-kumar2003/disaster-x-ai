import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { Analytics } from '../types'
import { TIER_COLORS } from '../types'

const TIER_ORDER = ['high', 'moderate-high', 'moderate', 'low'] as const

export function TierChart({ analytics }: { analytics: Analytics }) {
  const data = TIER_ORDER.map((t) => ({
    tier: t,
    cells: analytics.tier_distribution[t] ?? 0,
  }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
        <XAxis
          dataKey="tier"
          tick={{ fill: '#8aa3bc', fontSize: 11 }}
          stroke="#1b3247"
        />
        <YAxis tick={{ fill: '#8aa3bc', fontSize: 11 }} stroke="#1b3247" />
        <Tooltip
          contentStyle={{
            background: '#0b1b2b',
            border: '1px solid #1b3247',
            borderRadius: 8,
            color: '#dce9f5',
          }}
        />
        <Bar dataKey="cells" radius={[4, 4, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.tier} fill={TIER_COLORS[d.tier]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function FeatureChart({ analytics }: { analytics: Analytics }) {
  const data = analytics.top_features.slice(0, 8).map((f) => ({
    name: f.feature.replace(/_/g, ' '),
    importance: f.importance,
  }))
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 8, right: 12, bottom: 0, left: 40 }}
      >
        <XAxis
          type="number"
          tick={{ fill: '#8aa3bc', fontSize: 11 }}
          stroke="#1b3247"
        />
        <YAxis
          type="category"
          dataKey="name"
          width={120}
          tick={{ fill: '#8aa3bc', fontSize: 11 }}
          stroke="#1b3247"
        />
        <Tooltip
          contentStyle={{
            background: '#0b1b2b',
            border: '1px solid #1b3247',
            borderRadius: 8,
            color: '#dce9f5',
          }}
        />
        <Bar dataKey="importance" fill="#25D9F2" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
