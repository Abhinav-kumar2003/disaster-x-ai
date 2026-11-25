export type RiskTier = 'low' | 'moderate' | 'moderate-high' | 'high'

export interface Prediction {
  lat: number
  lon: number
  risk_p: number
  risk_tier: RiskTier
  tile: string
  label: number
}

export interface AlertDriver {
  feature: string
  what_it_means: string
  value: number
  training_median: number
}

export interface Alert {
  cell_id: string
  lat: number
  lon: number
  tile: string
  risk_p: number
  risk_tier: RiskTier
  early_flood_months: number
  top_drivers: AlertDriver[]
}

export interface FeatureImportance {
  feature: string
  importance: number
}

export interface Analytics {
  top_features: FeatureImportance[]
  tier_distribution: Record<RiskTier, number>
  n_cells: number
  tier_thresholds: { high: number; 'moderate-high': number; moderate: number }
}

export const TIER_COLORS: Record<RiskTier, string> = {
  high: '#E5484D',
  'moderate-high': '#F76B15',
  moderate: '#FFC53D',
  low: '#46A758',
}

export const TIER_LABELS: Record<RiskTier, string> = {
  high: 'High',
  'moderate-high': 'Moderate-high',
  moderate: 'Moderate',
  low: 'Low',
}
