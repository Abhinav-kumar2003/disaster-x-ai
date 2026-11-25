import type { Alert, Analytics, Prediction } from './types'

/**
 * In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.ts).
 * In production behind Docker Compose, nginx proxies /api -> api:8000.
 */
const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${path}`)
  return (await r.json()) as T
}

export const fetchPredictions = (tier?: string, limit = 2000, offset = 0) =>
  get<{ total: number; predictions: Prediction[] }>(
    `/predictions?limit=${limit}&offset=${offset}${tier ? `&tier=${tier}` : ''}`,
  )

export const fetchAlerts = (tile?: string, limit = 20) =>
  get<{ total: number; alerts: Alert[] }>(
    `/alerts?limit=${limit}${tile ? `&tile=${tile}` : ''}`,
  )

export const fetchAnalytics = () => get<Analytics>('/analytics/features')

export const fetchModels = () => get<Record<string, Record<string, unknown>>>('/models')

export const checkHealth = () => get<{ status: string; models: string[] }>('/health')
