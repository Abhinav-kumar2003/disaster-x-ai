# DISASTER-X AI — Frontend

React 19 + TypeScript + Tailwind CSS 4 + react-leaflet + recharts.

## Develop
```bash
npm install
npm run dev        # vite dev server with /api -> localhost:8000 proxy
```

## Build
```bash
npm run build      # tsc -b && vite build -> dist/
npm run preview
```

## Endpoints used
- GET  /api/health, /api/models
- GET  /api/predictions?tier=&limit=&offset=
- GET  /api/alerts?tile=&limit=
- GET  /api/analytics/features

In dev, Vite proxies `/api` to the FastAPI service (see vite.config.ts).
In production (Docker), nginx performs the same rewrite (see docker/nginx.conf).

A lighter static alternative with embedded data lives in ../dashboard/index.html.
