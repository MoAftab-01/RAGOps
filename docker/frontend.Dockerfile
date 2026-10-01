# ---------------------------------------------------------------------------
# RAGOps frontend: build the Vite bundle, serve it from nginx, and proxy
# /api to the backend so the browser sees a single origin in Docker.
# ---------------------------------------------------------------------------
FROM node:20-slim AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build


FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
