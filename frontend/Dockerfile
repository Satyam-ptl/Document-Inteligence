FROM node:22-slim AS build
WORKDIR /app
# Baked in at build time: this becomes `import.meta.env.VITE_API_BASE_URL`
# in the compiled JS bundle (Vite inlines VITE_-prefixed env vars at build
# time, not runtime — see src/services/api.ts). Defaults to the backend
# port docker-compose publishes to the host, since the browser running this
# bundle is outside the Compose network and can't resolve the `backend`
# service name — only `localhost:8000` (or whatever the host actually maps
# it to) works from there. Override with `--build-arg` if the backend is
# published on a different host/port.
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_API_KEY=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_API_KEY=$VITE_API_KEY
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
