# vid-anomaly-app

Video anomaly detection dashboard.

## Getting Started

docker-compose

```bash
docker compose up vid-anomaly-app-gpu # if you have a GPU
docker compose up vid-anomaly-app-cpu # if you don't have a GPU
```

```bash
docker build -f Dockerfile.gpu -t vid-anomaly-app-gpu .
docker tag vid-anomaly-app-gpu asia-southeast1-docker.pkg.dev/vid-anomaly-app/vid-anomaly-app/vid-anomaly-app-gpu # docker tag SOURCE-IMAGE LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY/IMAGE:TAG
docker push asia-southeast1-docker.pkg.dev/vid-anomaly-app/vid-anomaly-app/vid-anomaly-app-gpu
```

```bash
docker build -f Dockerfile.cpu -t vid-anomaly-app-cpu .
docker tag vid-anomaly-app-cpu asia-southeast1-docker.pkg.dev/vid-anomaly-app/vid-anomaly-app/vid-anomaly-app-cpu # docker tag SOURCE-IMAGE LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY/IMAGE:TAG
docker push asia-southeast1-docker.pkg.dev/vid-anomaly-app/vid-anomaly-app/vid-anomaly-app-cpu
```
