# vid-anomaly-app

Video anomaly detection dashboard.

## Getting Started

docker-compose

```bash
docker compose up vid-anomaly-app-gpu # if you have a GPU
docker compose up vid-anomaly-app-cpu # if you don't have a GPU

docker compose stop # to stop the container
```

```bash
docker build -f Dockerfile.gpu -t vadapp-gpu .
docker tag vadapp-gpu jiaherng/vadapp:latest-gpu
docker push jiaherng/vadapp:latest-gpu

docker build -f Dockerfile.cpu -t vadapp-cpu .
docker tag vadapp-cpu jiaherng/vadapp:latest-cpu
docker push jiaherng/vadapp:latest-cpu
```
