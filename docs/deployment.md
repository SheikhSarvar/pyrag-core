# Production Deployment Guide

## HTTPS / TLS — T57

PyRAG Core's FastAPI app serves plain HTTP on port 8000. TLS termination
is handled by a reverse proxy, not the application itself. This keeps
certificate rotation out of the app lifecycle.

### Option A — Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    client_max_body_size 60M;  # match MAX_UPLOAD_SIZE_MB + headroom

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support (chat/stream endpoint)
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

### Option B — Caddy (automatic HTTPS)

```caddyfile
api.yourdomain.com {
    reverse_proxy api:8000 {
        flush_interval -1   # disable buffering for SSE
    }
}
```

### Option C — Cloud load balancer

If deploying behind AWS ALB / GCP Load Balancer / Azure App Gateway,
terminate TLS at the load balancer and forward HTTP to the container.
Ensure `X-Forwarded-Proto` is set so FastAPI's `request.url.scheme`
reflects `https`.

---

## Environment hardening checklist

Before going to production, verify:

- [ ] `SECRET_KEY` is a strong random 32+ char value, not the dev default
- [ ] `CORS_ORIGINS` is an explicit list, never `*`
- [ ] `API_KEYS` env var is set with rotated, randomly generated keys
- [ ] `ENVIRONMENT=production` (disables `/docs`, `/redoc`, `/openapi.json`)
- [ ] Postgres, Redis, Qdrant, MinIO ports are NOT exposed externally
      (see `docker-compose.prod.yml` — ports are stripped for these services)
- [ ] TLS certificates are valid and auto-renewing (Let's Encrypt / ACM)
- [ ] Rate limiting is enabled (`RATE_LIMIT_PER_MINUTE` tuned for your traffic)
- [ ] Langfuse keys configured for observability
- [ ] Database backups scheduled (pg_dump cron or managed Postgres snapshots)
- [ ] MinIO bucket versioning enabled for raw document recovery

---

## Running in production

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

This applies:
- Production Docker image target (non-root user, no source mount)
- 2 API replicas + 2 Celery worker replicas
- Closed ports on internal services
- Resource limits (2 CPU / 2GB per API replica)

## Performance validation — T58

Run the load test script against your deployment before declaring it
production-ready:

```bash
python scripts/load_test.py --target search --concurrency 50 --requests 500 \
    --base-url https://api.yourdomain.com --api-key $API_KEY

python scripts/load_test.py --target chat --concurrency 20 --requests 100 \
    --base-url https://api.yourdomain.com --api-key $API_KEY
```

Targets from the PRD:
| Metric | Target |
|---|---|
| Search latency (p95) | < 1s |
| Chat latency (p95) | < 3s |
| Concurrent users | 500+ |
| 50MB document processing | < 20s |

If p95 exceeds target under load, scale `celery_worker` replicas for
ingestion-bound workloads, or `api` replicas for retrieval-bound workloads.
Qdrant and Postgres connection pool sizes (`DATABASE_POOL_SIZE`) may also
need tuning at high concurrency.
