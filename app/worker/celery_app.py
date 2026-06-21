from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "pyrag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.worker.tasks.ingestion",  # registered in T13
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # fair dispatch for long ingestion tasks
    task_routes={
        "app.worker.tasks.ingestion.*": {"queue": "ingestion"},
    },
    task_soft_time_limit=300,   # 5 min soft limit
    task_time_limit=600,        # 10 min hard limit
    result_expires=86400,       # results kept 24h
)
