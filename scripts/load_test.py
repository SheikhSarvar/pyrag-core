#!/usr/bin/env python3
"""
Load test — T58.
Validates PyRAG Core against the PRD performance targets:
  - Search latency      < 1s
  - Chat latency         < 3s
  - Concurrent users     500+
  - Document processing  50MB < 20s

Usage:
    python scripts/load_test.py --target search --concurrency 50 --requests 500
    python scripts/load_test.py --target chat --concurrency 20 --requests 100

Requires: pip install httpx
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class LoadTestResult:
    target: str
    total_requests: int
    successful: int
    failed: int
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.latencies_ms:
            return {"error": "no successful requests"}
        sorted_lat = sorted(self.latencies_ms)
        n = len(sorted_lat)
        return {
            "target": self.target,
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.successful / self.total_requests * 100, 2),
            "latency_ms": {
                "min": round(min(sorted_lat), 1),
                "max": round(max(sorted_lat), 1),
                "mean": round(statistics.mean(sorted_lat), 1),
                "p50": round(sorted_lat[int(n * 0.50)], 1),
                "p95": round(sorted_lat[int(n * 0.95)], 1) if n > 1 else sorted_lat[0],
                "p99": round(sorted_lat[min(int(n * 0.99), n - 1)], 1),
            },
        }


_TARGETS = {
    "search": {
        "method": "POST",
        "path": "/api/v1/search",
        "body": lambda: {"dataset_id": "load-test-ds", "query": "What is the quarterly revenue?", "top_k": 10},
        "threshold_ms": 1000,
    },
    "chat": {
        "method": "POST",
        "path": "/api/v1/chat",
        "body": lambda: {"dataset_id": "load-test-ds", "message": "Summarize the key findings"},
        "threshold_ms": 3000,
    },
    "health": {
        "method": "GET",
        "path": "/health",
        "body": lambda: None,
        "threshold_ms": 100,
    },
}


async def _single_request(
    client: httpx.AsyncClient,
    target_cfg: dict,
    api_key: str | None,
) -> tuple[bool, float, str]:
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.monotonic()
    try:
        if target_cfg["method"] == "GET":
            resp = await client.get(target_cfg["path"], headers=headers, timeout=30)
        else:
            resp = await client.post(
                target_cfg["path"], json=target_cfg["body"](), headers=headers, timeout=30
            )
        latency_ms = (time.monotonic() - start) * 1000
        if resp.status_code >= 400:
            return False, latency_ms, f"HTTP {resp.status_code}: {resp.text[:200]}"
        return True, latency_ms, ""
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return False, latency_ms, str(exc)


async def run_load_test(
    base_url: str,
    target: str,
    concurrency: int,
    total_requests: int,
    api_key: str | None = None,
) -> LoadTestResult:
    target_cfg = _TARGETS[target]
    result = LoadTestResult(target=target, total_requests=total_requests, successful=0, failed=0)

    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_request(client: httpx.AsyncClient) -> None:
        async with semaphore:
            success, latency_ms, error = await _single_request(client, target_cfg, api_key)
            if success:
                result.successful += 1
                result.latencies_ms.append(latency_ms)
            else:
                result.failed += 1
                result.errors.append(error)

    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks = [_bounded_request(client) for _ in range(total_requests)]
        await asyncio.gather(*tasks)

    return result


def print_report(result: LoadTestResult, threshold_ms: float) -> None:
    summary = result.summary()
    print(f"\n{'='*60}")
    print(f"LOAD TEST: {result.target.upper()}")
    print(f"{'='*60}")
    if "error" in summary:
        print(f"FAILED — {summary['error']}")
        for err in result.errors[:5]:
            print(f"  - {err}")
        return

    print(f"Total requests:   {summary['total_requests']}")
    print(f"Successful:       {summary['successful']}")
    print(f"Failed:           {summary['failed']}")
    print(f"Success rate:     {summary['success_rate']}%")
    print(f"\nLatency (ms):")
    lat = summary["latency_ms"]
    print(f"  min:  {lat['min']}")
    print(f"  mean: {lat['mean']}")
    print(f"  p50:  {lat['p50']}")
    print(f"  p95:  {lat['p95']}")
    print(f"  p99:  {lat['p99']}")
    print(f"  max:  {lat['max']}")

    passed = lat["p95"] <= threshold_ms
    status = "PASS" if passed else "FAIL"
    print(f"\nTarget: p95 < {threshold_ms}ms — {status}")

    if result.errors:
        print(f"\nSample errors:")
        for err in result.errors[:5]:
            print(f"  - {err}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="PyRAG Core load test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--target", choices=list(_TARGETS.keys()), default="search")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    print(f"Running load test: {args.target} | concurrency={args.concurrency} | requests={args.requests}")
    result = await run_load_test(
        base_url=args.base_url,
        target=args.target,
        concurrency=args.concurrency,
        total_requests=args.requests,
        api_key=args.api_key,
    )
    print_report(result, threshold_ms=_TARGETS[args.target]["threshold_ms"])


if __name__ == "__main__":
    asyncio.run(main())
