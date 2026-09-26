"""Load generator.

Drives a steady checkout workload with a diurnal wave and the occasional
benign spike, so the detectors are exercised against load variation that is
*not* an incident.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("loadgen")

GATEWAY = os.environ.get("API_GATEWAY_URL", "http://api-gateway:8000")
BASE_RPS = float(os.environ.get("LOADGEN_RPS", "20"))
SPIKE_PROBABILITY = float(os.environ.get("LOADGEN_SPIKE_PROBABILITY", "0.004"))


async def fire(client: httpx.AsyncClient) -> None:
    try:
        if random.random() < 0.35:
            await client.get(f"{GATEWAY}/products/{random.randint(1, 20)}", timeout=5.0)
        else:
            await client.post(
                f"{GATEWAY}/checkout",
                json={
                    "user_id": random.randint(1, 500),
                    "product_id": random.randint(1, 20),
                    "quantity": random.randint(1, 3),
                },
                timeout=10.0,
            )
    except Exception:
        pass  # failed requests are the point; the gateway records them


async def main() -> None:
    await asyncio.sleep(float(os.environ.get("LOADGEN_WARMUP_SECONDS", "10")))
    log.info("driving %s at ~%.0f rps", GATEWAY, BASE_RPS)
    spike_until = 0.0
    spike_factor = 1.0
    started = time.time()
    async with httpx.AsyncClient() as client:
        while True:
            now = time.time()
            wave = 1.0 + 0.18 * math.sin((now - started) / 240.0 * 2 * math.pi)
            if now > spike_until and random.random() < SPIKE_PROBABILITY:
                spike_factor = random.uniform(1.6, 2.6)
                spike_until = now + random.uniform(90, 240)
                log.info("benign traffic spike x%.1f for %.0fs", spike_factor, spike_until - now)
            factor = spike_factor if now < spike_until else 1.0
            rps = max(1.0, BASE_RPS * wave * factor)
            for _ in range(max(1, int(rps / 10))):
                asyncio.create_task(fire(client))
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
