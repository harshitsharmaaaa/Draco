"""Bounded admission to the LLM. vLLM serves a couple of sequences; unbounded fan-in only builds a queue of
requests whose clients have already timed out. Waiting briefly is allowed, then the caller gets 503 + Retry-After."""
import asyncio
import contextlib

from fastapi import HTTPException

from app.core import observability as obs
from app.core.http_settings import get_platform_settings

_sem: asyncio.Semaphore | None = None
_loop = None


def _semaphore() -> asyncio.Semaphore:
    global _sem, _loop
    loop = asyncio.get_running_loop()
    if _sem is None or _loop is not loop:  # a new event loop (tests, reload) needs its own semaphore
        _sem, _loop = asyncio.Semaphore(get_platform_settings().llm_max_concurrent), loop
    return _sem


async def acquire_llm_slot() -> asyncio.Semaphore:
    s = get_platform_settings()
    sem = _semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), s.llm_queue_wait_s)
    except asyncio.TimeoutError:
        obs.LLM_SLOT_REJECTIONS.inc()
        raise HTTPException(503, "The model is at capacity. Retry shortly.", headers={"Retry-After": "5"}) from None
    obs.LLM_INFLIGHT.inc()
    return sem


def release_llm_slot(sem: asyncio.Semaphore | None = None) -> None:
    (sem if sem is not None else _semaphore()).release()
    obs.LLM_INFLIGHT.dec()


@contextlib.asynccontextmanager
async def llm_slot():
    sem = await acquire_llm_slot()
    try:
        yield
    finally:
        release_llm_slot(sem)
