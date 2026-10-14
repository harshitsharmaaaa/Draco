"""HTTP surface of the agent layer. Approval/reject/ingest are the human-in-the-loop step: they are
authenticated API calls that write the approval record; no model-callable capability can."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import artifacts as A
from app.agent.catalog import build_registry
from app.agent.graph import AgentRunner
from app.agent.registry import CapabilityContext
from app.agent.schemas import AgentChatRequest
from app.agent.evidence import EvidenceStore
from app.core import concurrency
from app.api.chat import current_user, heavy_user, read_user
from app.inference.config import get_inference_settings
from app.inference.memory import SqliteConversationStore
from app.inference.schemas import InferenceError
from app.retrieval.service import get_service

logger = logging.getLogger("rag.agent")
router = APIRouter(prefix="/agent", tags=["agent"])
_runner: AgentRunner | None = None


def get_runner() -> AgentRunner:
    global _runner
    if _runner is None:
        from app.agent.wiring import build_services
        inf = get_inference_settings()
        svc, adapter = build_services(inf, get_service())
        _runner = AgentRunner(svc, build_registry(), adapter, inf, SqliteConversationStore(inf.store_path))
    return _runner


def _err(code) -> int:
    return {"invalid_input": 422, "not_found": 404, "denied": 403, "approval_required": 409, "not_configured": 503,
            "timeout": 504, "upstream_error": 502, "unsafe_url": 422, "limit_exceeded": 413, "conflict": 409}.get(code, 500)


async def _cap(runner: AgentRunner, name: str, args: dict, user: str):
    ctx = CapabilityContext(user_id=user, trace_id="api", services=runner.svc, evidence=EvidenceStore(),
                            permissions=frozenset({"read", "write", "generate"}))
    res = await runner.reg.execute(name, args, ctx, purpose="api request")
    if not res.ok:
        raise HTTPException(_err(res.error.code.value), f"{res.error.message} ({res.error.code.value})")
    return res


@router.post("/chat")
async def agent_chat(req: AgentChatRequest, user: str = Depends(heavy_user), runner: AgentRunner = Depends(get_runner)):
    try:
        async with concurrency.llm_slot():
            return (await runner.run(req, user)).model_dump()
    except HTTPException:
        raise
    except InferenceError as exc:
        raise HTTPException(exc.status_code, f"{exc} ({exc.code})") from exc
    except Exception as exc:
        logger.exception("agent_chat_unexpected")
        raise HTTPException(500, "Internal error while answering") from exc


@router.get("/capabilities")
async def capabilities(user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)):
    caps = runner.adapter.caps
    return {"capabilities": [c.describe() for c in runner.reg.all()], "groups": runner.reg.groups(),
            "unconfigured": runner.svc.missing(),
            "runtime": {"model": caps.model, "profile": caps.profile, "supports_thinking_toggle": caps.supports_toggle,
                        "fixed_mode": caps.fixed_mode, "template_declares_flag": caps.template_declares_flag,
                        "probe_verified": caps.probe_verified, "notes": caps.notes}}


class ApproveBody(BaseModel):
    ingest: bool = Field(default=False, description="start ingestion immediately after approving")


@router.post("/sources/{source_id}/approve")
async def approve_source(source_id: str, body: ApproveBody | None = None, user: str = Depends(read_user),
                         runner: AgentRunner = Depends(get_runner)):
    try:
        ap = runner.svc.store.approve_source(source_id, user, approve=True, ttl_hours=runner.svc.settings.approval_ttl_hours)
    except KeyError:
        raise HTTPException(404, "Source not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    out = {"source_id": source_id, "decision": ap["decision"], "expires_at": ap["expires_at"]}
    if body and body.ingest:
        out["ingestion"] = (await _cap(runner, "ingest_approved_source", {"source_id": source_id}, user)).model_dump()
    return out


@router.post("/sources/{source_id}/reject")
async def reject_source(source_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)):
    try:
        ap = runner.svc.store.approve_source(source_id, user, approve=False)
    except KeyError:
        raise HTTPException(404, "Source not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"source_id": source_id, "decision": ap["decision"]}


@router.post("/sources/{source_id}/ingest")
async def ingest_source(source_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)):
    return (await _cap(runner, "ingest_approved_source", {"source_id": source_id}, user)).model_dump()


@router.get("/sources/{source_id}")
async def source_provenance(source_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)):
    return (await _cap(runner, "inspect_source_provenance", {"source_id": source_id}, user)).model_dump()


@router.get("/artifacts/{artifact_id}")
async def download_artifact(artifact_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)):
    try:
        rec, path = runner.svc.artifact_store.resolve(artifact_id, user)   # owner, expiry, file present
    except A.ArtifactError:
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, media_type=A.MIME.get(rec["fmt"], "application/octet-stream"), filename=rec["filename"],
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"})


class FeedbackBody(BaseModel):
    target_type: str = Field(pattern="^(answer|artifact|source)$")
    target_id: str = Field(min_length=1, max_length=256)
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/feedback")
async def feedback(body: FeedbackBody, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)):
    return (await _cap(runner, "record_feedback", body.model_dump(), user)).model_dump()
