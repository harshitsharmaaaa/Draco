"""The Draco control plane: one entry point that runs any registered workload with identity,
policy, budget, cancellation, tracing and persistence, and returns a normalized ExecutionResult."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from draco.core.budget import BudgetLimits, BudgetTracker
from draco.core.config import DracoSettings
from draco.core.context import RESERVED_METADATA, CancellationToken, ExecutionContext, new_id, use_context
from draco.core.errors import ContractError, DracoError, PersistenceError, PolicyViolation, RoutingError
from draco.core.events import Event, EventBus
from draco.core.identity import Identity
from draco.core.policy import Policy
from draco.core.protocols import Workload
from draco.core.results import ErrorInfo, ExecutionResult, Handoff, Status, Usage, Waiting, status_for
from draco.observability.redaction import Redactor
from draco.observability.trace import Exporter, JsonlExporter, Tracer
from draco.plugins.registry import PluginHost
from draco.plugins.services import ServiceRegistry
from draco.policies.guardrails import Guardrail, run_guardrails
from draco.security.audit import authorize
from draco.security.authorization import Authorizer, Resource, RoleAuthorizer
from draco.state.memory import InMemoryStore
from draco.state.stores import ExecutionRecord

logger = logging.getLogger("draco.harness")
_RESUMABLE = {"failed", "timeout", "cancelled", "waiting", "running"}  # running = interrupted process


@dataclass
class WorkloadSpec:
    name: str
    fn: Workload
    description: str = ""
    capabilities: frozenset[str] = frozenset()
    meta: dict[str, Any] = field(default_factory=dict)


class Harness:
    def __init__(self, settings: DracoSettings | None = None, *, store: Any = None,
                 host: PluginHost | None = None, exporters: list[Exporter] | None = None,
                 strict_persistence: bool = False, authorizer: Authorizer | None = None):
        self.settings = settings or DracoSettings()
        self.host = host or PluginHost()
        self.services: ServiceRegistry = self.host.services
        self.bus: EventBus = self.host.bus
        self.store = store or InMemoryStore()
        self.strict_persistence = strict_persistence
        if exporters is None and self.settings.trace_path:
            exporters = [JsonlExporter(self.settings.trace_path)]
        self.tracer = Tracer(self.bus, exporters, Redactor(self.settings.redact_keys, self.settings.max_payload_chars, pii_keys=self.settings.pii_keys))
        self.authorizer: Authorizer = authorizer or RoleAuthorizer(require_identity=self.settings.require_identity)
        for name, svc in (("authorizer", self.authorizer), ("events", self.bus), ("settings", self.settings), ("store", self.store),
                          ("tracer", self.tracer), ("harness", self)):
            if not self.services.has(name):  # protected: plugins cannot replace these
                self.services.provide(name, svc, protected=True)
        self.bus.subscribe("*", self._persist_event)
        self._workloads: dict[str, WorkloadSpec] = {}
        self._inflight: dict[str, ExecutionContext] = {}
        self._children: dict[str, list[str]] = {}
        self._sem = asyncio.Semaphore(self.settings.max_concurrency)
        self._input_guards: list[Guardrail] = []
        self._output_guards: list[Guardrail] = []

    # -- registration / hooks -------------------------------------------------------------
    def register(self, name: str, fn: Workload, *, description: str = "",
                 capabilities: frozenset[str] | set[str] = frozenset(), **meta: Any) -> None:
        if name in self._workloads:
            raise ContractError(f"workload '{name}' already registered")
        self._workloads[name] = WorkloadSpec(name, fn, description, frozenset(capabilities), meta)

    def get(self, name: str) -> WorkloadSpec:
        try:
            return self._workloads[name]
        except KeyError:
            raise RoutingError(f"unknown workload '{name}'", details={"known": sorted(self._workloads)}) from None

    def names(self) -> list[str]:
        return sorted(self._workloads)

    def on(self, pattern: str, handler: Callable[[Event], Any], *, critical: bool = False) -> Callable[[], None]:
        """Subscribe to lifecycle events: execution.started/finished/failed/cancelled/timeout, model.*, tool.*,
        handoff, retry, step.*. `critical=True` handlers run first and may veto by raising."""
        return self.bus.subscribe(pattern, handler, critical=critical)

    def add_input_guardrail(self, g: Guardrail) -> None:
        self._input_guards.append(g)

    def add_output_guardrail(self, g: Guardrail) -> None:
        self._output_guards.append(g)

    async def _persist_event(self, e: Event) -> None:
        if e.execution_id and hasattr(self.store, "append_event"):
            await self.store.append_event(e.execution_id, e.name, e.ts, self.tracer.redactor.redact(e.data))

    def cancel(self, execution_id: str, reason: str = "cancelled by caller", *, identity: Identity | None = None) -> bool:
        """Cancel a running execution. A tenant-scoped execution can only be cancelled by an identity of
        the same tenant; without one it reports False (not found) rather than revealing it exists."""
        ctx = self._inflight.get(execution_id)
        if ctx is None:
            return False
        if ctx.tenant_id is not None and (identity is None or identity.tenant_id != ctx.tenant_id):
            return False
        ctx.cancel.cancel(reason)
        return True

    # -- running ----------------------------------------------------------------------------
    def _new_context(self, *, parent: ExecutionContext | None, target: str, session_id: str | None,
                     user_id: str | None, tenant_id: str | None, policy: Policy | None,
                     limits: BudgetLimits | None, timeout_s: float | None, cancel: CancellationToken | None,
                     correlation_id: str | None, agent_id: str | None, metadata: dict[str, Any] | None,
                     identity: Identity | None = None) -> ExecutionContext:
        timeout = timeout_s if timeout_s is not None else self.settings.default_timeout_s
        deadline = time.monotonic() + timeout if timeout else None
        if parent is not None:
            if identity is not None and identity != parent.identity:
                raise PolicyViolation("a child execution inherits its parent's identity and cannot change it")
            if (user_id and user_id != parent.user_id) or (tenant_id and tenant_id != parent.tenant_id):
                raise PolicyViolation("a child execution cannot change user or tenant")
            return parent.child(agent_id=agent_id or target, deadline=deadline, policy=policy, metadata=metadata)
        lim = limits or self.settings.limits()
        if identity is not None:
            if (user_id and user_id != identity.subject) or (tenant_id and tenant_id != identity.tenant_id):
                raise PolicyViolation("user_id/tenant_id contradict the authenticated identity")
            user_id, tenant_id = identity.subject, identity.tenant_id
        return ExecutionContext(
            identity=identity,
            session_id=session_id, agent_id=agent_id or target, user_id=user_id, tenant_id=tenant_id,
            correlation_id=correlation_id or new_id(), cancel=cancel or CancellationToken(),
            deadline=deadline, budget=BudgetTracker(lim), policy=policy or Policy(), services=self.services,
            metadata=dict(metadata or {}))

    async def run(self, target: str, input: Any = None, *, parent: ExecutionContext | None = None,
                  session_id: str | None = None, user_id: str | None = None, tenant_id: str | None = None,
                  policy: Policy | None = None, limits: BudgetLimits | None = None,
                  timeout_s: float | None = None, cancel: CancellationToken | None = None,
                  idempotency_key: str | None = None, correlation_id: str | None = None,
                  agent_id: str | None = None, metadata: dict[str, Any] | None = None,
                  identity: Identity | None = None) -> ExecutionResult:
        """`identity` is the trusted caller (from your Authenticator). Plain `user_id`/`tenant_id` are accepted
        for trusted in-process callers but carry no authentication; they cannot contradict an identity, and
        children always inherit the parent's. Idempotency keys are scoped per tenant."""
        try:
            ctx = self._new_context(parent=parent, target=target, session_id=session_id or (parent.session_id if parent else None),
                                    user_id=user_id, tenant_id=tenant_id, policy=policy, limits=limits,
                                    timeout_s=timeout_s, cancel=cancel, correlation_id=correlation_id,
                                    agent_id=agent_id, metadata=metadata, identity=identity)
            _reserved_check(metadata)
        except DracoError as exc:  # e.g. depth budget exhausted while creating a child
            return ExecutionResult(status_for(exc), new_id(), error=ErrorInfo.from_exception(exc),
                                   parent_execution_id=parent.execution_id if parent else None,
                                   finished_at=time.time())
        if ctx.services is None:
            ctx = ExecutionContext(**{**ctx.__dict__, "services": self.services})
        return await self._execute(ctx, target, input, idempotency_key=scope_key(idempotency_key, ctx.tenant_id), attempts=1, resumed=False,
                                   is_root=parent is None)

    async def resume(self, execution_id: str, *, input: Any = None, timeout_s: float | None = None,
                     policy: Policy | None = None, limits: BudgetLimits | None = None,
                     identity: Identity | None = None) -> ExecutionResult:
        """Re-run a failed, waiting or interrupted execution under the same execution id. Completed `step()`s
        return their recorded results; unfinished non-idempotent steps raise ReconciliationRequired."""
        rec = await self.store.get_execution(execution_id)
        if rec is None:
            raise ContractError(f"unknown execution '{execution_id}'")
        if rec.status not in _RESUMABLE:
            raise ContractError(f"execution '{execution_id}' is {rec.status} and cannot be resumed")
        if execution_id in self._inflight:
            raise ContractError(f"execution '{execution_id}' is still running")
        timeout = timeout_s if timeout_s is not None else self.settings.default_timeout_s
        if rec.tenant_id is not None and (identity is None or identity.tenant_id != rec.tenant_id):
            raise ContractError(f"unknown execution '{execution_id}'")  # same answer as a missing id: no cross-tenant probing
        if rec.user_id is not None and (identity is None or identity.subject != rec.user_id):
            raise ContractError(f"unknown execution '{execution_id}'")  # prevent same-tenant users from resuming another user's execution
        ctx = ExecutionContext(
            execution_id=rec.execution_id, parent_execution_id=rec.parent_execution_id, session_id=rec.session_id,
            identity=identity, user_id=identity.subject if identity else rec.user_id, tenant_id=rec.tenant_id,
            agent_id=rec.agent_id, correlation_id=rec.correlation_id or new_id(),
            deadline=time.monotonic() + timeout if timeout else None,
            budget=BudgetTracker(limits or self.settings.limits()), policy=policy or Policy(),
            services=self.services, metadata={"resumed": True, "previous_status": rec.status})
        await authorize(ctx, "execution.resume", Resource("execution", rec.execution_id, rec.tenant_id))
        return await self._execute(ctx, rec.target, rec.input if input is None else input,
                                   idempotency_key=rec.idempotency_key, attempts=rec.attempts + 1, resumed=True,
                                   is_root=rec.parent_execution_id is None)

    async def _save(self, rec: ExecutionRecord) -> PersistenceError | None:
        try:
            await self.store.save_execution(rec)
            return None
        except Exception as exc:
            err = exc if isinstance(exc, PersistenceError) else PersistenceError(f"state store failed: {type(exc).__name__}")
            logger.warning("persistence_failed", extra={"execution_id": rec.execution_id})
            return err

    async def _execute(self, ctx: ExecutionContext, target: str, input: Any, *, idempotency_key: str | None,
                       attempts: int, resumed: bool, is_root: bool) -> ExecutionResult:
        started = time.time()
        res = ExecutionResult(Status.failed, ctx.execution_id, parent_execution_id=ctx.parent_execution_id,
                              session_id=ctx.session_id, agent_id=ctx.agent_id, correlation_id=ctx.correlation_id,
                              started_at=started)
        if ctx.parent_execution_id:
            self._children.setdefault(ctx.parent_execution_id, []).append(ctx.execution_id)
        usage0 = ctx.budget.snapshot()
        rec = ExecutionRecord(ctx.execution_id, target, "running", started, ctx.parent_execution_id, ctx.session_id,
                              ctx.agent_id, ctx.correlation_id, idempotency_key=idempotency_key, attempts=attempts,
                              input=_jsonable(input), tenant_id=ctx.tenant_id, user_id=ctx.user_id,
                              roles=sorted(ctx.identity.roles) if ctx.identity else [])
        sem_held = False
        try:
            # 1. resolve + authorize
            spec = self.get(target)
            ctx.policy.authorize_agent(target)
            # 2. idempotency: a completed run with this key is returned, never re-executed
            if idempotency_key and not resumed:
                prev = await self.store.find_by_key(idempotency_key)
                if prev is not None:
                    if prev.status == "completed":
                        res.status, res.output, res.replayed = Status.completed, prev.output, True
                        res.execution_id = prev.execution_id
                        res.finished_at = time.time()
                        await self.bus.emit(Event("execution.replayed", prev.execution_id, data={"key_reused": True}))
                        return res
                    if prev.status == "running":
                        raise ContractError("an execution with this idempotency key is in progress or was "
                                            "interrupted; use resume() or a new key", retryable=False,
                                            details={"execution_id": prev.execution_id})
                    prev.idempotency_key = None  # superseded by this attempt
                    await self.store.save_execution(prev)
                    rec.attempts = prev.attempts + 1
            if is_root:
                await self._sem.acquire()
                sem_held = True
            self._inflight[ctx.execution_id] = ctx
            perr = await self._save(rec)
            if perr and self.strict_persistence:
                raise perr
            await self.bus.emit(Event("execution.resumed" if resumed else "execution.started", ctx.execution_id,
                                      ctx.parent_execution_id, ctx.correlation_id,
                                      {"agent_id": ctx.agent_id, "target": target, "depth": ctx.depth,
                                       "user_id": ctx.user_id, "tenant_id": ctx.tenant_id, "session_id": ctx.session_id}))
            await authorize(ctx, "workload.run", Resource("workload", target, attributes=dict(spec.meta)))
            await run_guardrails(self._input_guards, ctx, input)
            ctx.check()
            # 3. execute
            output = await self._run_workload(ctx, spec, input)
            await run_guardrails(self._output_guards, ctx, output)
            if isinstance(output, Handoff):
                res.status, res.handoff = Status.handoff, output
                await self.bus.emit(Event("handoff", ctx.execution_id, ctx.parent_execution_id, ctx.correlation_id,
                                          {"destination": output.destination, "reason": output.reason}))
            elif isinstance(output, Waiting):
                res.status, res.output, res.checkpoint_id = Status.waiting, output.payload, ctx.execution_id
            else:
                res.status, res.output = Status.completed, output
        except asyncio.CancelledError:
            res.status, res.error = Status.cancelled, ErrorInfo("CancelledError", "cancelled", "cancelled")
            await self._finalize(ctx, res, rec, usage0, started, target)
            raise  # the caller cancelled us: propagate after recording
        except Exception as exc:
            res.status, res.error = status_for(exc), ErrorInfo.from_exception(exc)
            if not isinstance(exc, DracoError):
                logger.error("execution_failed", extra={"execution_id": ctx.execution_id, "error": type(exc).__name__})
        finally:
            if sem_held:
                self._sem.release()
        await self._finalize(ctx, res, rec, usage0, started, target)
        return res

    async def _run_workload(self, ctx: ExecutionContext, spec: WorkloadSpec, input: Any) -> Any:
        async def body() -> Any:
            with use_context(ctx):
                return await spec.fn(ctx, input)

        loop = asyncio.get_running_loop()
        task = loop.create_task(body())

        def cancel_task() -> None:
            loop.call_soon_threadsafe(task.cancel)  # token.cancel() may be called from any thread
        ctx.cancel.on_cancel(cancel_task)
        timeout = ctx.time_left()
        rem = ctx.budget.remaining_duration()
        if rem is not None:
            timeout = rem if timeout is None else min(timeout, rem)
        try:
            done, _ = await asyncio.wait({task}, timeout=max(timeout, 0) if timeout is not None else None)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            from draco.core.errors import ExecutionTimeout
            raise ExecutionTimeout("execution deadline exceeded")
        if task.cancelled():
            from draco.core.errors import ExecutionCancelled
            raise ExecutionCancelled(ctx.cancel.reason or "cancelled")
        return task.result()

    async def _finalize(self, ctx: ExecutionContext, res: ExecutionResult, rec: ExecutionRecord,
                        usage0: dict[str, Any], started: float, target: str) -> None:
        self._inflight.pop(ctx.execution_id, None)
        res.finished_at = time.time()
        snap = ctx.budget.snapshot()
        res.usage = Usage(model_calls=int(snap["model_calls"] - usage0["model_calls"]),
                          tool_calls=int(snap["tool_calls"] - usage0["tool_calls"]),
                          retries=int(snap["retries"] - usage0["retries"]),
                          tokens=int(snap["tokens"] - usage0["tokens"]),
                          duration_ms=round((res.finished_at - started) * 1e3, 3))
        res.children = self._children.pop(ctx.execution_id, [])
        rec.status, rec.finished_at = res.status.value, res.finished_at
        rec.output = _jsonable(res.output) if res.status != Status.handoff else _jsonable(res.handoff)
        rec.error = _jsonable(res.error)
        rec.usage = _jsonable(res.usage)
        perr = await self._save(rec)
        if perr:
            await self._safe_emit(ctx, "persistence.failed", error=perr.message)
            if self.strict_persistence and res.status == Status.completed:
                res.status, res.error = Status.failed, ErrorInfo.from_exception(perr)
        data = {"status": res.status.value, "agent_id": ctx.agent_id, "target": target,
                "error_code": res.error.code if res.error else None, "usage": _jsonable(res.usage)}
        if res.status in (Status.failed, Status.cancelled, Status.timeout, Status.budget_exhausted):
            await self._safe_emit(ctx, {Status.failed: "execution.failed", Status.cancelled: "execution.cancelled",
                                        Status.timeout: "execution.timeout",
                                        Status.budget_exhausted: "execution.budget_exhausted"}[res.status], **data)
        await self._safe_emit(ctx, "execution.finished", **data)

    async def _safe_emit(self, ctx: ExecutionContext, name: str, **data: Any) -> None:
        try:
            await self.bus.emit(Event(name, ctx.execution_id, ctx.parent_execution_id, ctx.correlation_id, data))
        except Exception as exc:  # a critical handler vetoing after the fact cannot change the outcome
            logger.warning("post_execution_handler_failed", extra={"event": name, "error": type(exc).__name__})


def scope_key(key: str | None, tenant_id: str | None) -> str | None:
    """Idempotency keys live in a per-tenant namespace, so one tenant can never replay another's result."""
    return None if key is None else f"{tenant_id or '-'}\x1f{key}"


def _reserved_check(metadata: dict[str, Any] | None) -> None:
    clash = RESERVED_METADATA & set(metadata or {})
    if clash:
        raise PolicyViolation(f"metadata cannot set trusted fields {sorted(clash)}")


def _jsonable(v: Any) -> Any:
    import dataclasses
    import enum
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _jsonable(dataclasses.asdict(v))
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
