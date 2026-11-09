import asyncio

import pytest

from draco.core import (
    BudgetLimits,
    ContractError,
    DracoSettings,
    Handoff,
    Policy,
    PolicyViolation,
    Status,
    Waiting,
)
from draco.core.errors import DracoError, PersistenceError
from draco.harness import Harness, call_model, call_with_retry, step
from draco.harness.runner import scope_key
from draco.state import InMemoryStore, SqliteStore


def make(**kw):
    return Harness(DracoSettings(default_timeout_s=kw.pop("timeout", 5)), **kw)


async def test_success_result_and_trace():
    h = make()

    async def echo(ctx, x):
        return {"echo": x, "agent": ctx.agent_id}
    h.register("echo", echo)
    r = await h.run("echo", "hi", session_id="s1", user_id="u")
    assert r.status == Status.completed and r.output == {"echo": "hi", "agent": "echo"}
    assert r.session_id == "s1" and r.finished_at and r.usage.duration_ms >= 0
    names = [e["event"] for e in h.tracer.trace(r.execution_id)]
    assert names[0] == "execution.started" and names[-1] == "execution.finished"
    rec = await h.store.get_execution(r.execution_id)
    assert rec.status == "completed" and rec.output == {"echo": "hi", "agent": "echo"}
    assert [e["name"] for e in await h.store.list_events(r.execution_id)] == names


async def test_unknown_workload_and_agent_policy():
    h = make()
    h.register("a", lambda c, x: asyncio.sleep(0))
    r = await h.run("nope")
    assert r.status == Status.failed and r.error.code == "routing"
    r = await h.run("a", policy=Policy(allowed_agents=frozenset({"other"})))
    assert r.error.code == "policy"


async def test_guardrails_and_critical_hook_fail_closed_and_telemetry_isolated():
    h = make()
    ran = []

    async def w(ctx, x):
        ran.append(x)
        return x
    h.register("w", w)

    def boom(e):
        raise RuntimeError("exporter down")
    h.on("execution.*", boom)  # telemetry failure must not break the run
    assert (await h.run("w", 1)).status == Status.completed

    h.add_input_guardrail(lambda ctx, x: (_ for _ in ()).throw(PolicyViolation("blocked")) if x == "bad" else None)
    r = await h.run("w", "bad")
    assert r.status == Status.failed and r.error.code == "policy" and "bad" not in ran

    def veto(e):
        raise PolicyViolation("tenant suspended")
    h.on("execution.started", veto, critical=True)
    r = await h.run("w", 2)
    assert r.error.code == "policy" and 2 not in ran

    h2 = make()
    h2.register("w", w)
    h2.add_output_guardrail(lambda ctx, out: (_ for _ in ()).throw(ValueError("oops")))
    r = await h2.run("w", 3)
    assert r.error.code == "policy"  # a crashing guardrail blocks instead of passing


async def test_timeout_and_cancellation_outcomes():
    h = make()
    cleaned = []

    async def slow(ctx, x):
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.append(True)
    h.register("slow", slow)
    r = await h.run("slow", timeout_s=0.05)
    assert r.status == Status.timeout and r.error.retryable and cleaned == [True]

    cleaned.clear()
    t = asyncio.create_task(h.run("slow"))
    await asyncio.sleep(0.05)
    (eid,) = list(h._inflight)
    assert h.cancel(eid, "user stopped")
    r = await t
    assert r.status == Status.cancelled and cleaned == [True] and not h._inflight
    assert (await h.store.get_execution(eid)).status == "cancelled"


async def test_outer_task_cancellation_is_recorded_and_propagated():
    h = make()

    async def slow(ctx, x):
        await asyncio.sleep(10)
    h.register("slow", slow)
    t = asyncio.create_task(h.run("slow"))
    await asyncio.sleep(0.05)
    (eid,) = list(h._inflight)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert (await h.store.get_execution(eid)).status == "cancelled" and not h._inflight


async def test_budget_exhaustion_via_model_calls_and_usage():
    h = make()

    async def llm():
        return {"tokens": 7}

    async def agent(ctx, x):
        for _ in range(5):
            await call_model(ctx, llm, provider="fake", model="m", usage_of=lambda r: r["tokens"])
    h.register("agent", agent)
    r = await h.run("agent", limits=BudgetLimits(max_model_calls=3))
    assert r.status == Status.budget_exhausted and r.error.details["resource"] == "model_calls"
    assert r.usage.model_calls == 3 and r.usage.tokens == 21
    events = [e["event"] for e in h.tracer.trace(r.execution_id)]
    assert events.count("model.after") == 3 and "execution.budget_exhausted" in events


async def test_retry_only_for_retryable_and_charged():
    h = make()
    calls = {"n": 0}

    class Flaky(DracoError):
        retryable = True

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Flaky("again")
        return "ok"

    async def perm():
        calls["n"] += 1
        raise DracoError("permanent")

    async def w(ctx, x):
        return await call_with_retry(ctx, flaky, base_delay=0.001)

    async def p(ctx, x):
        return await call_with_retry(ctx, perm, base_delay=0.001)
    h.register("w", w)
    h.register("p", p)
    r = await h.run("w")
    assert r.output == "ok" and r.usage.retries == 2
    calls["n"] = 0
    r = await h.run("p")
    assert r.status == Status.failed and calls["n"] == 1  # permanent: no retry
    calls["n"] = 0
    r = await h.run("w", limits=BudgetLimits(max_retries=1))
    assert r.status == Status.budget_exhausted


async def test_nested_execution_parent_child_tree_and_depth_limit():
    h = make()

    async def leaf(ctx, x):
        return x * 2

    async def mid(ctx, x):
        r = await h.run("leaf", x, parent=ctx)
        return r.output + 1

    async def top(ctx, x):
        r = await h.run("mid", x, parent=ctx)
        return r.output

    async def deep(ctx, x):
        return (await h.run("deep", x, parent=ctx)).to_dict()
    for n, f in (("leaf", leaf), ("mid", mid), ("top", top), ("deep", deep)):
        h.register(n, f)
    r = await h.run("top", 5, session_id="S")
    assert r.output == 11 and len(r.children) == 1
    tree = h.tracer.tree(r.execution_id)
    assert [tree["agent_id"], tree["children"][0]["agent_id"], tree["children"][0]["children"][0]["agent_id"]] == ["top", "mid", "leaf"]
    leaf_rec = await h.store.get_execution(tree["children"][0]["children"][0]["execution_id"])
    assert leaf_rec.parent_execution_id == tree["children"][0]["execution_id"] and leaf_rec.session_id == "S"
    assert leaf_rec.correlation_id == r.correlation_id

    r = await h.run("deep", 1, limits=BudgetLimits(max_depth=2))
    chain, d = [], r.output
    while d:
        chain.append(d["status"])
        d = d["output"] if isinstance(d["output"], dict) else None
    assert chain[-1] == "budget_exhausted"  # recursion stopped by the depth limit


async def test_handoff_and_waiting_statuses():
    h = make()
    h.register("h", lambda c, x: asyncio.sleep(0, Handoff("other", "needs math", {"q": x})))
    h.register("w", lambda c, x: asyncio.sleep(0, Waiting("approval", {"id": 1})))
    r = await h.run("h", 3)
    assert r.status == Status.handoff and r.handoff.destination == "other"
    r = await h.run("w")
    assert r.status == Status.waiting and r.output == {"id": 1} and r.checkpoint_id == r.execution_id


async def test_idempotency_replay_does_not_reexecute():
    h = make()
    calls = []

    async def pay(ctx, x):
        calls.append(x)
        return {"charged": x}
    h.register("pay", pay)
    r1 = await h.run("pay", 10, idempotency_key="k1")
    r2 = await h.run("pay", 10, idempotency_key="k1")
    assert calls == [10] and r2.replayed and r2.output == r1.output and r2.execution_id == r1.execution_id


async def test_idempotency_after_failure_reruns_with_attempt_count():
    h = make()
    state = {"fail": True}

    async def w(ctx, x):
        if state["fail"]:
            raise ValueError("first try")
        return "fine"
    h.register("w", w)
    assert (await h.run("w", idempotency_key="k")).status == Status.failed
    state["fail"] = False
    r = await h.run("w", idempotency_key="k")
    assert r.status == Status.completed and not r.replayed
    assert (await h.store.find_by_key(scope_key("k", None))).attempts == 2


class BrokenStore(InMemoryStore):
    async def save_execution(self, rec):
        raise PersistenceError("disk full")


async def test_persistence_failure_semantics():
    async def w(ctx, x):
        return "done"
    lenient = make(store=BrokenStore())
    lenient.register("w", w)
    r = await lenient.run("w")
    assert r.status == Status.completed  # computed result is not discarded
    assert "persistence.failed" in [e["event"] for e in lenient.tracer.trace(r.execution_id)]
    strict = make(store=BrokenStore(), strict_persistence=True)
    strict.register("w", w)
    r = await strict.run("w")
    assert r.status == Status.failed and r.error.code == "persistence"


async def test_resume_replays_completed_steps_and_requires_reconciliation(tmp_path):
    store = SqliteStore(str(tmp_path / "state.db"))
    h = make(store=store)
    effects = []
    state = {"crash_after_charge": True}

    async def workflow(ctx, x):
        async def charge():
            effects.append("charge")
            return {"charge_id": "c1"}
        c = await step(ctx, "charge", charge)

        async def email():
            effects.append("email")
            if state["crash_after_charge"]:
                raise RuntimeError("crash mid-step")
            return "sent"
        return {"c": c, "e": await step(ctx, "email", email, idempotent=True)}
    h.register("wf", workflow)
    r = await h.run("wf", {"order": 1})
    assert r.status == Status.failed and effects == ["charge", "email"]
    state["crash_after_charge"] = False
    r2 = await h.resume(r.execution_id)
    assert r2.status == Status.completed and r2.execution_id == r.execution_id
    assert effects == ["charge", "email", "email"]  # charge NOT repeated; idempotent email retried
    assert (await store.get_execution(r.execution_id)).attempts == 2
    with pytest.raises(ContractError):
        await h.resume(r.execution_id)  # completed runs cannot be resumed

    # non-idempotent step interrupted mid-flight -> reconciliation, not blind replay
    h2 = make(store=store)

    async def wf2(ctx, x):
        async def send():
            raise RuntimeError("timeout talking to payment provider")
        return await step(ctx, "send", send)
    h2.register("wf2", wf2)
    r = await h2.run("wf2")
    r2 = await h2.resume(r.execution_id)
    assert r2.status == Status.failed and r2.error.code == "reconciliation_required"
    store.close()


async def test_concurrent_sessions_are_isolated():
    h = make()

    async def w(ctx, x):
        ctx.metadata["mine"] = x
        await asyncio.sleep(0.01 * (5 - x))
        from draco.core import current_context
        assert current_context() is ctx
        return (ctx.session_id, ctx.metadata["mine"], ctx.execution_id)
    h.register("w", w)
    results = await asyncio.gather(*(h.run("w", i, session_id=f"s{i}") for i in range(5)))
    assert [r.output[:2] for r in results] == [(f"s{i}", i) for i in range(5)]
    assert len({r.execution_id for r in results}) == 5


async def test_root_concurrency_limit():
    h = Harness(DracoSettings(max_concurrency=2))
    cur = {"n": 0, "max": 0}

    async def w(ctx, x):
        cur["n"] += 1
        cur["max"] = max(cur["max"], cur["n"])
        await asyncio.sleep(0.02)
        cur["n"] -= 1
    h.register("w", w)
    await asyncio.gather(*(h.run("w") for _ in range(6)))
    assert cur["max"] == 2


async def test_unexpected_exception_is_not_leaked():
    h = make()

    async def w(ctx, x):
        raise RuntimeError("password=hunter2 at /etc/secret")
    h.register("w", w)
    r = await h.run("w")
    assert r.status == Status.failed and "hunter2" not in r.error.message and "hunter2" not in str(r.to_dict())


async def test_user_ids_reach_policy_hooks_but_are_pseudonymised_in_traces_and_persisted_events():
    h = make()
    seen = []
    h.on("execution.started", lambda e: seen.append(e.data["user_id"]), critical=True)
    h.register("w", lambda ctx, x: asyncio.sleep(0, 1))
    r = await h.run("w", user_id="alice@example.com", session_id="s", tenant_id="acme")
    assert seen == ["alice@example.com"]  # enforcement code sees the real id
    blob = str(h.tracer.trace(r.execution_id)) + str(await h.store.list_events(r.execution_id))
    assert "alice@example.com" not in blob and "acme" not in blob and "id:" in blob


def test_redactor_secrets_tokens_and_truncation():
    from draco.observability import Redactor
    r = Redactor(max_chars=10).redact({"api_key": "sk-abcdefgh12345", "note": "Bearer abcdefghijklmnop ok", "x": "y" * 50,
                                       "nested": {"Password": "p", "n": [1, 2]}, "obj": object()})
    assert r["api_key"] == "<redacted>" and "abcdefgh" not in r["note"] and r["x"].startswith("yyyyyyyyyy...")
    assert r["nested"] == {"Password": "<redacted>", "n": [1, 2]} and r["obj"] == "<object>"


async def test_resume_requires_original_subject_within_same_tenant():
    from draco.core import Identity

    h = make()
    owner = Identity(subject="alice", tenant_id="acme")
    intruder = Identity(subject="bob", tenant_id="acme")

    async def fail(ctx, x):
        raise RuntimeError("interrupted")

    h.register("fail", fail)
    r = await h.run("fail", identity=owner)
    assert r.status == Status.failed

    with pytest.raises(ContractError, match="unknown execution"):
        await h.resume(r.execution_id, identity=intruder)

    resumed = await h.resume(r.execution_id, identity=owner)
    assert resumed.status == Status.failed
