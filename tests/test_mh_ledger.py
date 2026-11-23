"""Durable GitLab CAS ledger (mh_ledger) - offline unit tests with a mocked
HTTP layer, plus a real-API optimistic-concurrency test that only runs when
the ledger env is configured. The real test writes to a dedicated
nonsensitive self-test file that is PRESERVED (never deleted)."""
import json
import threading

import pytest

from renderello import mh_ledger, studio


@pytest.fixture
def cfg_env(monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_DURABLE_LEDGER", "1")
    monkeypatch.setenv("RENDERELLO_MH_LEDGER_PROJECT", "agent%2Fledger")
    monkeypatch.setenv("RENDERELLO_MH_LEDGER_TOKEN", "t0ken")
    monkeypatch.setenv("RENDERELLO_MH_LEDGER_FILE", "spend.json")
    monkeypatch.setenv("RENDERELLO_MH_LEDGER_BRANCH", "main")


class FakeAPI:
    """In-memory Repository Files API stand-in with CAS semantics."""

    def __init__(self, spent=0, conflicts=0):
        import base64
        self.spent = spent
        self.commit = 1
        self.conflicts_left = conflicts
        self.writes = 0
        self._b64 = base64

    def __call__(self, method, url, token, payload=None):
        if method == "GET":
            content = self._b64.b64encode(json.dumps(
                {"spent": self.spent, "updated": "t"}).encode()).decode()
            return 200, {"content": content, "last_commit_id": str(self.commit)}
        # PUT
        if self.conflicts_left:
            self.conflicts_left -= 1
            return 409, {"message": "conflict"}
        if payload["last_commit_id"] != str(self.commit):
            return 400, {"message": "stale"}
        self.spent = json.loads(payload["content"])["spent"]
        self.commit += 1
        self.writes += 1
        return 200, {}


def test_reserve_requires_config(monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_DURABLE_LEDGER", "1")
    monkeypatch.delenv("RENDERELLO_MH_LEDGER_PROJECT", raising=False)
    monkeypatch.delenv("RENDERELLO_MH_LEDGER_TOKEN", raising=False)
    with pytest.raises(mh_ledger.LedgerError):
        mh_ledger.reserve(30)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RENDERELLO_MH_DURABLE_LEDGER", raising=False)
    assert not mh_ledger.enabled()


def test_reserve_fail_closed_on_transport(cfg_env, monkeypatch):
    def boom(method, url, token, payload=None):
        raise mh_ledger.LedgerError("unreachable")
    monkeypatch.setattr(mh_ledger, "_request", boom)
    with pytest.raises(mh_ledger.LedgerError):
        mh_ledger.reserve(30)


def test_reserve_increments(cfg_env, monkeypatch):
    api = FakeAPI(spent=75)
    monkeypatch.setattr(mh_ledger, "_request", api)
    assert mh_ledger.reserve(30) == 105
    assert api.spent == 105


def test_cas_conflict_retries_and_lands(cfg_env, monkeypatch):
    api = FakeAPI(spent=10, conflicts=2)
    monkeypatch.setattr(mh_ledger, "_request", api)
    monkeypatch.setattr(mh_ledger.time, "sleep", lambda s: None)
    assert mh_ledger.reserve(5) == 15
    assert api.writes == 1


def test_cas_persistent_conflict_fails_closed(cfg_env, monkeypatch):
    api = FakeAPI(spent=10, conflicts=99)
    monkeypatch.setattr(mh_ledger, "_request", api)
    monkeypatch.setattr(mh_ledger.time, "sleep", lambda s: None)
    with pytest.raises(mh_ledger.LedgerError):
        mh_ledger.reserve(5)
    assert api.spent == 10  # nothing committed


def test_reconcile_equal_is_noop(cfg_env, monkeypatch):
    api = FakeAPI(spent=100)
    monkeypatch.setattr(mh_ledger, "_request", api)
    assert mh_ledger.reconcile(reserved=30, observed=30) == 100
    assert api.writes == 0  # no change, no write


def test_reconcile_down_subtracts(cfg_env, monkeypatch):
    api = FakeAPI(spent=100)
    monkeypatch.setattr(mh_ledger, "_request", api)
    assert mh_ledger.reconcile(reserved=30, observed=0) == 70
    assert api.spent == 70


def test_reconcile_upward_persists_overcharge(cfg_env, monkeypatch):
    # Listed price is a floor: an observed charge ABOVE the reservation is
    # real spend and MUST be added, or a restart would under-count it.
    api = FakeAPI(spent=100)
    monkeypatch.setattr(mh_ledger, "_request", api)
    assert mh_ledger.reconcile(reserved=30, observed=45) == 115
    assert api.spent == 115


def test_errors_never_carry_token(cfg_env, monkeypatch):
    def boom(method, url, token, payload=None):
        return 500, {"raw": "server said no"}
    monkeypatch.setattr(mh_ledger, "_request", boom)
    with pytest.raises(mh_ledger.LedgerError) as e:
        mh_ledger.reserve(30)
    assert "t0ken" not in str(e.value)
    assert "server said no" not in str(e.value)


# ---- studio.py wiring ----

@pytest.fixture
def ledger_env(cfg_env, monkeypatch, tmp_path):
    monkeypatch.setenv("RENDERELLO_MH_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.setenv("RENDERELLO_MH_BUDGET_CREDITS", "925")
    monkeypatch.setenv("RENDERELLO_MH_MAX_BATCH_CREDITS", "120")
    # Live spend also requires the balance anchor; anchor with zero spent.
    monkeypatch.setenv("RENDERELLO_MH_CAP_ANCHOR", "11431:925")
    monkeypatch.delenv("RENDERELLO_MH_MIN_BALANCE", raising=False)

    class FakeBackend:
        def __init__(self, model=None, live=False):
            pass

        def balance(self):
            return {"credits": 11431}

    monkeypatch.setattr(
        "renderello.backends.magichour.MagicHourBackend", FakeBackend)


def test_reserve_budget_durable_failure_blocks_local(ledger_env, monkeypatch):
    def boom(credits):
        raise mh_ledger.LedgerError("durable down")
    monkeypatch.setattr(mh_ledger, "reserve", boom)
    with pytest.raises(mh_ledger.LedgerError):
        studio.reserve_budget("seedream-v5-flash", 1)
    # local ledger must NOT hold a reservation the durable ledger lacks
    assert studio._ledger_spent() == 0


def test_reserve_budget_durable_success(ledger_env, monkeypatch):
    calls = []
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: calls.append(c) or c)
    monkeypatch.setattr(mh_ledger, "spent", lambda: 0)
    need = studio.reserve_budget("seedream-v5-flash", 1)
    assert need == 30 and calls == [30]
    assert studio._ledger_spent() == 30


def test_reconcile_failure_keeps_reservation(ledger_env, monkeypatch):
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    monkeypatch.setattr(mh_ledger, "spent", lambda: 0)
    studio.reserve_budget("seedream-v5-flash", 1)

    def boom(reserved, observed):
        raise mh_ledger.LedgerError("durable down")
    monkeypatch.setattr(mh_ledger, "reconcile", boom)
    studio.reconcile_budget("seedream-v5-flash", 30, 0)
    assert studio._ledger_spent() == 30  # reservation kept


def test_reconcile_unknown_charge_keeps_reservation(ledger_env, monkeypatch):
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    monkeypatch.setattr(mh_ledger, "spent", lambda: 0)
    studio.reserve_budget("seedream-v5-flash", 1)
    calls = []
    monkeypatch.setattr(mh_ledger, "reconcile",
                        lambda r, o: calls.append((r, o)) or 0)
    studio.reconcile_budget("seedream-v5-flash", 30, None)
    assert calls == []  # unknown observed charge: no downward write
    assert studio._ledger_spent() == 30


def test_overcharge_reconcile_records_positive_delta(ledger_env, monkeypatch):
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    monkeypatch.setattr(mh_ledger, "spent", lambda: 0)
    seen = []
    monkeypatch.setattr(mh_ledger, "reconcile",
                        lambda r, o: seen.append((r, o)) or 0)
    studio.reserve_budget("seedream-v5-flash", 1)
    studio.reconcile_budget("seedream-v5-flash", 30, 45)
    assert seen == [(30, 45)]
    assert studio._ledger_spent() == 45  # local ledger got the +15 too


def test_durable_max_wins_over_local(ledger_env, monkeypatch):
    monkeypatch.setattr(mh_ledger, "spent", lambda: 900)
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        studio.reserve_budget("seedream-v5-flash", 1)  # 900 + 30 > 925


def test_durable_mismatch_with_anchor_freezes(ledger_env, monkeypatch):
    # Same units: durable total must cover BASELINE + spend-since-anchor.
    # Baseline 75, anchor-derived 200 -> durable must be >= 275. 50 means
    # the durable ledger lost history -> freeze.
    monkeypatch.setenv("RENDERELLO_MH_CAP_ANCHOR", "11431:925")
    monkeypatch.setenv("RENDERELLO_MH_SPENT_BASELINE", "75")
    monkeypatch.setattr(mh_ledger, "spent", lambda: 50)

    class FakeBackend:
        def __init__(self, model=None, live=False):
            pass

        def balance(self):
            return {"credits": 11231}  # 11431 - 11231 = 200 spent

    monkeypatch.setattr(
        "renderello.backends.magichour.MagicHourBackend", FakeBackend)
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    with pytest.raises(RuntimeError, match="disagrees with the balance anchor"):
        studio.reserve_budget("seedream-v5-flash", 1)


def test_baseline_aligned_counters_do_not_freeze(ledger_env, monkeypatch):
    # Baseline 75 + anchor-derived 0 == durable 75: consistent, no freeze.
    monkeypatch.setenv("RENDERELLO_MH_SPENT_BASELINE", "75")
    monkeypatch.setattr(mh_ledger, "spent", lambda: 75)
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    assert studio.reserve_budget("seedream-v5-flash", 1) == 30


def test_config_coupling_total_1000_seed_75(ledger_env, monkeypatch):
    # TOTAL round cap 1000 with durable seeded at 75 -> 925 headroom, the
    # same figure the anchor 11231:925 authorizes since anchor. Setting
    # budget to 925 here would wrongly double-subtract the seed.
    monkeypatch.setenv("RENDERELLO_MH_BUDGET_CREDITS", "1000")
    monkeypatch.setenv("RENDERELLO_MH_MAX_BATCH_CREDITS", "1000")
    monkeypatch.setattr(mh_ledger, "spent", lambda: 75)
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    # 925 credits of headroom: 30 x 30 = 900 fits
    assert studio.reserve_budget("seedream-v5-flash", 30) == 900
    monkeypatch.setattr(studio, "_ledger_spent", lambda: 0)
    # 30 x 31 = 930 > 925 headroom -> refused
    with pytest.raises(RuntimeError, match="budget exhausted"):
        studio.reserve_budget("seedream-v5-flash", 31)


def test_budget_status_includes_durable(ledger_env, monkeypatch):
    monkeypatch.setattr(mh_ledger, "spent", lambda: 75)
    st = studio.budget_status()
    assert st["durable_spent"] == 75
    assert st["spent"] == 75


def test_anchor_required_for_live_reserve(cfg_env, monkeypatch, tmp_path):
    # Durable ledger enabled but NO balance anchor/floor: refuse.
    monkeypatch.setenv("RENDERELLO_MH_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.setenv("RENDERELLO_MH_BUDGET_CREDITS", "1000")
    monkeypatch.delenv("RENDERELLO_MH_CAP_ANCHOR", raising=False)
    monkeypatch.delenv("RENDERELLO_MH_MIN_BALANCE", raising=False)
    monkeypatch.setattr(mh_ledger, "spent", lambda: 75)
    with pytest.raises(RuntimeError, match="CAP_ANCHOR unset"):
        studio.reserve_budget("seedream-v5-flash", 1)


def test_restart_balance_lower_than_counted_rejects(ledger_env, monkeypatch):
    # Scenario: a paid call was counted (durable 105 = baseline 75 + 30),
    # the process restarted (_LAST_BALANCE reset), and the account balance
    # now shows a LARGER cumulative drop than the durable ledger recorded
    # (anchor 11431 -> balance 11341 = 90 since anchor; baseline 75 + 90 =
    # 165 > durable 105). Reserve must REJECT (freeze), not silently trust
    # the smaller durable figure.
    monkeypatch.setenv("RENDERELLO_MH_SPENT_BASELINE", "75")
    monkeypatch.setenv("RENDERELLO_MH_CAP_ANCHOR", "11431:925")
    studio._LAST_BALANCE["value"] = None  # restarted process
    monkeypatch.setattr(mh_ledger, "spent", lambda: 105)

    class FakeBackend:
        def __init__(self, model=None, live=False):
            pass

        def balance(self):
            return {"credits": 11341}

    monkeypatch.setattr(
        "renderello.backends.magichour.MagicHourBackend", FakeBackend)
    with pytest.raises(RuntimeError, match="disagrees with the balance anchor"):
        studio.reserve_budget("seedream-v5-flash", 1)


# ---- live model allowlist ----

def test_non_allowlisted_model_refused_live(ledger_env, monkeypatch):
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    monkeypatch.setattr(mh_ledger, "spent", lambda: 0)
    with pytest.raises(RuntimeError, match="not on the live allowlist"):
        studio.reserve_budget("nano-banana-2.1", 1)
    assert studio._ledger_spent() == 0


def test_allowlist_env_override(ledger_env, monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_LIVE_MODELS", "seedream-v5-flash,nano-banana-2.1")
    monkeypatch.setattr(mh_ledger, "reserve", lambda c: c)
    monkeypatch.setattr(mh_ledger, "spent", lambda: 0)
    assert studio.reserve_budget("nano-banana-2.1", 1) == 50


# ---- post-call charge: balance cross-check ----

class _FakeBackendState:
    def __init__(self, submission_charge=None, final_charge=None):
        self.last_result = {}
        if submission_charge is not None:
            self.last_result["submission"] = {"credits_charged": submission_charge}
        if final_charge is not None:
            self.last_result["final"] = {"status": "complete",
                                         "credits_charged": final_charge}


def _balance_stub(monkeypatch, value):
    class B:
        def __init__(self, model=None, live=False):
            pass

        def balance(self):
            if isinstance(value, Exception):
                raise value
            return {"credits": value}

    monkeypatch.setattr("renderello.backends.magichour.MagicHourBackend", B)


def test_post_call_normal_observed(monkeypatch):
    studio._LAST_BALANCE["value"] = 11261
    _balance_stub(monkeypatch, 11231)  # drop 30, observed 30
    charge, abort = studio._post_call_charge(_FakeBackendState(30), 30, "m")
    assert charge == 30 and abort is None


def test_post_call_balance_drop_exceeds_observed(monkeypatch):
    studio._LAST_BALANCE["value"] = 11261
    _balance_stub(monkeypatch, 11211)  # drop 50, observed 30
    charge, abort = studio._post_call_charge(_FakeBackendState(30), 30, "m")
    assert charge == 50  # the larger figure is counted
    assert "dropped 50" in abort


def test_post_call_refund_never_reduces(monkeypatch):
    studio._LAST_BALANCE["value"] = 11261
    _balance_stub(monkeypatch, 11241)  # drop 20 < observed 30
    charge, abort = studio._post_call_charge(_FakeBackendState(30), 30, "m")
    assert charge == 30 and abort is None


def test_post_call_balance_unreadable_aborts(monkeypatch):
    studio._LAST_BALANCE["value"] = 11261
    _balance_stub(monkeypatch, RuntimeError("net down"))
    charge, abort = studio._post_call_charge(_FakeBackendState(30), 30, "m")
    assert charge == 30 and "unreadable" in abort


def test_post_call_unknown_charge_falls_back_listed(monkeypatch):
    studio._LAST_BALANCE["value"] = 11261
    _balance_stub(monkeypatch, 11261)  # no drop, nothing observed
    charge, abort = studio._post_call_charge(_FakeBackendState(), 30, "m")
    assert charge == 30 and abort is None


def test_post_call_observed_above_listed_aborts(monkeypatch):
    studio._LAST_BALANCE["value"] = 11261
    _balance_stub(monkeypatch, 11221)  # drop 40, observed 40 > listed 30
    charge, abort = studio._post_call_charge(_FakeBackendState(40), 30, "m")
    assert charge == 40 and abort is not None


# ---- real API optimistic-concurrency proof (env-gated) ----

def test_real_api_concurrent_cas(monkeypatch):
    """Two threads hammer the SAME durable file with +1 reserves; strict CAS
    must land every increment exactly once. Runs only when the ledger env is
    live; targets a dedicated nonsensitive self-test file that is kept."""
    import os
    if not (os.environ.get("RENDERELLO_MH_LEDGER_TOKEN")
            and os.environ.get("RENDERELLO_MH_LEDGER_PROJECT")):
        pytest.skip("durable ledger env not configured")
    monkeypatch.setenv("RENDERELLO_MH_DURABLE_LEDGER", "1")
    monkeypatch.setenv("RENDERELLO_MH_LEDGER_FILE", "cas-selftest.json")
    cfg = mh_ledger._config()
    before = mh_ledger.read_state(cfg)["spent"]
    errors = []

    def bump():
        try:
            for _ in range(5):
                mh_ledger.reserve(1)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=bump) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    after = mh_ledger.read_state(cfg)["spent"]
    assert after == before + 10
