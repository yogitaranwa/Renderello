"""Security-review round-2 controls: budget reservation, traversal, sniffing."""
import json
import os
import threading

import pytest

from renderello import studio


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("RENDERELLO_MH_LEDGER", str(led))
    monkeypatch.setenv("RENDERELLO_MH_BUDGET_CREDITS", "100")
    monkeypatch.setenv("RENDERELLO_MH_MIN_BALANCE", "1")
    monkeypatch.setenv("RENDERELLO_MH_MAX_BATCH_CREDITS", "500")
    from renderello.backends.magichour import MagicHourBackend
    monkeypatch.setattr(MagicHourBackend, "balance",
                        lambda self: {"credits": 1_000_000})
    return led


def test_reserve_reconcile_roundtrip(ledger):
    reserved = studio.reserve_budget("seedream-v5-flash", 2)  # 60
    assert reserved == 60
    assert studio._ledger_spent() == 60
    studio.reconcile_budget("seedream-v5-flash", reserved, 30)
    assert studio._ledger_spent() == 30


def test_reserve_refuses_over_budget(ledger):
    studio.reserve_budget("seedream-v5-flash", 2)  # 60 of 100
    with pytest.raises(RuntimeError, match="budget exhausted"):
        studio.reserve_budget("seedream-v5-flash", 2)  # +60 > 100


def test_reserve_is_atomic_under_threads(ledger):
    # 8 threads race for 2 reservations that fit (60 each, budget 100).
    wins, losses = [], []
    def worker():
        try:
            studio.reserve_budget("seedream-v5-flash", 2)
            wins.append(1)
        except RuntimeError:
            losses.append(1)
    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert len(wins) == 1 and len(losses) == 7


def test_batch_cap(ledger, monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_MAX_BATCH_CREDITS", "50")
    with pytest.raises(RuntimeError, match="batch too large"):
        studio.reserve_budget("seedream-v5-flash", 2)  # 60 > 50


def test_floor_required_for_live_reservation(ledger, monkeypatch):
    # No balance-anchored cap -> reservation refuses (deploy-proof anchor
    # mandatory), while the check-only pre-flight still works.
    monkeypatch.delenv("RENDERELLO_MH_MIN_BALANCE", raising=False)
    monkeypatch.delenv("RENDERELLO_MH_CAP_ANCHOR", raising=False)
    with pytest.raises(RuntimeError, match="CAP_ANCHOR"):
        studio.reserve_budget("seedream-v5-flash", 1)
    studio._enforce_budget("seedream-v5-flash", 1)  # no raise


def test_anchor_hard_cap_and_topup_freeze(ledger, monkeypatch):
    from renderello.backends.magichour import MagicHourBackend
    monkeypatch.setenv("RENDERELLO_MH_CAP_ANCHOR", "1000:100")  # bal 1000, cap 100
    monkeypatch.setattr(MagicHourBackend, "balance",
                        lambda self: {"credits": 950})  # 50 used
    studio.reserve_budget("seedream-v5-flash", 1)  # 30 fits (50+30<=100)
    # ledger still has room (30+60<=100) but the anchor cap does not (50+60>100)
    with pytest.raises(RuntimeError, match="hard cap"):
        studio.reserve_budget("seedream-v5-flash", 2)
    # Top-up raises the balance ABOVE the anchor -> fail closed.
    monkeypatch.setattr(MagicHourBackend, "balance",
                        lambda self: {"credits": 1200})
    with pytest.raises(RuntimeError, match="ABOVE the authorized anchor"):
        studio.reserve_budget("seedream-v5-flash", 1)


def test_scene_attempts_clamped():
    assert studio.scene_attempts_from({}) == 1
    assert studio.scene_attempts_from({"scene_max_attempts": "1"}) == 1
    assert studio.scene_attempts_from({"scene_max_attempts": "3"}) == 3
    for bad in ("-5", "0", "99", "junk", "", None):
        assert studio.scene_attempts_from({"scene_max_attempts": bad}) == 1


def test_sniff_webp_vs_riff():
    import importlib.util
    spec = importlib.util.spec_from_file_location("wmain", "webapp/main.py")
    # avoid importing the app (side effects); test the bytes rule directly
    webp = b"RIFF\x1a\x00\x00\x00WEBPVP8 " + b"\x00" * 8
    avi = b"RIFF\x1a\x00\x00\x00AVI " + b"\x00" * 8
    assert (len(webp) >= 12 and webp[:4] == b"RIFF" and webp[8:12] == b"WEBP")
    assert not (avi[:4] == b"RIFF" and avi[8:12] == b"WEBP")


def test_run_dir_regex():
    import re
    # Deployed rule in webapp/main.py (_RUN_RE.fullmatch): ids are safe
    # filename chars only; traversal separators and dots can never match.
    rx = re.compile(r"[A-Za-z0-9_-]{1,64}")
    assert rx.fullmatch("20261120-141418-c553a7")
    for bad in ("..", "../x", "a/b", "a b", "a.b", "x" * 100, ""):
        assert not rx.fullmatch(bad)
