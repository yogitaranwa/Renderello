"""Evidence-bound original-brief economics. Offline; never authorizes generation.

Legacy costs.py remains a published-catalog projection. This path uses typed
attempt records from brief_acceptance, keeps missing charges unknown, and
requires explicitly dated pricing evidence. It cannot turn ratings into sales.
"""
from dataclasses import dataclass, asdict
from datetime import date
from decimal import Decimal, InvalidOperation
import argparse
import json
from pathlib import Path
from .brief_acceptance import PlannedOutput, Attempt, Rating, evaluate


def amount(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(name + ' must be a decimal amount')
    try:
        d = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(name + ' must be a decimal amount') from exc
    if not d.is_finite() or d < 0 or (positive and d == 0):
        raise ValueError(name + ' must be finite and ' + ('positive' if positive else 'nonnegative'))
    if d.adjusted() > 15 or d.as_tuple().exponent < -12:
        raise ValueError(name + ' exceeds decimal precision budget')
    return d


def text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(name + ' requires a bounded evidence reference')


@dataclass(frozen=True)
class PriceEvidence:
    currency: str
    per_credit: str
    rate_basis: str
    source_reference: str
    observed_on: str
    valid_through: str
    credit_system: str

    def __post_init__(self):
        if not isinstance(self.currency, str) or len(self.currency) != 3 or not self.currency.isascii() or not self.currency.isupper() or not self.currency.isalpha():
            raise ValueError('explicit three-letter currency required')
        amount(self.per_credit, 'per_credit', positive=True)
        if self.rate_basis not in ('included_subscription_allocation', 'purchased_credit_pack', 'usage_invoice'):
            raise ValueError('explicit non-free credit valuation basis required')
        text(self.source_reference, 'price source')
        text(self.credit_system, 'credit system')
        try:
            start, end = date.fromisoformat(self.observed_on), date.fromisoformat(self.valid_through)
        except (TypeError, ValueError) as exc:
            raise ValueError('ISO observation/validity dates required') from exc
        if start > end:
            raise ValueError('price validity ends before observation')


def build_report(plan, attempts, ratings, *, price=None, as_of, sale_price=None,
                 sale_reference=None, other_cost_per_usable=None,
                 other_cost_reference=None, publish_threshold=4, charge_credit_system=None):
    """Report a scenario, not actual profit, a current vendor quote or go decision.

    Missing first attempts and unknown statuses prevent a complete-cost claim.
    All attempted charges (including failed and retried jobs) stay in numerator.
    Distinct physical artifact hashes are the denominators, never renamed slots.
    Other per-usable costs are mandatory for a contribution-margin scenario;
    explicit zero is accepted only with an accompanying evidence/assumption note.
    """
    try:
        day = date.fromisoformat(as_of)
    except (TypeError, ValueError) as exc:
        raise ValueError('explicit ISO as_of date required') from exc
    plan, attempts, ratings = list(plan), list(attempts), list(ratings)
    summary = evaluate(plan, attempts, ratings, publish_threshold=publish_threshold)
    if price is not None and not isinstance(price, PriceEvidence):
        raise ValueError('typed dated pricing evidence required')
    if charge_credit_system is not None: text(charge_credit_system, 'charge credit system')
    if price is not None and price.credit_system != charge_credit_system:
        raise ValueError('price and charge credit systems must explicitly match')
    if (sale_price is None) != (sale_reference is None):
        raise ValueError('sale scenario and reference must be supplied together')
    if (other_cost_per_usable is None) != (other_cost_reference is None):
        raise ValueError('other-cost scenario and reference must be supplied together')
    sale = None if sale_price is None else amount(sale_price, 'sale price', positive=True)
    overhead = None if other_cost_per_usable is None else amount(other_cost_per_usable, 'other cost')
    if sale is not None: text(sale_reference, 'sale price')
    if overhead is not None: text(other_cost_reference, 'other cost')
    complete = (bool(attempts) and summary['missing_attempt_outputs'] == 0
                and summary['unknown_attempts'] == 0 and summary['charged_credits_complete'])
    known = sum((amount(a.credits_charged, 'charged credits') for a in attempts if a.credits_charged is not None), Decimal(0))
    successful = {a.output_sha256 for a in attempts if a.status == 'success'}
    reviewed_hashes = {r.output_sha256 for r in ratings}
    artifact_reviews_complete = bool(successful) and successful <= reviewed_hashes
    usable = {r.output_sha256 for r in ratings if r.no_critical and r.scores['publish_ready_usefulness'] >= publish_threshold}
    eligible = {a.output_id: [] for a in attempts}
    for a in attempts: eligible[a.output_id].append(a)
    selected = []
    for p in plan:
        history = eligible.get(p.output_id, [])
        selected.extend(history[:1] if p.first_try else history)
    first_eligible = {a.output_sha256 for a in selected if a.status == 'success'}
    usable &= first_eligible
    # Identical bytes with conflicting reviews are not a usable-unit shortcut.
    rejected_hashes = {r.output_sha256 for r in ratings if not r.no_critical or r.scores['publish_ready_usefulness'] < publish_threshold}
    conflicting_hashes = usable & rejected_hashes
    usable -= rejected_hashes
    blockers = []
    if conflicting_hashes: blockers.append('CONFLICTING_DUPLICATE_ARTIFACT_REVIEWS')
    if not complete: blockers.append('INCOMPLETE_ATTEMPT_OR_CHARGE_EVIDENCE')
    if not summary['all_reviews_complete']: blockers.append('INCOMPLETE_PLANNED_SLOT_REVIEWS_SEE_ACCEPTANCE')
    if not artifact_reviews_complete: blockers.append('INCOMPLETE_SUCCESSFUL_ARTIFACT_REVIEWS')
    if not successful: blockers.append('NO_SUCCESSFUL_ARTIFACTS')
    if not usable: blockers.append('NO_REVIEWED_USABLE_ARTIFACTS')
    fresh = price is not None and date.fromisoformat(price.observed_on) <= day <= date.fromisoformat(price.valid_through)
    if price is None: blockers.append('PRICE_EVIDENCE_MISSING')
    elif not fresh: blockers.append('PRICE_EVIDENCE_OUTSIDE_DECLARED_VALIDITY')
    total = known * amount(price.per_credit, 'per_credit') if complete and fresh else None
    per_generated = total / len(successful) if total is not None and successful else None
    per_usable = total / len(usable) if total is not None and usable and artifact_reviews_complete else None
    if sale is None: blockers.append('SALE_PRICE_SCENARIO_MISSING')
    if overhead is None: blockers.append('OTHER_COST_SCENARIO_MISSING')
    margin = sale - per_usable - overhead if sale is not None and per_usable is not None and overhead is not None else None
    dec = lambda v: None if v is None else format(v, 'f')
    return dict(schema='brief-economics.v1', as_of=as_of, method=plan[0].method,
        acceptance=summary, pricing=None if price is None else asdict(price), price_in_declared_validity=fresh,
        evidence_semantics='caller_supplied_records_not_provider_or_reviewer_authentication',
        charge_credit_system=charge_credit_system, credit_scope_semantics='caller_declared_single_credit_system_for_all_attempts; split_provider_or_account_denominations_before_reporting',
        charge_evidence_complete=complete, known_charged_credits=dec(known),
        all_successful_artifacts_reviewed=artifact_reviews_complete,
        distinct_successful_artifacts=len(successful), distinct_usable_artifacts=len(usable),
        successful_artifact_semantics='any_recorded_attempt_including_retries_not_first_try_quality_rate',
        total_attempt_credit_cost=dec(total), cost_per_distinct_generated_artifact=dec(per_generated),
        cost_per_distinct_reviewed_usable_artifact=dec(per_usable),
        sale_price_scenario=dec(sale), sale_reference=sale_reference,
        other_cost_per_usable_scenario=dec(overhead), other_cost_reference=other_cost_reference,
        contribution_per_usable_scenario=dec(margin),
        contribution_margin_fraction_scenario=dec(None if margin is None else margin / sale),
        margin_semantics='hypothetical_per_reviewed_usable_delivery_not_actual_revenue_or_profit; other_cost_input_must_cover_relevant_infrastructure_support_fees',
        zero_recorded_credits_semantics='recorded_free_test_not_proof_of_zero_future_serving_cost',
        best_quality_model=None, value_fallback=None,
        recommendation='DEFERRED_UNTIL_MATCHED_QUALITY_AND_SPECIALIST_REVIEW',
        blockers=blockers, production_go=False, live_calls=0,
        authority='NO_SEND_SPEND_SUBSCRIBE_OR_DEPLOY_AUTHORITY')


def from_payload(payload, *, evidence_path=None):
    """Strict reusable adapter for existing brief_acceptance records."""
    if not isinstance(payload, dict): raise ValueError('object payload required')
    if 'evidence_pair_required' in payload:
        if payload['evidence_pair_required'] is not True or evidence_path is None:
            raise ValueError('paired provider export requires evidence_path and complete-pair validation')
        from .evidence_pair import validate_pair
        validate_pair(evidence_path)
        if json.loads(Path(evidence_path).read_text())!=payload:
            raise ValueError('supplied payload differs from validated pair')
        payload={k:v for k,v in payload.items() if k!='evidence_pair_required'}
    if set(payload) - {'plan', 'attempts', 'ratings', 'price', 'as_of', 'sale_price', 'sale_reference', 'other_cost_per_usable', 'other_cost_reference', 'publish_threshold', 'charge_credit_system'}:
        raise ValueError('unknown report fields')
    plan = [PlannedOutput(**r) for r in payload['plan']]
    attempts = [Attempt(**r) for r in payload['attempts']]
    ratings = []
    for r in payload['ratings']:
        r = dict(r); r['critical_errors'] = tuple(r['critical_errors']); ratings.append(Rating(**r))
    opts = {k: v for k, v in payload.items() if k not in ('plan', 'attempts', 'ratings', 'price')}
    price = None if payload.get('price') is None else PriceEvidence(**payload['price'])
    return build_report(plan, attempts, ratings, price=price, **opts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.input.is_symlink() or not a.input.is_file() or a.input.stat().st_size > 2_000_000:
            raise ValueError('bounded regular input JSON required')
        report = from_payload(json.loads(a.input.read_text()), evidence_path=a.input)
        with a.output.open('x') as f: json.dump(report, f, indent=2)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        p.exit(2, 'REFUSED: ' + str(exc) + '\n')
    print('Saved offline economics report: ' + str(a.output))


if __name__ == '__main__': main()
