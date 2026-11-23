"""Original-brief acceptance/economics accounting, without invented ratings.

The brief proposes80% no critical alteration and70% publish-usable. Those are
separate gates. Every preregistered first-try output remains in the denominator,
including failed generations and missing reviews. Software execution is not a
publishable result. A reported rating/provenance is a claim, not authentication.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from decimal import Decimal
import math
import statistics
from .review import DIMENSIONS


def _id(value, name):
    if not isinstance(value,str) or not value.strip() or len(value)>240:
        raise ValueError(name+' must be a short nonempty identifier')


def _number(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError(name+' must be finite nonnegative number')


@dataclass(frozen=True)
class PlannedOutput:
    output_id: str
    source_id: str
    template_id: str
    method: str
    alternative: int
    first_try: bool = True
    source_sha256: str | None = None

    def __post_init__(self):
        for name in ('output_id','source_id','template_id','method'):_id(getattr(self,name),name)
        if type(self.alternative) is not int or not 0<=self.alternative<=100:
            raise ValueError('bounded integer alternative required')
        if type(self.first_try) is not bool:raise ValueError('first_try must be boolean')
        if self.source_sha256 is not None and (not isinstance(self.source_sha256,str) or len(self.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.source_sha256)):
            raise ValueError('source hash must be exact lowercase sha256')


@dataclass(frozen=True)
class Attempt:
    output_id: str
    attempt_id: str
    status: str
    seconds: float | None
    credits_charged: float | None
    output_sha256: str | None = None
    failure: str = ''

    def __post_init__(self):
        _id(self.output_id,'output_id');_id(self.attempt_id,'attempt_id')
        if self.status not in ('success','failed','unknown'):raise ValueError('explicit attempt status required')
        for name in ('seconds','credits_charged'):
            value=getattr(self,name)
            if value is not None:_number(value,name)
        if self.status=='success':
            if not isinstance(self.output_sha256,str) or len(self.output_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.output_sha256):
                raise ValueError('successful artifact needs exact lowercase sha256')
            if self.failure:raise ValueError('success cannot have failure text')
        elif self.output_sha256 is not None:
            raise ValueError('failed/unknown attempt cannot carry a successful output hash')
        elif not isinstance(self.failure,str) or not self.failure.strip():
            raise ValueError('failed/unknown attempt needs reason')


@dataclass(frozen=True)
class Rating:
    output_id: str
    output_sha256: str
    scores: dict
    no_critical: bool
    critical_errors: tuple[str,...]
    provenance: str
    review_reference: str

    def __post_init__(self):
        _id(self.output_id,'output_id');_id(self.review_reference,'review_reference')
        if not isinstance(self.scores,Mapping) or set(self.scores)!=set(DIMENSIONS):
            raise ValueError('all seven scores required exactly')
        if any(type(v) is not int or not 1<=v<=5 for v in self.scores.values()):
            raise ValueError('integer scores1..5 required')
        object.__setattr__(self,'scores',MappingProxyType(dict(self.scores)))
        if type(self.no_critical) is not bool:raise ValueError('explicit boolean critical decision required')
        if type(self.critical_errors) is not tuple or any(not isinstance(x,str) or not x.strip() for x in self.critical_errors):
            raise ValueError('explicit critical-error tuple required')
        if self.no_critical == bool(self.critical_errors):raise ValueError('critical decision conflicts with errors')
        if self.provenance not in ('assistant_judgment','owner','claimed_independent'):
            raise ValueError('explicit review provenance required')



def _artifact_matching_count(graph):
    """One distinct artifact per coverage identity; repeated bytes cannot pad it.

    Iterative augmenting paths avoid recursion limits. This establishes byte
    diversity only, not visual variety, source authorship or template quality.
    """
    from collections import deque
    assigned = {}
    for identity in sorted(graph):
        queue=deque([identity]);seen={identity};parent={};end=None
        while queue and end is None:
            node=queue.popleft()
            for artifact in sorted(graph[node]):
                if artifact in parent:continue
                parent[artifact]=node
                if artifact not in assigned:
                    end=artifact;break
                other=assigned[artifact]
                if other not in seen:seen.add(other);queue.append(other)
        if end is not None:
            reverse={node:artifact for artifact,node in assigned.items()}
            while True:
                node=parent[end];previous=reverse.get(node);assigned[end]=node
                if previous is None:break
                end=previous
    return len(assigned)


def evaluate(plan, attempts, ratings, *, publish_threshold=4, minimum_sources=20,
             minimum_templates=8, competing_specialists=()):
    """Complete accounting, not an automatic deployment or spending decision.

    A score>=4 is a declared operational interpretation of publish-usable, not a
    threshold stated in the brief. Return it prominently so it cannot be hidden.
    Plan methods must be analyzed separately; mixed methods are refused.
    """
    if not plan or any(not isinstance(x,PlannedOutput) for x in plan):raise ValueError('nonempty typed plan required')
    if len({x.output_id for x in plan})!=len(plan):raise ValueError('duplicate planned output')
    if len({(x.source_id,x.template_id,x.alternative) for x in plan})!=len(plan):
        raise ValueError('duplicate logical source/template/alternative slot')
    source_hashes = {}
    for x in plan:
        if x.source_id in source_hashes and source_hashes[x.source_id] != x.source_sha256:
            raise ValueError('same source identifier has conflicting hash')
        source_hashes[x.source_id] = x.source_sha256
    if len({x.method for x in plan})!=1:raise ValueError('evaluate one method at a time')
    for name,value,upper in [('publish_threshold',publish_threshold,5),('minimum_sources',minimum_sources,10000),('minimum_templates',minimum_templates,1000)]:
        if type(value) is not int or not 1<=value<=upper:raise ValueError('invalid '+name)
    ids={x.output_id for x in plan};by_id={x.output_id:[] for x in plan};seen=set()
    for x in attempts:
        if not isinstance(x,Attempt) or x.output_id not in ids:raise ValueError('unplanned attempt')
        if x.attempt_id in seen:raise ValueError('duplicate attempt id')
        seen.add(x.attempt_id);by_id[x.output_id].append(x)
    rated={}
    for r in ratings:
        if not isinstance(r,Rating) or r.output_id not in ids:raise ValueError('unplanned rating')
        if r.output_id in rated:raise ValueError('multiple ratings require an explicit aggregation policy')
        successes=[a for a in by_id[r.output_id] if a.status=='success']
        if not any(a.output_sha256==r.output_sha256 for a in successes):raise ValueError('rating does not bind a successful output')
        planned = next(p for p in plan if p.output_id == r.output_id)
        if planned.first_try and (not by_id[r.output_id] or by_id[r.output_id][0].status != 'success' or by_id[r.output_id][0].output_sha256 != r.output_sha256):
            raise ValueError('first-try plan cannot substitute a later successful retry')
        rated[r.output_id]=r
    count=len(plan);complete=len(rated)==count
    no_critical=sum(r.no_critical for r in rated.values())
    usable=sum(r.scores['publish_ready_usefulness']>=publish_threshold and r.no_critical for r in rated.values())
    # Exact rational comparison avoids rounding79.95% up to80%.
    fidelity_pass=no_critical*100>=80*count
    useful_pass=usable*100>=70*count
    bound_sources={h for h in source_hashes.values() if h is not None}
    planned_coverage=len(bound_sources)>=minimum_sources and len({p.template_id for p in plan})>=minimum_templates
    eligible={}
    for p in plan:
        history=by_id[p.output_id]
        selected=history[:1] if p.first_try else history
        eligible[p.output_id]={a.output_sha256 for a in selected if a.status=='success'}
    generated_source_graph={};generated_template_graph={}
    reviewed_source_graph={};reviewed_template_graph={}
    for p in plan:
        hashes=eligible[p.output_id]
        if hashes:
            generated_template_graph.setdefault(p.template_id,set()).update(hashes)
            if p.source_sha256 is not None:
                generated_source_graph.setdefault(p.source_sha256,set()).update(hashes)
        if p.output_id in rated:
            h=rated[p.output_id].output_sha256
            reviewed_template_graph.setdefault(p.template_id,set()).add(h)
            if p.source_sha256 is not None:
                reviewed_source_graph.setdefault(p.source_sha256,set()).add(h)
    source_evidence=_artifact_matching_count(reviewed_source_graph)
    template_evidence=_artifact_matching_count(reviewed_template_graph)
    coverage=source_evidence>=minimum_sources and template_evidence>=minimum_templates
    successful_artifact_hashes=[a.output_sha256 for a in attempts if a.status=='success']
    specialist_names=set(competing_specialists)
    if len(specialist_names)!=len(competing_specialists) or any(not isinstance(x,str) or not x.strip() for x in specialist_names):
        raise ValueError('unique explicit specialist identities required')
    # Merely naming competitors does not establish a matched blind study.
    blockers=[]
    if not complete:blockers.append('INCOMPLETE_REVIEWS')
    if not coverage:blockers.append('SOURCE_OR_TEMPLATE_COVERAGE_BELOW_BRIEF')
    if not fidelity_pass:blockers.append('NO_CRITICAL_ALTERATION_GATE_NOT_ESTABLISHED')
    if not useful_pass:blockers.append('PUBLISH_USABLE_GATE_NOT_ESTABLISHED')
    blockers.append('TEMPLATE_VISUAL_DISTINCTNESS_REQUIRES_REVIEW')
    blockers.append('MATCHED_SPECIALIST_COMPETITIVENESS_REQUIRES_REVIEW')
    times=[a.seconds for a in attempts if a.seconds is not None]
    known_credits=sum(Decimal(str(a.credits_charged)) for a in attempts if a.credits_charged is not None)
    costs_complete=all(a.credits_charged is not None for a in attempts) and bool(attempts)
    missing_attempts=sum(not by_id[p.output_id] for p in plan)
    generated=sum(any(a.status=='success' for a in by_id[p.output_id]) for p in plan)
    success_times=[a.seconds for a in attempts if a.status=='success' and a.seconds is not None]
    return dict(method=plan[0].method,planned_outputs=count,distinct_sources=len({p.source_id for p in plan}),
        bound_distinct_source_hashes=len(bound_sources),unbound_source_ids=sum(h is None for h in source_hashes.values()),
        source_hash_semantics='declared_file_hash_not_proof_of_source_variety_or_authorship',
        distinct_templates=len({p.template_id for p in plan}),first_try_planned=sum(p.first_try for p in plan),
        planned_coverage_gate=planned_coverage,
        generated_bound_source_hashes=len(generated_source_graph),generated_template_ids=len(generated_template_graph),
        reviewed_bound_source_hashes=len(reviewed_source_graph),reviewed_template_ids=len(reviewed_template_graph),
        generated_artifact_bound_sources=_artifact_matching_count(generated_source_graph),
        generated_artifact_bound_templates=_artifact_matching_count(generated_template_graph),
        reviewed_artifact_bound_sources=source_evidence,reviewed_artifact_bound_templates=template_evidence,
        unique_successful_artifact_hashes=len(set(successful_artifact_hashes)),
        repeated_successful_artifact_occurrences=len(successful_artifact_hashes)-len(set(successful_artifact_hashes)),
        eligible_generated_output_slots=sum(bool(v)for v in eligible.values()),
        coverage_semantics='reviewed_source_and_template_identities_each_matched_to_distinct_artifact_bytes_not_proof_of_visual_variety',
        distinct_templates_semantics='planned_labels_not_polished_or_visually_distinct_template_count',
        template_visual_distinctness='NOT_ESTABLISHED', 
        generated_outputs=generated,missing_attempt_outputs=missing_attempts,
        failed_attempts=sum(a.status=='failed' for a in attempts),unknown_attempts=sum(a.status=='unknown' for a in attempts),
        retry_attempts=sum(max(0,len(v)-1) for v in by_id.values()),
        rated_outputs=len(rated),unrated_outputs=count-len(rated),rating_provenance=sorted({r.provenance for r in ratings}),
        no_critical_count=no_critical,publish_usable_count=usable,
        no_critical_fraction=no_critical/count,publish_usable_fraction=usable/count,
        fraction_semantics='confirmed_passes_over_all_planned_outputs_unrated_not_assumed_good',
        publish_score_threshold=publish_threshold,publish_threshold_origin='caller_policy_not_explicit_brief_threshold',
        fidelity_gate=fidelity_pass,usefulness_gate=useful_pass,coverage_gate=coverage,
        all_reviews_complete=complete,named_specialists=sorted(specialist_names),
        blind_competitive_acceptance='NOT_ESTABLISHED',blockers=blockers,production_go=False,
        attempts=len(attempts),latency_samples=len(times),latency_complete=len(times)==len(attempts) and bool(attempts),
        median_attempt_seconds=statistics.median(times) if times else None,
        successful_latency_samples=len(success_times),
        median_success_attempt_seconds=statistics.median(success_times) if success_times else None,
        charged_credits_known=float(known_credits),charged_credits_complete=costs_complete,
        credits_per_generated_output=float(known_credits/generated) if costs_complete and generated else None,
        credits_per_publish_usable_output=float(known_credits/usable) if costs_complete and usable and complete else None,
        price_and_margin='NOT_COMPUTED_REQUIRES_CURRENT_RATE_AND_SALE_PRICE',
        authority='ACCOUNTING_ONLY_NO_SEND_SPEND_OR_DEPLOY_AUTHORITY')
