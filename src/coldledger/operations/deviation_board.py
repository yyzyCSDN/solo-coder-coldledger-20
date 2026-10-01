from dataclasses import dataclass,replace
from datetime import datetime
from coldledger.claims.excursions import DEFAULT_SEVERITY_THRESHOLDS,severity_from_exposure
@dataclass(frozen=True)
class Deviation:
    reference:str;severity:int;opened_at:datetime;owner:str;state:str='open';actions:tuple=();revision:int=0
VALID={'open','investigating','mitigating','verification','closed'}
def open_from_excursion(reference,excursion,owner,opened_at,thresholds=DEFAULT_SEVERITY_THRESHOLDS):
    """Open a case whose severity comes from the excursion's time integral."""
    return Deviation(reference,severity_from_exposure(excursion.exposure,thresholds),opened_at,owner)
def add_action(case,actor,text,at):
    if case.state=='closed':raise ValueError('closed case')
    if not actor or not text:raise ValueError('actor and text required')
    action=(actor,text,at.isoformat())
    return replace(case,actions=case.actions+(action,),revision=case.revision+1)
def transition(case,target,expected_revision):
    if expected_revision!=case.revision:raise RuntimeError('revision conflict')
    if target not in VALID:raise ValueError('unknown target')
    allowed={'open':{'investigating'},'investigating':{'mitigating','verification'},'mitigating':{'verification'},'verification':{'closed','mitigating'},'closed':set()}
    if target not in allowed[case.state]:raise ValueError('illegal transition')
    if target=='closed' and not case.actions:raise ValueError('closure requires evidence')
    return replace(case,state=target,revision=case.revision+1)
def aging_hours(case,now):
    return max(0.0,(now-case.opened_at).total_seconds()/3600)
def priority(case,now):
    age=min(aging_hours(case,now),720);return case.severity*100+age*(1+case.severity/10)
def rank(cases,now):
    return sorted(cases,key=lambda c:(-priority(c,now),c.reference))
