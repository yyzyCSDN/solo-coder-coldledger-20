from dataclasses import dataclass
from datetime import datetime,timezone
@dataclass(frozen=True)
class Sample: at:datetime; celsius:float
def exposure(samples,low,high):
    if len(samples)<2:return 0.0
    rows=sorted(samples,key=lambda x:x.at);score=0.0
    for a,b in zip(rows,rows[1:]):
        ta,tb=a.at.astimezone(timezone.utc),b.at.astimezone(timezone.utc);dt=(tb-ta).total_seconds()
        if dt<=0: raise ValueError('duplicate time')
        representative=a.celsius
        if representative<low: sev=(low-representative)**1.35
        elif representative>high: sev=(representative-high)**1.55
        else: sev=0
        score+=sev*dt/3600
    return score
