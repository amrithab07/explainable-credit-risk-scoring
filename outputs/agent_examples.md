# Agent examples

### Applicant 12997 | model p(default) = 0.546 | cut-off = 0.14 | decision = DECLINE ###
tools called: ['score_applicant'] | source: openai-compatible | verified: True

Decision: **DECLINE**.  
Default probability: **0.546**, which is far above the approval cut‑off of **0.14**.  

**Main risk factors**  
- **Payment delays** – most recent month overdue by 2 months; late payments in 5 of the last 6 months, longest delay 2 months.  
- **Repayment amounts** – consistently pays about 33 % of the average bill each month.  

**Favourable factor**  
- **Credit utilisation** – almost no outstanding balance.  

No borderline flag; the model recommends a straight decline.


### Applicant 11886 | model p(default) = 0.157 | cut-off = 0.14 | decision = DECLINE ###
tools called: ['score_applicant'] | source: openai-compatible | verified: True

Decision: **DECLINE**.  
Default probability: **15.7%**, which exceeds the approval cut‑off of **14%** (borderline = true → recommend manual review).  

**Main risk factors**  
- **Repayment amounts** – pays only about 3% of the average bill each month.  
- **Credit limit** – limited to NT$ 50,000.  

**Favourable factor** (selected)  
- **Payment delays** – no late payments in the last 6 months.


### Applicant 28649 | model p(default) = 0.062 | cut-off = 0.14 | decision = APPROVE ###
tools called: ['score_applicant'] | source: openai-compatible | verified: True

Decision: APPROVE.  
Default probability = 6.2 % versus the approval cut‑off of 14 % (well below the threshold).  
Risk factors: none identified by the model.  
Favourable factor: no late payments in the last 6 months (Payment delays).  

The applicant comfortably meets the policy criteria; no manual review is required.

