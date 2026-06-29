# Chemical Phase Interface Scoring

## The problem

Given one photograph of a chemical vessel, I have to output an `interface_burden`
index from 0 to 100 saying how difficult the visible phase state is. A vessel can
show clear liquid, turbid liquid, settled solids, residue on the wall and empty
headspace all at once, and a higher score means more heterogeneity, more solid
burden, more vertical phase transitions and more suspended material.

The test split holds out entire experiment groups, so anything I learn about
image style, color palette or repeated video frames is worthless on the day.

## What I did

The first thing I did was take the metric apart, and it changes the whole
approach. Although I emit one continuous number, the grader is roughly 80 percent
an ordinal four zone classifier, with zone boundaries at 12, 35 and 48 and a flat
penalty for landing in the wrong zone. Mean absolute error contributes only about
10 percent. So there is very little value in nudging a prediction from 30 to 32,
and enormous value in not falling off a zone edge.

That means I optimize for zone placement and treat the continuous value as a way
of positioning myself safely inside a zone rather than as the target itself.
Design decisions are grounded in direct measurement of the public data under
`eda/` and a literature pass recorded in `research_findings.md`.

## Layout

`solution.py` and `solution_core.py` are the entry points, `eda/` holds the data
analysis, `Approach.md` the write up. Datasets are not committed.
