"""Count auto-validated vs manually-reviewed split for the new cohort."""

from collections import Counter

from build.markup import loader

eps = loader.load(cohort='mixed')
print(f'total: {len(eps)}')
by_src = Counter(e.cohort_source for e in eps)
for k, v in by_src.items():
    print(f'  {k}: {v}')

manual = [e for e in eps if e.cohort_source == 'manual']
print(f'manual cohort: {len(manual)}')
