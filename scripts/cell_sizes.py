"""Print per-(model, object) episode counts and ranges for the headline cohort."""

from collections import defaultdict

from build.markup import loader

TRAINED = ('Towels', 'Wooden spoons', 'Scissors', 'Batteries')


def main():
    eps = loader.load(cohort='mixed')
    by_mo = defaultdict(int)
    for e in eps:
        if e.eval_object in TRAINED:
            by_mo[(e.model_display, e.eval_object)] += 1

    by_model = defaultdict(list)
    for (m, _o), c in by_mo.items():
        by_model[m].append(c)

    for m, sizes in sorted(by_model.items()):
        print(f'{m}: counts={sorted(sizes)} range={min(sizes)}-{max(sizes)} sum={sum(sizes)}')


if __name__ == '__main__':
    main()
