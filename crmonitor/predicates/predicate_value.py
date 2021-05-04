class PredicateValue:
    def __init__(self, predicate_str, vehicle_ids, time_step, value=None):
        self.predicate_str = predicate_str
        self.vehicle_ids = tuple(vehicle_ids)
        self.time_step = time_step
        self.value = value

    def __hash__(self):
        return hash((self.predicate_str, self.vehicle_ids, self.time_step))

    def __eq__(self, o) -> bool:
        return (
            self.predicate_str == o.predicate_str
            and self.vehicle_ids == o.vehicle_ids
            and self.time_step == o.time_step
        )


class PredicateValueCollection:
    def __init__(self, l=None):
        if l is None:
            l = set()
        self._predicate_values = set(l)

    def __iter__(self):
        return self._predicate_values.__iter__()

    def __contains__(self, item):
        return item in self._predicate_values

    def __getitem__(self, item):
        intersect = self._predicate_values.intersection([item])
        if len(intersect) == 0:
            raise KeyError
        else:
            return intersect.pop()

    def clear(self):
        self._predicate_values.clear()

    def get_single_value(self):
        assert (
            len(self._predicate_values) == 1
        ), f"PredicateValueCollection contains {len(self._predicate_values)} != 1 value!"
        return list(self._predicate_values)[0]

    def by_name(self, name):
        pred = [x for x in self._predicate_values if x.predicate_str == name]
        return PredicateValueCollection(pred)

    def by_time_step(self, time_step):
        pred = [x for x in self._predicate_values if x.time_step == time_step]
        return PredicateValueCollection(pred)

    def by_ids(self, ids):
        pred = [x for x in self._predicate_values if x.vehicle_ids == ids]
        return PredicateValueCollection(pred)

    def get_time_steps(self):
        tsteps = set()
        for p in self._predicate_values:
            tsteps.add(p.time_step)
        return tuple(sorted(tsteps))

    def append(self, element):
        self._predicate_values.add(element)

    def extend(self, other):
        self._predicate_values.update(other._predicate_values)
