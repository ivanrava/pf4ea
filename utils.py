class Path(list):
    def __getitem__(self, t):
        try:
            return super().__getitem__(t)
        except IndexError:
            return super().__getitem__(-1)

    def __add__(self, other):
        return Path([x for x in self] + [x for x in other])

    def waits(self):
        count = 0
        for t in range(len(self) - 1):
            if self[t] == self[t + 1]:
                count += 1
        return count


def extract_min(structure: set, function):
    min_el = next(iter(structure))
    min_score = function(min_el)
    for el in structure:
        score = function(el)
        if score < min_score:
            min_score = score
            min_el = el
    return min_el
