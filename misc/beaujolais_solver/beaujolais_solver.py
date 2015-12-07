from collections import defaultdict
import operator
import z3

OP = {
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    '>=': operator.ge,
    '>': operator.gt,
}


def parse(solver, catch_copies):
    for catch_copy in filter(bool, catch_copies.splitlines()):
        (lhs, op, rhs) = catch_copy.split()
        solver.add(OP[op](z3.Real(lhs), z3.Real(rhs)))
    return solver


def as_dict(solution):
    solution = solution.replace('[', '{')
    solution = solution.replace(']', '}')
    solution = solution.replace(' = ', '": ')
    solution = solution.replace('x', '"x')
    return eval(solution)


def ranking(d):
    indices = defaultdict(list)
    for (idx, val) in d.items():
        indices[val].append(idx)
    sorted_indices = sorted(indices.items(), reverse=True)
    for (rank, (_, indices)) in enumerate(sorted_indices, start=1):
        print(rank, indices)


def solve(catch_copies):
    solver = parse(z3.Solver(), catch_copies)
    print('solution:')
    print(solver.check())
    print(solver.model())
    print('ranking:')
    d = as_dict(solver.model().__str__())
    ranking(d)


if __name__ == '__main__':
    import sys
    catch_copy_file = sys.argv[1]
    catch_copies = open(catch_copy_file).read()
    solve(catch_copies)
