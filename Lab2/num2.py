def jacobian(F, x, h=1e-6):
    """
    Численное вычисление матрицы Якоби.

    F  - список функций
    x  - текущая точка
    h  - шаг для конечной разности
    """

    n = len(x)
    J = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            x1 = x[:]
            x2 = x[:]

            x1[j] += h
            x2[j] -= h

            J[i][j] = (F[i](x1) - F[i](x2)) / (2 * h)

    return J


def lu_decomposition(A):
    """
    LU-разложение матрицы A.
    A = L * U
    """

    n = len(A)

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):

        # Вычисляем U
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]

            U[i][j] = A[i][j] - s

        # Вычисляем L
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]

            if abs(U[i][i]) < 1e-12:
                raise ValueError("Матрица вырождена")

            L[j][i] = (A[j][i] - s) / U[i][i]

    return L, U


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i][j] * y[j]

        y[i] = b[i] - s

    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]

        if abs(U[i][i]) < 1e-12:
            raise ValueError("Матрица вырождена")

        x[i] = (y[i] - s) / U[i][i]

    return x


def solve_lu(A, b):
    L, U = lu_decomposition(A)

    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x


def newton_system(F, x0, eps=1e-8, max_iter=100):
    """
    F         - список функций
    x0        - начальное приближение
    eps       - точность
    max_iter  - максимум итераций
    """

    x = x0[:]
    n = len(x)

    for iteration in range(max_iter):

        Fx = [F[i](x) for i in range(n)]

        J = jacobian(F, x)

        b = [-v for v in Fx]

        dx = solve_lu(J, b)

        x_new = [x[i] + dx[i] for i in range(n)]

        norm = max(abs(dx[i]) for i in range(n))

        print(f"Итерация {iteration + 1}: x = {x_new}")

        if norm < eps:
            return x_new

        x = x_new

    raise ValueError("Метод не сошелся")


def is_new_solution(solutions, x, eps=1e-6):
    """
    Проверка, найдено ли решение ранее.
    """

    for sol in solutions:

        equal = True

        for i in range(len(x)):
            if abs(sol[i] - x[i]) > eps:
                equal = False
                break

        if equal:
            return False

    return True


def f1(v):
    x, y = v
    return x * x + y * y - 4


def f2(v):
    x, y = v
    return x - y - 1


def main():
    F = [f1, f2]

    solutions = []

    # Набор начальных приближений
    starts = [
        [2.0, 1.0],
        [-2.0, -1.0],
        [1.0, 2.0],
        [-1.0, -2.0]
    ]

    for x0 in starts:

        try:
            solution = newton_system(F, x0)

            if is_new_solution(solutions, solution):
                solutions.append(solution)

        except:
            pass

    print("\nНайденные решения:")

    for k, sol in enumerate(solutions):

        print(f"\nРешение {k + 1}:")

        for i, value in enumerate(sol):
            print(f"x{i + 1} = {value}")


if __name__ == "__main__":
    main()
