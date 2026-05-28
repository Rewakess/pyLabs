import math
import matplotlib.pyplot as plt

# параметры задачи
a = -1
b = 1
h = 0.0001


# функции уравнения:
# x'' + p(t)x' + q(t)x = f(t)

def p(t):
    return t


def q(t):
    return 1 + t * t


def f(t):
    return math.sin(math.pi * t)


# базисная функция phi_i
def phi(i, t):
    return math.sin(i * math.pi * (t + 1) / 2)


# первая производная
def dphi(i, t):
    return (phi(i, t + h) - phi(i, t - h)) / (2 * h)


# вторая производная
def ddphi(i, t):
    return (phi(i, t - h) - 2 * phi(i, t) + phi(i, t + h)) / (h * h)


# оператор L(phi)
def L(i, t):
    return ddphi(i, t) + p(t) * dphi(i, t) + q(t) * phi(i, t)


# интеграл методом Симпсона
def simpson(func, left, right, N):
    if N % 2 == 1:
        N += 1

    h = (right - left) / N
    s = 0.0

    for i in range(N):
        x0 = left + i * h
        x1 = left + (i + 1) * h
        xm = (x0 + x1) / 2

        s += func(x0) + 4 * func(xm) + func(x1)

    return s * h / 6


# скалярное произведение <f, g>
def scalar_product(i, j):
    def integrand(t):
        return L(j, t) * phi(i, t)

    return simpson(integrand, a, b, 200)


# правая часть
def right_part(i):
    def integrand(t):
        return f(t) * phi(i, t)

    return simpson(integrand, a, b, 200)


# решение системы методом Гаусса
def gauss(A, B):
    n = len(B)

    for k in range(n):
        # главный элемент
        max_row = k

        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i

        A[k], A[max_row] = A[max_row], A[k]
        B[k], B[max_row] = B[max_row], B[k]

        # нормализация
        pivot = A[k][k]

        for j in range(k, n):
            A[k][j] /= pivot

        B[k] /= pivot

        # исключение
        for i in range(k + 1, n):
            factor = A[i][k]

            for j in range(k, n):
                A[i][j] -= factor * A[k][j]

            B[i] -= factor * B[k]

    # обратный ход
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        x[i] = B[i]

        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]

    return x


def main():
    # ввод количества функций
    N = int(input("Введите N: "))

    # матрица системы
    A = []
    B = []

    for i in range(1, N + 1):
        row = []

        for j in range(1, N + 1):
            row.append(scalar_product(i, j))

        A.append(row)
        B.append(right_part(i))

    # коэффициенты
    C = gauss(A, B)

    # приближенное решение
    def xN(t):
        s = 0

        for i in range(N):
            s += C[i] * phi(i + 1, t)

        return s

    # точки для графика
    X = []
    Y = []

    points = 300

    for k in range(points + 1):
        t = a + (b - a) * k / points

        X.append(t)
        Y.append(xN(t))

    # график
    plt.plot(X, Y)
    plt.xlabel("t")
    plt.ylabel("xN(t)")
    plt.title("Метод Галеркина")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
