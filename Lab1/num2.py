import math
import matplotlib.pyplot as plt


def f(x, t):
    return math.log(1 + t * x) / math.sqrt(x)


def simpson(a, b, N, t):
    h = (b - a) / N
    s = 0.0

    for i in range(N):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        xm = (x0 + x1) / 2

        s += f(x0, t) + 4 * f(xm, t) + f(x1, t)

    return s * h / 6


def integrate(a, b, t, eps):
    N = 4
    I_N = simpson(a, b, N, t)

    while True:
        N *= 2
        I_2N = simpson(a, b, N, t)

        if abs(I_2N - I_N) / 15 < eps:
            return I_2N

        I_N = I_2N


def improper_integral(t, eps):
    delta = 1e-2

    while True:
        main_part = integrate(delta, 1, t, eps)
        tail = integrate(delta / 2, delta, t, eps)

        if abs(tail) < eps:
            return main_part

        delta /= 2


def main():
    alpha = float(input("Введите альфа: "))
    beta = float(input("Введите бета: "))
    eps = float(input("Введите эпсилон: "))
    point_count = int(input("Количество точек: "))

    ts = []
    vals = []

    step = (beta - alpha) / (point_count - 1)

    for i in range(point_count):
        t = alpha + i * step
        I = improper_integral(t, eps)

        ts.append(t)
        vals.append(I)

        print(f"t={t:.5f}, I(t)={I:.8f}")

    plt.plot(ts, vals)
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.show()


if __name__ == "__main__":
    main()