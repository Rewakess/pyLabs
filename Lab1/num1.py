import math
import matplotlib.pyplot as plt


def f(x, t):
    return math.sin(t * x)


def left_rectangles(a, b, N, t):
    h = (b - a) / N
    s = 0.0
    for i in range(N):
        x = a + i * h
        s += f(x, t)
    return s * h


def right_rectangles(a, b, N, t):
    h = (b - a) / N
    s = 0.0
    for i in range(1, N + 1):
        x = a + i * h
        s += f(x, t)
    return s * h


def middle_rectangles(a, b, N, t):
    h = (b - a) / N
    s = 0.0
    for i in range(N):
        x0 = a + i * h
        x1 = x0 + h
        xm = (x0 + x1) / 2
        s += f(xm, t)

    return s * h


def trapezoid(a, b, N, t):
    h = (b - a) / N
    s = (f(a, t) + f(b, t)) / 2
    for i in range(1, N):
        x = a + i * h
        s += f(x, t)

    return s * h


def simpson(a, b, N, t):
    if N % 2 == 1:
        N += 1
    h = (b - a) / N
    s = 0.0
    for i in range(N):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        xm = (x0 + x1) / 2

        s += f(x0, t) + 4 * f(xm, t) + f(x1, t)

    return s * h / 6


def integrate(method, a, b, t, eps):
    s = get_order(method)
    N = 4
    I_N = calculate(method, a, b, N, t)
    while True:
        N *= 2
        I_2N = calculate(method, a, b, N, t)
        error = abs(I_2N - I_N) / (2 ** s - 1)
        if error < eps:
            return I_2N
        I_N = I_2N


def get_order(method):
    if method in ["left", "right"]:
        return 1
    elif method in ["middle", "trapezoid"]:
        return 2
    elif method == "simpson":
        return 4
    else:
        raise ValueError("Неизвестный метод")


def calculate(method, a, b, N, t):
    if method == "left":
        return left_rectangles(a, b, N, t)
    elif method == "right":
        return right_rectangles(a, b, N, t)
    elif method == "middle":
        return middle_rectangles(a, b, N, t)
    elif method == "trapezoid":
        return trapezoid(a, b, N, t)
    elif method == "simpson":
        return simpson(a, b, N, t)
    else:
        raise ValueError("Неизвестный метод")


def main():

    print("Методы:")
    print("left")
    print("right")
    print("middle")
    print("trapezoid")
    print("simpson")

    method = input("Введите метод: ")

    a = float(input("Введите a: "))
    b = float(input("Введите b: "))
    alpha = float(input("Введите alpha: "))
    beta = float(input("Введите beta: "))
    eps = float(input("Введите epsilon: "))
    point_count = int(input("Количество точек: "))

    if point_count < 2:
        print("Количество точек должно быть >= 2")
        return

    ts = []
    vals = []
    step = (beta - alpha) / (point_count - 1)

    for i in range(point_count):
        t = alpha + i * step
        I = integrate(method, a, b, t, eps)
        ts.append(t)
        vals.append(I)
        print(f"t = {t:.5f}, I(t) = {I:.10f}")

    plt.figure(figsize=(10, 6))
    plt.plot(ts, vals)
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.show()


if __name__ == "__main__":
    main()
