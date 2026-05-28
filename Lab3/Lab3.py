import math
import matplotlib.pyplot as plt


# целевая функция
def f(x, y):
    return (x - 1) ** 2 + 5 * (y - 2) ** 2 + 0.5 * math.sin(3 * x) * math.cos(2 * y)


# частная производная по x (центральные разности)
def dfdx(x, y, h=1e-6):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


# частная производная по y (центральные разности)
def dfdy(x, y, h=1e-6):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


# норма вектора (gx, gy)
def norm(gx, gy):
    return math.sqrt(gx * gx + gy * gy)


# метод золотого сечения — минимум функции phi на [a, b]
def golden_section(phi, a, b, eps=1e-10):
    PHI = (1 + math.sqrt(5)) / 2

    while abs(b - a) > eps:
        alpha = b - (b - a) / PHI
        beta  = a + (b - a) / PHI

        if phi(alpha) <= phi(beta):
            b = beta
        else:
            a = alpha

    return (a + b) / 2


# покоординатный спуск
def coordinate_descent(x0, y0, eps, a, b, c, d):
    x, y = x0, y0
    path = [(x, y)]

    for _ in range(5000):
        x_old, y_old = x, y

        def f_by_x(t):
            return f(t, y)

        def f_by_y(t):
            return f(x, t)

        x = golden_section(f_by_x, a, b)
        y = golden_section(f_by_y, c, d)

        path.append((x, y))

        if norm(x - x_old, y - y_old) < eps:
            break

    return path


# наискорейший градиентный спуск
def steepest_descent(x0, y0, eps, a, b, c, d):
    x, y = x0, y0
    path = [(x, y)]

    for _ in range(10000):
        gx = dfdx(x, y)
        gy = dfdy(x, y)

        if norm(gx, gy) < 1e-12:
            break

        alpha_max = max(b - a, d - c)

        def f_along_gradient(alpha):
            return f(x - alpha * gx, y - alpha * gy)

        alpha = golden_section(f_along_gradient, 0, alpha_max)

        x_new = max(a, min(b, x - alpha * gx))
        y_new = max(c, min(d, y - alpha * gy))

        path.append((x_new, y_new))

        if norm(x_new - x, y_new - y) < eps:
            x, y = x_new, y_new
            break

        x, y = x_new, y_new

    return path


# построение графика с траекторией
def plot_result(path, a, b, c, d):
    n = 300
    xs = [a + (b - a) * i / (n - 1) for i in range(n)]
    ys = [c + (d - c) * i / (n - 1) for i in range(n)]

    X = []
    Y = []
    Z = []

    for y in ys:
        row_x = []
        row_y = []
        row_z = []
        for x in xs:
            row_x.append(x)
            row_y.append(y)
            row_z.append(f(x, y))
        X.append(row_x)
        Y.append(row_y)
        Z.append(row_z)

    px = [pt[0] for pt in path]
    py = [pt[1] for pt in path]

    plt.contourf(X, Y, Z, levels=20)
    plt.contour(X, Y, Z, levels=20, colors="black", linewidths=0.5)
    plt.plot(px, py, "w-o", markersize=3, linewidth=1)
    plt.scatter(px[0],  py[0],  color="white", edgecolors="black", zorder=5)
    plt.scatter(px[-1], py[-1], color="red", marker="*", s=150, zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# главная функция
def main():
    a = float(input("Введите a: "))
    b = float(input("Введите b: "))
    c = float(input("Введите c: "))
    d = float(input("Введите d: "))

    while True:
        x0 = float(input(f"\nВведите x0 (в [{a}, {b}]): "))
        y0 = float(input(f"Введите y0 (в [{c}, {d}]): "))

        if a <= x0 <= b and c <= y0 <= d:
            break

        print("Точка вне области. Попробуйте снова.")

    eps = float(input("Введите epsilon: "))

    print("\nВыберите метод:")
    print("1 — Покоординатный спуск")
    print("2 — Наискорейший градиентный спуск")
    choice = input().strip()

    if choice == "1":
        path = coordinate_descent(x0, y0, eps, a, b, c, d)
    else:
        path = steepest_descent(x0, y0, eps, a, b, c, d)

    xm, ym = path[-1]
    print(f"\nИтераций:    {len(path) - 1}")
    print(f"Минимум:     x = {xm:.8f},  y = {ym:.8f}")
    print(f"f(x*, y*) =  {f(xm, ym):.10f}")

    plot_result(path, a, b, c, d)


if __name__ == "__main__":
    main()