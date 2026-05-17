def f(x, mu):
    return x ** 3 - mu * x + 1


def derivative(x, mu, h=1e-6):
    return (f(x + h, mu) - f(x - h, mu)) / (2 * h)


def bisection(a, b, mu, eps=1e-6, max_iter=100):
    if f(a, mu) * f(b, mu) > 0:
        return None

    for _ in range(max_iter):
        c = (a + b) / 2

        if abs(f(c, mu)) < eps or abs(b - a) < eps:
            return c

        if f(a, mu) * f(c, mu) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2


def newton(x0, mu, eps=1e-6, max_iter=100):
    x = x0

    for _ in range(max_iter):

        d = derivative(x, mu)

        if abs(d) < 1e-12:
            return None

        x_new = x - f(x, mu) / d

        if abs(x_new - x) < eps:
            return x_new

        x = x_new

    return None


def find_roots(a, b, mu, method="bisection", step=0.5, eps=1e-6):
    roots = []

    x = a

    while x < b:

        x_next = min(x + step, b)

        if f(x, mu) * f(x_next, mu) <= 0:

            if method == "bisection":
                root = bisection(x, x_next, mu, eps)

            elif method == "newton":
                x0 = (x + x_next) / 2
                root = newton(x0, mu, eps)

            else:
                raise ValueError("Неизвестный метод")

            if root is not None:

                duplicate = False

                for r in roots:
                    if abs(r - root) < eps:
                        duplicate = True
                        break

                if not duplicate:
                    roots.append(root)

        x = x_next

    return roots


def main():
    print("Решение нелинейного уравнения f(x, mu) = 0")

    # Ввод параметров
    a = float(input("Введите a: "))
    b = float(input("Введите b: "))

    alpha = float(input("Введите alpha: "))
    beta = float(input("Введите beta: "))

    mu_step = float(input("Шаг по mu: "))

    eps = float(input("Точность eps: "))

    method = input("Метод (bisection/newton): ").strip()

    print("\nНайденные решения:\n")

    mu = alpha

    while mu <= beta + 1e-12:

        roots = find_roots(a, b, mu, method=method, eps=eps)

        if len(roots) == 0:
            print(f"mu = {mu:.4f} -> решений нет")
        else:
            for r in roots:
                print(f"mu = {mu:.4f}, x = {r:.8f}")

        mu += mu_step


if __name__ == "__main__":
    main()
