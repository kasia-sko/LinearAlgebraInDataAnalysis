import matplotlib.pyplot as plt
import numpy as np

def rysuj_macierz(ax, macierz, tytul, min, max):
    ax.imshow(macierz, cmap='gray_r', interpolation='nearest', vmin=min, vmax=max)
    ax.set_title(tytul)
    ax.axis('off')

def stworz_Ai(U, S, Vt, i):
    if i >= len(U):
        Ai = np.zeros_like(U)
        return Ai
    ui = U[:, i].reshape(-1, 1)
    vti = Vt[i, :].reshape(1, -1)
    s = S[i][i]
    Ai = ui * s @ vti
    return Ai


def stworz_skale(lista):
    min = np.inf
    max = -np.inf
    for i in range(len(lista)):
        if np.min(lista[i]) < min:
            min = np.min(lista[i])
    for i in range(len(lista)):
        if np.max(lista[i]) > max:
            max = np.max(lista[i])
    if abs(min) > abs(max):
        return abs(min)
    else:
        return abs(max)

def main():
    # Przykładowa macierz A
    A = np.array([
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    ])

    # Przeprowadzenie rozkładu SVD
    U, s, Vt = np.linalg.svd(A)

    # Stworzenie macierzy diagonalnej S
    S = np.diag(s)

    A1 = stworz_Ai(U, S, Vt, 0)
    A2 = stworz_Ai(U, S, Vt, 1)
    A3 = stworz_Ai(U, S, Vt, 2)
    A4 = stworz_Ai(U, S, Vt, 3)
    A5 = stworz_Ai(U, S, Vt, 4)
    A6 = stworz_Ai(U, S, Vt, 5)
    A7 = stworz_Ai(U, S, Vt, 6)
    A8 = stworz_Ai(U, S, Vt, 7)
    A9 = stworz_Ai(U, S, Vt, 8)
    A10 = stworz_Ai(U, S, Vt, 9)

    lista_macierzy = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]
    max = stworz_skale(lista_macierzy)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for i in range(6):
        rysuj_macierz(axes[i // 2, i % 2], np.abs(lista_macierzy[i]), f"Macierz $A_{i + 1}$", 0, max)

    plt.tight_layout()
    plt.savefig("macierze_A1_A6.png")
    plt.show()

if __name__ == "__main__":
    main()