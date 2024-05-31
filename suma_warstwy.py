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
    min = 100
    max = -100
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
    lista_macierzy = [A1, A2, A3, A4, A5, A6, A7, A8, A9]
    max = stworz_skale(lista_macierzy)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    #Rysowanie każdej macierzy na osobnym subplotcie
    rysuj_macierz(axes[0,0], A1 + A2, "Macierz A1 + A2", np.min(A1 + A2), np.max(A1 + A2))
    rysuj_macierz(axes[0,1], A1 + A2 + A3, "Macierz A1 + A2 + A3", np.min(A1 + A2 + A3), np.max(A1 + A2 + A3))
    rysuj_macierz(axes[1,0], A1 + A2 + A3 + A4, "Macierz A1 + A2 + A3 + A4", np.min(A1 + A2 + A3 + A4),
                  np.max(A1 + A2 + A3 + A4))
    rysuj_macierz(axes[1,1], A1 + A2 + A3 + A4 + A5, "Macierz A1 + A2 + A3 + A4 + A5", np.min(A1 + A2 + A3 + A4 + A5),
                  np.max(A1 + A2 + A3 + A4 + A5))
    rysuj_macierz(axes[2,0], A1 + A2 + A3 + A4 + A5 + A6, "Macierz A1 + A2 + A3 + A4 + A5 + A6", np.min(A1 + A2 + A3 + A4 + A5 + A6),
                  np.max(A1 + A2 + A3 + A4 + A5 + A6))
    rysuj_macierz(axes[2,1], A1 + A2 + A3 + A4 + A5 + A6 + A7, "Macierz A1 + A2 + A3 + A4 + A5 + A6 + A7",
                  np.min(A1 + A2 + A3 + A4 + A5 + A6 + A7),
                  np.max(A1 + A2 + A3 + A4 + A5 + A6 + A7))

    plt.tight_layout()
    plt.savefig("macierze_sumy.png")
    plt.show()


if __name__ == "__main__":
    main()