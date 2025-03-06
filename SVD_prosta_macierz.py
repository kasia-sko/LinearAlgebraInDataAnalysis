import matplotlib.pyplot as plt
import numpy as np
import moje_funkcje as mf

def rysuj_macierz(ax, macierz, tytul, min, max):
    ax.imshow(macierz, cmap='gray_r', interpolation='nearest', vmin=min, vmax=max)
    ax.set_title(tytul)
    ax.axis('off')

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
    print(U, s, Vt)
    # Tworzenie rysunków na jednej figurze
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    rysuj_macierz(axes[0, 0], A, "Macierz $A$", np.min(A), np.max(A))
    rysuj_macierz(axes[0, 1], U, "Macierz $U$", np.min(U), np.max(U))
    rysuj_macierz(axes[1, 0], S, "Macierz $\\Sigma$", np.min(S), np.max(S))
    rysuj_macierz(axes[1, 1], Vt, "Macierz $V^T$", np.min(Vt), np.max(Vt))

    plt.tight_layout()
    plt.savefig("wyjsciowe_macierze.png")
    plt.show()

    # Zsumowane macierze
    A1 = mf.stworz_Ai(U, S, Vt, 0)
    A2 = mf.stworz_Ai(U, S, Vt, 1)
    A3 = mf.stworz_Ai(U, S, Vt, 2)
    A4 = mf.stworz_Ai(U, S, Vt, 3)
    A5 = mf.stworz_Ai(U, S, Vt, 4)
    A6 = mf.stworz_Ai(U, S, Vt, 5)
    A7 = mf.stworz_Ai(U, S, Vt, 6)
    A8 = mf.stworz_Ai(U, S, Vt, 7)
    A9 = mf.stworz_Ai(U, S, Vt, 8)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

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