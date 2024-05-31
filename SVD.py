import matplotlib.pyplot as plt
import numpy as np

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

    # Tworzenie rysunków na jednej figurze
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    rysuj_macierz(axes[0, 0], A, "Macierz $A$", np.min(A), np.max(A))
    rysuj_macierz(axes[0, 1], U, "Macierz $U$", np.min(U), np.max(U))
    rysuj_macierz(axes[1, 0], S, "Macierz $\\Sigma$", np.min(S), np.max(S))
    rysuj_macierz(axes[1, 1], Vt, "Macierz $V^T$", np.min(Vt), np.max(Vt))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()