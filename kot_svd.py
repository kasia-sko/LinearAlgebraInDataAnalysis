from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def rysuj_macierz(ax, macierz, tytul, vmin, vmax):
    ax.imshow(macierz, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
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

def sumuj_do_Ai(kot, i):
    # Przeprowadzenie rozkładu SVD
    U, s, Vt = np.linalg.svd(kot)
    # Stworzenie macierzy diagonalnej S
    S = np.diag(s)
    K = np.zeros_like(kot, dtype=np.float64)  # Rzutowanie na float64
    for j in range(i):
        Ai = stworz_Ai(U, S, Vt, j)
        Ai= np.pad(Ai, ((0, 0), (0, kot.shape[1] - Ai.shape[1])), mode='constant')
        K += Ai.astype(np.float64)  # Rzutowanie Ai na float64 przed dodaniem
        K = np.clip(K, 0, 255)
    return K.astype(np.uint8)  # Rzutowanie na uint8 przed zwróceniem


def main():
    image_path = 'kot.jpeg'

    # Załadowanie obrazu
    image = Image.open(image_path)

    # Konwertowanie obrazu na odcienie szarości
    image_gray = image.convert('L')
    kot = np.array(image_gray)

    lista_indeksow = [1, 2, 4, 5, 10, 20, 50, 200, 700, 1000]
    fig, axes = plt.subplots(5, 2, figsize=(12, 8))

    for i in range(len(lista_indeksow)):
        row = i // 2
        col = i % 2
        # Rysowanie każdej macierzy na osobnym subplotcie
        kot_i = sumuj_do_Ai(kot, lista_indeksow[i])
        rysuj_macierz(axes[row, col], kot_i, f"kot {lista_indeksow[i]}", np.min(kot_i), np.max(kot_i))

    plt.tight_layout()
    plt.savefig("kotek_ady.png")
    plt.show()



if __name__ == "__main__":
    main()
