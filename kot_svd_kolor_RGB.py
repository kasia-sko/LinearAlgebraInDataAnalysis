from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def rysuj_macierz(ax, macierz, tytul, vmin, vmax):
    ax.imshow(macierz, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(tytul)
    ax.axis('off')

def stworz_Ai(U, S, Vt, i):
    # jeśli macierz nie je
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
    image_path = 'prazek-min.jpg'

    # Załadowanie obrazu
    image = Image.open(image_path)
    image = image.convert('RGB')
    kot = np.array(image)

    # Wyodrębnienie poszczególnych kanałów
    R = kot[:, :, 0]
    G = kot[:, :, 1]
    B = kot[:, :, 2]

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 8))
    z = np.zeros_like(R)
    rysuj_macierz(axes1[0], np.stack((R, z, z), axis=-1), f"kot R", np.min(R), np.max(R))
    rysuj_macierz(axes1[1], np.stack((z, G, z), axis=-1), f"kot G", np.min(G), np.max(G))
    rysuj_macierz(axes1[2], np.stack((z, z, B), axis=-1), f"kot B", np.min(B), np.max(B))
    plt.tight_layout()
    plt.savefig("kotek_RGB.png")
    plt.show()


if __name__ == "__main__":
    main()