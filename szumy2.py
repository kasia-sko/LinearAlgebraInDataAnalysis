import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import random

def rysuj_macierz(ax, macierz, tytul, vmin, vmax):
    ax.imshow(macierz, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(tytul)
    ax.axis('off')

def odszumianie(Mnoise, int):
    sigma = int
    U, S, Vt = np.linalg.svd(Mnoise, full_matrices=False)
    R = Mnoise.shape[0]
    cutoff = (4 / np.sqrt(3)) * np.sqrt(R) * sigma
    r = np.max(np.where(S > cutoff))

    # Odtworzenie obrazu za pomocą ograniczonej liczby wartości osobliwych
    Mclean = U[:, :r+1] @ np.diag(S[:r+1]) @ Vt[:r+1, :]
    return Mclean

def dodaj_szum_solpieprz(M):
    Mnoise = np.copy(M)
    salt = np.random.rand(*M.shape) < 0.02
    pepper = np.random.rand(*M.shape) < 0.02
    Mnoise[salt] = 255
    Mnoise[pepper] = 0
    return Mnoise


def main():
    # Ścieżka do twojego obrazu
    image_path = 'stefan2.jpg'

    # Załaduj obraz
    image = Image.open(image_path)

    # Konwertuj obraz na odcienie szarości
    image_gray = image.convert('L')

    # Przekonwertuj obraz na macierz numpy
    M = np.array(image_gray)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    rysuj_macierz(ax[0,0], M, "Obraz wyjściowy", np.min(M), np.max(M))

    Mnoise = dodaj_szum_solpieprz(M)

    rysuj_macierz(ax[0,1], Mnoise, "Obraz zaszumiony", np.min(Mnoise), np.max(Mnoise))

    Mclean1 = odszumianie(Mnoise, 15)
    rysuj_macierz(ax[1,0], Mclean1, "Obraz odszumiony 1", np.min(Mclean1), np.max(Mclean1))
    Mclean2 = odszumianie(Mnoise, 20)
    rysuj_macierz(ax[1,1], Mclean2, "Obraz odszumiony 2", np.min(Mclean2), np.max(Mclean2))

    plt.show()

if __name__ == "__main__":
    main()