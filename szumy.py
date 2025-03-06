import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def dodaj_szum_bialy(M, mean=0, std=35):
    noise = np.random.normal(mean, std, M.shape)
    Mnoise = M.astype(np.float64) + noise
    Mnoise = np.clip(Mnoise, 0, 255)
    Mnoise = Mnoise.astype(np.uint8)
    return Mnoise

def main():

    # Szum biały
    image_path = 'koty/kot_3.jpg'
    image = Image.open(image_path)
    image_gray = image.convert('L')
    M = np.array(image_gray)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    rysuj_macierz(ax[0,0], M, "Obraz wyjściowy", np.min(M), np.max(M))

    Mnoise = dodaj_szum_bialy(M)

    rysuj_macierz(ax[0,1], Mnoise, "Obraz zaszumiony", np.min(Mnoise), np.max(Mnoise))

    Mclean1 = odszumianie(Mnoise, 20)
    rysuj_macierz(ax[1,0], Mclean1, "Obraz odszumiony 1", np.min(Mclean1), np.max(Mclean1))
    Mclean2 = odszumianie(Mnoise, 15)
    rysuj_macierz(ax[1,1], Mclean2, "Obraz odszumiony 2", np.min(Mclean2), np.max(Mclean2))
    plt.savefig("kot_w_4.png")
    plt.show()

    # Szum sól pieprz
    image_path_2 = 'koty/kot_4.jpg'
    image_2 = Image.open(image_path_2)
    image_gray_2 = image_2.convert('L')
    M_2 = np.array(image_gray_2)
    fig_2, ax_2 = plt.subplots(2, 2, figsize=(12, 8))

    rysuj_macierz(ax_2[0, 0], M_2, "Obraz wyjściowy", np.min(M_2), np.max(M_2))

    Mnoise_2 = dodaj_szum_solpieprz(M_2)

    rysuj_macierz(ax_2[0, 1], Mnoise_2, "Obraz zaszumiony", np.min(Mnoise_2), np.max(Mnoise_2))

    Mclean1_2 = odszumianie(Mnoise_2, 15)
    rysuj_macierz(ax_2[1, 0], Mclean1_2, "Obraz odszumiony 1", np.min(Mclean1_2), np.max(Mclean1_2))
    Mclean2_2 = odszumianie(Mnoise_2, 20)
    rysuj_macierz(ax_2[1, 1], Mclean2_2, "Obraz odszumiony 2", np.min(Mclean2_2), np.max(Mclean2_2))
    plt.savefig("kot_w_5.png")
    plt.show()

if __name__ == "__main__":
    main()