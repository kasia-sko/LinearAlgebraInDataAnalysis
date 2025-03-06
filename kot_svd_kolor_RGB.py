from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import moje_funkcje as mf

def main():
    image_path = 'koty/kot_2.jpg'

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
    mf.rysuj_macierz(axes1[0], np.stack((R, z, z), axis=-1), f"kot R", np.min(R), np.max(R))
    mf.rysuj_macierz(axes1[1], np.stack((z, G, z), axis=-1), f"kot G", np.min(G), np.max(G))
    mf.rysuj_macierz(axes1[2], np.stack((z, z, B), axis=-1), f"kot B", np.min(B), np.max(B))
    plt.tight_layout()
    plt.savefig("kot_5.png")
    plt.show()


if __name__ == "__main__":
    main()