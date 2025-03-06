from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import moje_funkcje as mf

def main():
    image_path = 'koty/kot_2.jpg'
    image = Image.open(image_path)
    image = image.convert('RGB')
    kot = np.array(image)

    # Wyodrębnienie poszczególne kanałów RBG
    R = kot[:, :, 0]
    G = kot[:, :, 1]
    B = kot[:, :, 2]

    lista_indeksow = [1, 2, 4, 5, 10, 20, 50, 200, 700]
    fig, axes = plt.subplots(5, 2, figsize=(12, 8))

    for i in range(len(lista_indeksow)):
        row = i // 2
        col = i % 2

        kot_i_R = mf.sumuj_do_Ai(R, lista_indeksow[i])
        kot_i_G = mf.sumuj_do_Ai(G, lista_indeksow[i])
        kot_i_B = mf.sumuj_do_Ai(B, lista_indeksow[i])
        kot_kolorowy = np.stack((kot_i_R, kot_i_G, kot_i_B), axis=-1)
        mf.rysuj_macierz(axes[row, col], kot_kolorowy, f"warstwy: {lista_indeksow[i]}", np.min(kot_kolorowy),
                         np.max(kot_kolorowy))

    mf.rysuj_macierz(axes[len(lista_indeksow) // 2, len(lista_indeksow) % 2], kot, "pierwotne zdjęcie:", np.min(kot),
                     np.max(kot))

    plt.tight_layout()
    plt.savefig("kot_w_2.png")
    plt.show()

if __name__ == "__main__":
    main()