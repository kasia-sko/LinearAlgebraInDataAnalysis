from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import moje_funkcje as mf

def main():
    # Kot svd czarnobiały
    image_path = 'koty/kot_1.jpeg'
    image = Image.open(image_path)
    image_gray = image.convert('L')
    kot = np.array(image_gray)

    lista_indeksow = [1, 2, 4, 5, 10, 20, 50, 200, 700]
    fig, axes = plt.subplots(5, 2, figsize=(12, 8))

    for i in range(len(lista_indeksow)):
        row = i // 2
        col = i % 2
        # Rysowanie każdej macierzy na osobnym subplotcie
        kot_i = mf.sumuj_do_Ai(kot, lista_indeksow[i])
        mf.rysuj_macierz(axes[row, col], kot_i, f"warstwy: {lista_indeksow[i]}", np.min(kot_i), np.max(kot_i))
    mf.rysuj_macierz(axes[len(lista_indeksow) // 2, len(lista_indeksow) % 2], kot, "pierwotne zdjęcie:", np.min(kot), np.max(kot))

    # Sprawdzenie ilości zajmowanego miejsca w pamięci
    wymiary_poczatkowe = np.shape(kot)
    pamiec_poczatkowa = wymiary_poczatkowe[0] * wymiary_poczatkowe[0] + wymiary_poczatkowe[1] * wymiary_poczatkowe[1] + wymiary_poczatkowe[0] * wymiary_poczatkowe[1]
    pamiec_aproksymacja = (wymiary_poczatkowe[0] * 50 + 50 * 50 + wymiary_poczatkowe[1] * 50)
    print("Wymiary początkowe: ", wymiary_poczatkowe)
    print("Stosunek zajmowanej pamięci : ", pamiec_aproksymacja / pamiec_poczatkowa)

    plt.tight_layout()
    plt.savefig("kot_w_1.png")
    plt.show()


if __name__ == "__main__":
    main()
