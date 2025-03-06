import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

    image_path = 'koty/kot_1.jpeg'
    image = Image.open(image_path)
    image_gray = image.convert('L')
    image_matrix = np.array(image_gray)

    U, s, Vt = np.linalg.svd(image_matrix)

    plt.figure(1)
    plt.plot(s, marker='o', linestyle='-',
             color='b')
    plt.yscale('log')
    plt.title('Wartości osobliwe')
    plt.xlabel('Indeksy wartości osobliwych')
    plt.ylabel('Wartości')
    plt.show()

    cumulative_sum = np.cumsum(s) / np.sum(s)

    # Skalowanie do zakresu 0-100
    cumulative_sum_scaled = cumulative_sum * 100

    # Wykres skumulowanych wartości osobliwych
    plt.figure(2)
    plt.plot(range(1, len(cumulative_sum_scaled) + 1), cumulative_sum_scaled, marker='o', linestyle='-', color='r')
    plt.title('Skumulowane wartości osobliwe')
    plt.xlabel('Liczba dodanych wartości osobliwych')
    plt.ylabel('Skumulowana suma (%)')
    plt.ylim(0, 110)
    plt.show()

    # Wykresy 2 i 3
    # Przeprowadzenie rozkładu SVD
    U, s, Vt = np.linalg.svd(A)
    plt.figure(1)
    plt.plot(s, marker='o', linestyle='-',
             color='b')
    plt.title('Wartości osobliwe')
    plt.xlabel('Indeksy wartości osobliwych')
    plt.ylabel('Wartości')
    plt.show()
    plt.figure(2)
    plt.plot(range(1, len(cumulative_sum_scaled) + 1), cumulative_sum_scaled, marker='o', linestyle='-', color='r')
    plt.title('Skumulowane wartości osobliwe')
    plt.xlabel('Liczba dodanych wartości osobliwych')
    plt.ylabel('Skumulowana suma (%)')
    plt.ylim(0, 110)
    plt.show()


if __name__ == "__main__":
    main()