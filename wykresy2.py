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

    #Przeprowadzenie rozkładu SVD
    U, s, Vt = np.linalg.svd(A)
    S = np.diag(s)

    plt.figure(1)
    plt.plot(s, marker='o', linestyle='-',
             color='b')  # Użyj marker='o', linestyle='-', aby uzyskać linię bez wypełnienia
    #plt.yscale('log')  # Ustaw skalę osi Y na logarytmiczną
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
    plt.ylim(0, 110)  # Ustawienie zakresu osi y
    plt.show()


if __name__ == "__main__":
    main()