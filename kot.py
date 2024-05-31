from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def rysuj_macierz(ax, macierz, tytul, vmin, vmax):
    ax.imshow(macierz, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(tytul)
    ax.axis('off')

def main():
    # Ścieżka do twojego obrazu
    image_path = 'kot.jpeg'

    # Załaduj obraz
    image = Image.open(image_path)

    # Konwertuj obraz na odcienie szarości
    image_gray = image.convert('L')

    # Przekonwertuj obraz na macierz numpy
    image_matrix = np.array(image_gray)

    print(image_matrix.shape)

    # Wyświetl macierz
    print(image_matrix)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    rysuj_macierz(ax, image_matrix, "kot", np.min(image_matrix), np.max(image_matrix))

    plt.show()

if __name__ == "__main__":
    main()
