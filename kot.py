from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def rysuj_macierz(ax, macierz, tytul, vmin, vmax):
    ax.imshow(macierz, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(tytul)
    ax.axis('off')

def main():
    image_path = 'kot.jpeg'

    # Załadowanie obrazu
    image = Image.open(image_path)

    # Konwertowanie obrazu na odcienie szarości
    image_gray = image.convert('L')

    image_matrix = np.array(image_gray)

    print(image_matrix)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    rysuj_macierz(ax, image_matrix, "kot", np.min(image_matrix), np.max(image_matrix))

    plt.show()

if __name__ == "__main__":
    main()
