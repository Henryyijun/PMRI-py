import matplotlib.pyplot as plt


def imshow(image, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(mask, cmap='gray')

    plt.show()