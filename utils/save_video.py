import os
import imageio


def save_video(folder, output_filename, fps=30):
    image = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.png'):
            image.append(imageio.v2.imread(os.path.join(folder, filename)))
    imageio.mimsave(output_filename, image, fps=fps)
