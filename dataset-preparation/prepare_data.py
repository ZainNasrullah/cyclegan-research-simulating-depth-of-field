import os
from PIL import Image, ImageOps
import argparse


def main(args):

    # load arguments
    resize_to = (args.size, args.size)
    input_dir = args.input
    out_dir = args.output

    # load files and track which ones are visited
    images = os.listdir(input_dir)
    visited = []

    for file in images:

        # open image
        path_to_file = os.path.join(input_dir, file)
        im = Image.open(path_to_file)

        # store resolution and file size
        file_size = os.path.getsize(path_to_file)
        resolution = im.size

        # skip if the image resolution and exact file size has been seen before
        unique_image = resolution + (file_size,)
        if unique_image not in visited and file_size > 100000:
            visited.append(unique_image)
        else:
            print("duplicate found:", file)
            continue

        # resize to specified size
        #im.thumbnail(resize_to, Image.ANTIALIAS)
        im = ImageOps.fit(im, resize_to, Image.ANTIALIAS)
        im.save(os.path.join(out_dir, file), "JPEG")

    print("done")


if __name__ == "__main__":

    # provide arguments for target resize, input and output directories
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=256)

    parser.add_argument(
        '-i', '--input', default=r"..\get-dataset-flickr\iphone_downloads")
    parser.add_argument('-o', '--output', default=r"selfies")

    args = parser.parse_args()

    main(args)
