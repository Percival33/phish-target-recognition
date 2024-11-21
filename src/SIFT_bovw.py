from scipy.cluster._vq import vq
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import cv2


def read_bw_images(img_paths):
    return [
        cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        if len(cv2.imread(img_path).shape) == 3
        else cv2.imread(img_path)
        for img_path in img_paths
    ]


def clean_descriptors(
    keypoints: list[cv2.KeyPoint], descriptors: list[np.ndarray]
) -> tuple[list[cv2.KeyPoint], list[np.ndarray]]:
    print(f"len before: {len(descriptors)}")
    # initialize list to store idx values of records to drop
    to_drop = []
    for i, img_descriptors in enumerate(descriptors):
        # if there are no descriptors, add record idx to drop list
        if img_descriptors is None:
            to_drop.append(i)

    print(f"indexes: {to_drop}")
    # delete from list in reverse order
    for i in sorted(to_drop, reverse=True):
        del descriptors[i], keypoints[i]

    print(f"len after: {len(descriptors)}")
    return keypoints, descriptors


def sift_features(images, clean=True) -> tuple[list[cv2.KeyPoint], list[np.ndarray]]:
    """
    Run SIFT on a list of black and white images
    :param images:
    :return: list of keypoints and descriptors
    """
    sift = cv2.SIFT_create()
    keypoints_agg: list[cv2.KeyPoint] = []
    descriptors_agg: list[np.ndarray] = []
    for img in images:
        # extract keypoints and descriptors for each image
        img_keypoints, img_descriptors = sift.detectAndCompute(img, None)
        keypoints_agg.append(img_keypoints)
        descriptors_agg.append(img_descriptors)

    return clean_descriptors(
        keypoints_agg, descriptors_agg
    ) if clean else keypoints_agg, descriptors_agg


def get_visual_words(descriptors, codebook):
    visual_words = []
    for img_descriptors in descriptors:
        # for each image, map each descriptor to the nearest codebook entry
        img_visual_words, distance = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)
    return visual_words


def get_frequency_vectors(visual_words, k):
    frequency_vectors = []
    for img_visual_words in visual_words:
        # create a frequency vector for each image
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    return np.stack(frequency_vectors)


def get_tfidf(frequency_vectors):
    """
    Compute the TF-IDF for the frequency vectors
    :param frequency_vectors:
    :return:
    """
    N = len(frequency_vectors)
    df = np.sum(frequency_vectors > 0, axis=0)
    print(f"df.shape, df[:5]: {df.shape}, {df[:5]}")
    idf = np.log(N / df)
    print(f"idf.shape, idf[:5]: {idf.shape}, {idf[:5]}")
    return frequency_vectors * idf


def search_test(
    image: np.ndarray,
    db: np.ndarray,
    top_k: int = 5,
    search_image=None,
    db_images: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search for the most similar images in the database
    :param image: tfidf vector for the search image
    :param db: tfidf vectors for the database images
    :param top_k: number of similar images to return
    :param search_image: (optional) black and white image to search for
    :param db_images: (optional) list of black and white images in the database
    :return: indices of k the most similar images and their cosine similarity
    """
    if search_image is not None:
        # assert search image is black and white
        assert len(search_image.shape) == 2
        print("Search image:")
        plt.imshow(search_image, cmap="gray")
        plt.show()
        print("-----------------------------------------------------")

    a = image
    b = db

    # get the cosine distance for the search image `a`
    cosine_similarity = np.dot(a, b.T) / (norm(a) * norm(b, axis=1))
    # get the top k indices for most similar vecs
    idx = np.argsort(-cosine_similarity)[:top_k]

    # display the results
    if db_images:
        for i in idx:
            # assert db images are black and white
            assert len(db_images[i].shape) == 2
            print(f"{i}: {round(cosine_similarity[i], 4)}")
            plt.imshow(db_images[i], cmap="gray")
            plt.show()

    return idx, cosine_similarity[idx]