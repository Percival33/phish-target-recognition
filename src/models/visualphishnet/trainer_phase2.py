import logging
from argparse import ArgumentParser

import numpy as np
from tools.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from triplet_sampling import TargetHelper, get_batch
import data
from keras.models import load_model
from keras import backend as K
from tqdm import tqdm


def order_random_array(orig_arr, y_orig_arr, targets):
    sorted_arr = np.zeros(orig_arr.shape)
    y_sorted_arr = np.zeros(y_orig_arr.shape)
    count = 0
    for i in range(0, targets):
        for j in range(0, orig_arr.shape[0]):
            if y_orig_arr[j] == i:
                sorted_arr[count, :, :, :] = orig_arr[j, :, :, :]
                y_sorted_arr[count, :] = i
                count = count + 1
    return sorted_arr, y_sorted_arr


# Store the start and end of each target in the phishing set (used later in triplet sampling)
# Not all targets might be in the phishing set
def start_end_each_target(num_target, labels):
    prev_target = 0
    start_end_each_target = np.zeros((num_target, 2))
    start_end_each_target[0, 0] = 0
    count_target = 0
    for i in range(1, labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[count_target, 1] = i - 1
            count_target = count_target + 1
            start_end_each_target[count_target, 0] = i
            prev_target = prev_target + 1
    start_end_each_target[num_target - 1, 1] = labels.shape[0] - 1
    return start_end_each_target


# Store the start and end of each target in the training set (used later in triplet sampling)
def all_targets_start_end(num_target, labels):
    prev_target = labels[0]
    start_end_each_target = np.zeros((num_target, 2))
    start_end_each_target[0, 0] = labels[0]
    if not labels[0] == 0:
        start_end_each_target[0, 0] = -1
        start_end_each_target[0, 1] = -1
    count_target = 0
    for i in range(1, labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[int(labels[i - 1]), 1] = int(i - 1)
            # count_target = count_target + 1
            start_end_each_target[int(labels[i]), 0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]), 1] = int(labels.shape[0] - 1)

    for i in range(1, num_target):
        if start_end_each_target[i, 0] == 0:
            print(i)
            start_end_each_target[i, 0] = -1
            start_end_each_target[i, 1] = -1
    return start_end_each_target


class HardSubsetSampling:
    # Compute distance between the query set and all training examples
    def compute_all_distances(self, fixed_set, train_legit, train_phish):
        # TODO: refactor with trainer_phase2.py
        train_size = train_legit.shape[0] + train_phish.shape[0]
        X_all_train = np.concatenate((train_legit, train_phish))
        pairwise_distance = np.zeros([fixed_set.shape[0], train_size])
        for i in range(0, fixed_set.shape[0]):
            pair1 = fixed_set[i, :]
            for j in range(0, train_size):
                pair2 = X_all_train[j, :]
                l2_diff = self.compute_distance_pair(pair1, pair2)
                pairwise_distance[i, j] = l2_diff
        return pairwise_distance

    # Main function for subset sampling
    # Steps:
    # Predict all images
    # Find pairwise distances between query and training set
    # Find indices of hard positive and negative examples
    # Find new training set
    # Order training set by targets
    def find_main_train(self, model, fixed_set, targets, X_train, y_train, X_train_new, y_train_new, X_train_legit,
                        X_train_phish):
        """
        :param model:
        :param fixed_set:
        :param targets:
        :param X_train:
        :param y_train:
        :param X_train_new:
        :param y_train_new:
        :param X_train_legit: all imgs train
        :param X_train_phish: phish test images
        :return:
        """
        X_train_legit_last_layer, X_train_phish_last_layer, fixed_set_last_layer = self.predict_all_imgs(model,
                                                                                                         X_train_legit=X_train_legit,
                                                                                                         X_train_phish=X_train_phish,
                                                                                                         fixed_set=fixed_set)
        pairwise_distance = self.compute_all_distances(fixed_set_last_layer, X_train_legit_last_layer,
                                                       X_train_phish_last_layer)
        n = 1
        all_idx = self.find_index_for_all_set(y_train, pairwise_distance, n)
        X_train_new, y_train_new = self.find_next_training_set(X_train=X_train, y_train=y_train,
                                                               X_train_new=X_train_new, y_train_new=y_train_new,
                                                               all_idx=all_idx, n=n)
        X_train_new, y_train_new = order_random_array(X_train_new, y_train_new, targets)
        labels_start_end_train = start_end_each_target(targets, y_train_new)
        return X_train_new, y_train_new, labels_start_end_train

    # Get the idx of false positives and false negatives for all query examples
    @classmethod
    def find_index_for_all_set(cls, y_train, distances, n):
        all_idx = np.zeros([distances.shape[0], 2, n])
        for i in range(0, distances.shape[0]):
            distance_i = distances[i, :]
            all_idx[i, 0, :] = cls.find_n_false_positives(y_train, distance_i, n, i)
            all_idx[i, 1, :] = cls.find_n_false_negatives(y_train, distance_i, n, i)
            return all_idx

    # Compute L2 distance between embeddings
    @staticmethod
    def compute_distance_pair(layer1, layer2):
        diff = layer1 - layer2
        l2_diff = np.mean(diff ** 2)
        return l2_diff

    # Compute the embeddings of the query set, the phishing training set, the training whitelist
    @staticmethod
    def predict_all_imgs(model, X_train_legit, X_train_phish, fixed_set):
        X_train_legit_last_layer = model.predict(X_train_legit, batch_size=10)
        X_train_phish_last_layer = model.predict(X_train_phish, batch_size=10)
        fixed_set_last_layer = model.predict(fixed_set, batch_size=10)

        return X_train_legit_last_layer, X_train_phish_last_layer, fixed_set_last_layer

    # Get index of false negatives (same-website examples with large distance) of one query image
    @staticmethod
    def find_n_false_negatives(y_train, distances, n, test_label):
        count = 0
        X_false_neg_idx = np.zeros([n, ])
        idx_max = np.argsort(distances)[::-1]
        for i in range(0, distances.shape[0]):
            next_max_idx = idx_max[i]
            n_label = y_train[next_max_idx]
            # false negatives (have large distance, although they are in the same category )
            if test_label == n_label:
                X_false_neg_idx[count] = next_max_idx
                count = count + 1
                if count == n:
                    break
        while count < n:
            idx_max[count] = -1
            count = count + 1
        return X_false_neg_idx

    # Find a query set for each target
    @staticmethod
    def find_fixed_set_idx(labels_start_end_train_legit, num_target):
        website_random_idx = np.zeros([num_target, ])
        for i in range(0, num_target):
            class_idx_start_end = labels_start_end_train_legit[i, :]
            website_random_idx[i] = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
        return website_random_idx

    # Get index of false positives (different-website examples with small distance) of one query image
    @staticmethod
    def find_n_false_positives(y_train, distances, n, test_label):
        count = 0
        X_false_pos_idx = np.zeros([n, ])
        idx_min = np.argsort(distances)
        for i in range(0, distances.shape[0]):
            next_min_idx = idx_min[i]
            n_label = y_train[next_min_idx]
            # false positives (have close distance even if they are from different category)
            if not (test_label == n_label):
                X_false_pos_idx[count] = next_min_idx
                count = count + 1
                if count == n:
                    break
        while count < n:
            idx_min[count] = -1
            count = count + 1
        return X_false_pos_idx

    # Form the new training set based on the hard examples indices of all query images
    @staticmethod
    def find_next_training_set(X_train, y_train, X_train_new, y_train_new, all_idx, n):
        # global X_train_new, y_train_new
        # FIXME: ?????? - does it work?
        all_idx = all_idx.astype(int)
        count = 0
        for i in range(all_idx.shape[0]):
            for j in range(0, n):
                if not all_idx[i, 0, j] == -1:
                    X_train_new[count, :, :, :] = X_train[all_idx[i, 0, j], :, :, :]
                    y_train_new[count, :] = y_train[all_idx[i, 0, j]]
                    count = count + 1
            for j in range(0, n):
                if not all_idx[i, 1, j] == -1:
                    X_train_new[count, :, :, :] = X_train[all_idx[i, 1, j], :, :, :]
                    y_train_new[count, :] = y_train[all_idx[i, 1, j]]
                    count = count + 1
        X_train_new = X_train_new[0:count, :]
        y_train_new = y_train_new[0:count, :]
        return X_train_new, y_train_new


def save_keras_model(model, output_dir, new_saved_model_name):
    # TODO: save artifact to wandb
    model.save((output_dir / new_saved_model_name).with_suffix('.h5'))
    logger.info("Saved model to disk")


def prepare_model(args):
    def loss(y_true, y_pred, margin):
        loss_value = K.maximum(y_true, margin + y_pred)
        loss_value = K.mean(loss_value, axis=0)
        return loss_value

    def custom_loss(margin):
        def loss(y_true, y_pred):
            loss_value = K.maximum(y_true, margin + y_pred)
            loss_value = K.mean(loss_value, axis=0)
            return loss_value

        return loss

    full_model = load_model((args.output_dir / args.saved_model_name).with_suffix('.h5'),
                            custom_objects={'loss': lambda y_true, y_pred: loss(y_true, y_pred, args.margin)})

    from keras import optimizers
    optimizer = optimizers.Adam(lr=args.start_lr)
    full_model.compile(loss=custom_loss(args.margin), optimizer=optimizer)

    return full_model


def train(args):
    logger.info('Training model')
    # log dataset hash

    logger.info('Check for pre-saved data or load images')

    # Define paths for saved .npy files
    imgs_train_path = args.output_dir / 'all_imgs_train.npy'
    labels_train_path = args.output_dir / 'all_labels_train.npy'
    file_names_train_path = args.output_dir / 'all_file_names_train.npy'

    imgs_test_path = args.output_dir / 'all_imgs_test.npy'
    labels_test_path = args.output_dir / 'all_labels_test.npy'
    file_names_test_path = args.output_dir / 'all_file_names_test.npy'

    # Initialize variables
    all_imgs_train, all_labels_train, all_file_names_train = None, None, None
    all_imgs_test, all_labels_test, all_file_names_test = None, None, None

    # Check if all .npy files exist
    if (imgs_train_path.exists() and labels_train_path.exists() and file_names_train_path.exists() and
            imgs_test_path.exists() and labels_test_path.exists() and file_names_test_path.exists()):
        logger.info('Loading pre-saved data')

        # Load pre-saved data
        all_imgs_train = np.load(imgs_train_path)
        all_labels_train = np.load(labels_train_path)
        all_file_names_train = np.load(file_names_train_path)

        all_imgs_test = np.load(imgs_test_path)
        all_labels_test = np.load(labels_test_path)
        all_file_names_test = np.load(file_names_test_path)

    else:
        logger.info('Processing and saving images')

        # Read images legit (train)
        data_path_trusted = args.dataset_path / 'trusted_list'
        targets_trusted = open(data_path_trusted / 'targets.txt', 'r').read()
        all_imgs_train, all_labels_train, all_file_names_train = data.read_imgs_per_website(data_path_trusted,
                                                                                            targets_trusted,
                                                                                            args.legit_imgs_num,
                                                                                            args.reshape_size, 0)

        np.save(imgs_train_path, all_imgs_train)
        np.save(labels_train_path, all_labels_train)
        np.save(file_names_train_path, all_file_names_train)

        # Read images phishing (test)
        data_path_phish = args.dataset_path / 'phishing'
        targets_phishing = open(data_path_phish / 'targets.txt', 'r').read()
        all_imgs_test, all_labels_test, all_file_names_test = data.read_imgs_per_website(data_path_phish,
                                                                                         targets_phishing,
                                                                                         args.phish_imgs_num,
                                                                                         args.reshape_size, 0)

        np.save(imgs_test_path, all_imgs_test)
        np.save(labels_test_path, all_labels_test)
        np.save(file_names_test_path, all_file_names_test)

    logger.info('Images loaded')

    X_train_legit = all_imgs_train
    y_train_legit = all_labels_train
    # Load the same train/split in phase 1
    phish_test_idx = np.load(args.output_dir / 'test_idx.npy')
    phish_train_idx = np.load(args.output_dir / 'train_idx.npy')

    X_test_phish = all_imgs_test[phish_test_idx, :]
    y_test_phish = all_labels_test[phish_test_idx, :]

    X_train_phish = all_imgs_test[phish_train_idx, :]
    y_train_phish = all_labels_test[phish_train_idx, :]

    labels_start_end_train_legit = all_targets_start_end(args.num_targets, y_train_legit)
    targetHelper = TargetHelper(data_path_phish)

    full_model = prepare_model(args)
    hard_subset_sampling = HardSubsetSampling()
    #########################################################################################
    n = 1  # number of wrong points

    # all training images
    X_train = np.concatenate([X_train_legit, X_train_phish])
    y_train = np.concatenate([y_train_legit, y_train_phish])

    # subset training
    X_train_new = np.zeros(
        [args.num_targets * 2 * n, X_train_legit.shape[1], X_train_legit.shape[2], X_train_legit.shape[3]])
    y_train_new = np.zeros([args.num_targets * 2 * n, 1])

    targets_train = np.zeros([args.batch_size, 1])
    tot_count = 0

    logger.info("Starting training process!")
    for k in tqdm(range(0, args.num_sets)):
        logger.info("Starting a new set!")
        # print("\n ------------- \n")
        X_train_legit = all_imgs_train
        y_train_legit = all_labels_train

        fixed_set_idx = hard_subset_sampling.find_fixed_set_idx(
            labels_start_end_train_legit=labels_start_end_train_legit, num_target=args.num_targets)
        fixed_set = X_train_legit[fixed_set_idx.astype(int), :, :, :]

        # TODO: add tqdm sub (progress) bar
        for j in range(0, args.iter_per_set):
            # TODO: log iteration to wandb
            model = full_model.layers[3]
            X_train_new, y_train_new, labels_start_end_train = hard_subset_sampling.find_main_train(
                model=model,
                fixed_set=fixed_set,
                targets=args.num_targets,
                X_train=X_train,
                y_train=y_train,
                X_train_new=X_train_new,
                y_train_new=y_train_new,
                X_train_legit=X_train_legit,
                X_train_phish=X_train_phish
            )

            for i in range(1, args.n_iter):
                tot_count = tot_count + 1
                inputs = get_batch(
                    targetHelper=targetHelper,
                    X_train_legit=X_train_legit,
                    X_train_new=X_train_new,
                    labels_start_end_train=labels_start_end_train,
                    batch_size=args.batch_size,
                    train_fixed_set=fixed_set,
                    num_targets=args.num_targets
                )
                loss_iteration = full_model.train_on_batch(inputs, targets_train)

                # print("\n ------------- \n")
                logger.info('Iteration: ' + str(i) + '. ' + "Loss: {0}".format(loss_iteration))
                # TODO: log loss to wandb

                if tot_count % args.save_interval == 0:
                    save_keras_model(full_model, args.output_dir, args.new_saved_model_name)

                if tot_count % args.lr_interval == 0:
                    start_lr = 0.99 * start_lr
                    K.set_value(full_model.optimizer.lr, start_lr)
                    logger.info("Learning rate changed to: " + str(start_lr))
                    # TODO: log learning rate to wandb

    save_keras_model(full_model, args.output_dir, args.new_saved_model_name)

    logger.info("Training finished!")

    # TODO: save artifact to wandb
    logger.info("Calculating embeddings for whitelist and phishing set")
    shared_model = full_model.layers[3]

    whitelist_emb = shared_model.predict(X_train_legit, batch_size=64)
    np.save(args.output_dir / 'whitelist_emb2', whitelist_emb)
    np.save(args.output_dir / 'whitelist_labels2', y_train_legit)

    phishing_emb = shared_model.predict(all_imgs_test, batch_size=64)
    np.save(args.output_dir / 'phishing_emb2', phishing_emb)
    np.save(args.output_dir / 'phishing_labels2', all_labels_test)
    logger.info("Embeddings saved to disk")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger()
    logger.info("VisualPhish - Training phase 2")

    init_parser = ArgumentParser(add_help=False)
    # TODO: enable wandb sweep
    init_parser.add_argument("--use-sweep", action="store_true", default=False)
    init_args, _ = init_parser.parse_known_args()

    if init_args.use_sweep:
        # TODO: wandb sweep
        # with open(LOGS_PATH("sweep_config.yaml")) as f:
        #     sweep_config = yaml.safe_load(f)
        # sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT)
        # wandb.agent(sweep_id, function=train, count=12)
        pass
    else:
        parser = ArgumentParser(parents=[init_parser])
        # Dataset parameters
        parser.add_argument('--dataset-path', type=str, default=PROCESSED_DATA_DIR / 'smallerSampleDataset')
        parser.add_argument('--reshape-size', default=[224, 224, 3])
        parser.add_argument('--phishing-test-size', default=0.4)
        parser.add_argument('--num-targets', type=int, default=5)
        parser.add_argument('--legit-imgs-num', default=420)
        parser.add_argument('--phish-imgs-num', default=160)
        # Model parameters
        parser.add_argument('--input-shape', default=[224, 224, 3])
        parser.add_argument('--margin', type=float, default=2.2)
        parser.add_argument('--new-conv-params', default=[5, 5, 512])
        # Training parameters
        parser.add_argument('--start-lr', type=float, default=2e-5)  # 0.00002
        parser.add_argument('--output-dir', type=str, default=INTERIM_DATA_DIR / 'smallerSampleDataset')
        parser.add_argument('--saved-model-name', type=str, default='model')  # from first training
        parser.add_argument('--new-saved-model-name', type=str, default='model2')
        parser.add_argument('--save-interval', type=int, default=2000)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--n-iter', type=int, default=50000)
        parser.add_argument('--lr-interval', type=int, default=250)
        # hard examples training
        parser.add_argument('--num-sets', type=int, default=100)
        parser.add_argument('--iter-per-set', type=int, default=8)
        # parser.add_argument('--n_iter', type=int, default=30)

        train(parser.parse_args())
