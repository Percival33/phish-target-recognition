import logging
from argparse import ArgumentParser

import numpy as np

from keras import backend as K
from tqdm import tqdm
import wandb

from HardSubsetSampling import HardSubsetSampling
from TargetHelper import TargetHelper
from RandomSampling import RandomSampling
from ModelHelper import ModelHelper
from triplet_sampling import get_batch_for_phase2
from tools.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
import DataHelper as data


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
            count_target = count_target + 1
            start_end_each_target[int(labels[i]), 0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]), 1] = int(labels.shape[0] - 1)

    for i in range(1, num_target):
        if start_end_each_target[i, 0] == 0:
            print(i)
            start_end_each_target[i, 0] = -1
            start_end_each_target[i, 1] = -1
    return start_end_each_target


# Order random phishing arrays per website (from 0 to 155 target)

def order_random_array(orig_arr, y_orig_arr, targets):
    # TODO: remove duplicate with HardSubsetSampling
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
def targets_start_end(num_target, labels):
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
            start_end_each_target[int(labels[i]), 0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]), 1] = int(labels.shape[0] - 1)

    for i in range(1, num_target):
        if start_end_each_target[i, 0] == 0:
            start_end_each_target[i, 0] = -1
            start_end_each_target[i, 1] = -1
    return start_end_each_target


def train_phase1(run, args):
    logger.info('Trainer phase 1')

    all_imgs_train, all_labels_train, all_file_names_train, all_imgs_test, all_labels_test, all_file_names_test = data.read_or_load_imgs(
        args)
    logger.info('Images loaded')

    X_train_legit = all_imgs_train
    y_train_legit = all_labels_train

    # TODO: if not existing, create this split -> log as artifact
    idx_test, idx_train = data.read_or_load_train_test_idx(output_dir=args.dataset_path, all_imgs_test=all_imgs_test,
                                                           all_labels_test=all_labels_test,
                                                           phishing_test_size=args.phishing_test_size)

    X_test_phish = all_imgs_test[idx_test, :]
    y_test_phish = all_labels_test[idx_test, :]

    X_train_phish = all_imgs_test[idx_train, :]
    y_train_phish = all_labels_test[idx_train, :]

    # create model
    modelHelper = ModelHelper()
    model = modelHelper.prepare_model(args.input_shape, args.new_conv_params, args.margin, args.lr)

    # order random array? -> po co?
    X_test_phish, y_test_phish = order_random_array(X_test_phish, y_test_phish, args.num_targets)
    X_train_phish, y_train_phish = order_random_array(X_train_phish, y_train_phish, args.num_targets)

    # labels_start_end_train_phish, labels_start_end_test_phish
    labels_start_end_train_phish = targets_start_end(args.num_targets, y_train_phish)
    labels_start_end_test_phish = targets_start_end(args.num_targets, y_test_phish)
    # labels_start_end_train_legit
    labels_start_end_train_legit = all_targets_start_end(args.num_targets, y_train_legit)

    targetHelper = TargetHelper(args.dataset_path / 'phishing')
    randomSampling = RandomSampling(targetHelper, labels_start_end_train_phish, labels_start_end_test_phish,
                                    labels_start_end_train_legit)
    # training
    logger.info("Starting training process! - phase 1")

    targets_train = np.zeros([args.batch_size, 1])
    run.log({'lr': args.lr})

    for i in range(1, args.n_iter):
        inputs = randomSampling.get_batch(targetHelper=targetHelper, X_train_legit=X_train_legit,
                                          y_train_legit=y_train_legit, X_train_phish=X_train_phish,
                                          labels_start_end_train_legit=labels_start_end_train_legit,
                                          batch_size=args.batch_size, num_targets=args.num_targets)
        loss_value = model.train_on_batch(inputs, targets_train)

        logger.info('Iteration: ' + str(i) + '. ' + "Loss: {0}".format(loss_value))
        run.log({"loss": loss_value})

        if i % args.save_interval == 0:
            # TODO: log model artifact if better accuracy
            testResults = modelHelper.get_embeddings(model, X_train_legit, y_train_legit, all_imgs_test,
                                                     all_labels_test, train_idx=idx_train,
                                                     test_idx=idx_test)
            acc = modelHelper.get_acc(testResults, args.dataset_path / 'trusted_list', args.dataset_path / 'phishing')
            run.log({"acc": acc})
            modelHelper.save_model(model, args.output_dir, args.saved_model_name)

        if i % args.lr_interval == 0:
            args.lr = 0.99 * args.lr
            K.set_value(model.optimizer.lr, args.lr)
            run.log({'lr': args.lr})

    modelHelper.save_model(model, args.output_dir, args.saved_model_name)
    # TODO: log artifact

    # TODO: log embeddings as artifacts
    # TODO: save model and cal embeddings -> log artifact
    # TODO: na bieżąco obliczaj jakość modelu


def train_phase2(run, args):
    logger.info('Trainer phase 2')
    # TODO: log dataset hash

    # Initialize variables
    data_path_phish = args.dataset_path / 'phishing'
    all_imgs_train, all_labels_train, all_file_names_train, all_imgs_test, all_labels_test, all_file_names_test = data.read_or_load_imgs(args)
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

    modelHelper = ModelHelper()
    full_model = modelHelper.load_trained_model(args.output_dir, args.saved_model_name, args.margin, args.lr)
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

    logger.info("Starting training process! - phase 2")
    run.log({'lr': args.lr})
    for k in tqdm(range(0, args.num_sets), desc="Sets"):
        logger.info(f"Starting a new set! - {k}")
        # print("\n ------------- \n")
        X_train_legit = all_imgs_train
        y_train_legit = all_labels_train

        fixed_set_idx = hard_subset_sampling.find_fixed_set_idx(
            labels_start_end_train_legit=labels_start_end_train_legit, num_target=args.num_targets)
        fixed_set = X_train_legit[fixed_set_idx.astype(int), :, :, :]

        for j in tqdm(range(0, args.iter_per_set), desc="Iterations of set", leave=False):
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

            for i in range(1, args.hard_n_iter):
                tot_count = tot_count + 1
                inputs = get_batch_for_phase2(
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
                run.log({"loss": loss_iteration})

                if tot_count % args.save_interval == 0:
                    # TODO: log model artifact if better accuracy
                    testResults = modelHelper.get_embeddings(model, X_train_legit, y_train_legit, all_imgs_test,
                                                             all_labels_test, train_idx=phish_train_idx,
                                                             test_idx=phish_test_idx)
                    acc = modelHelper.get_acc(testResults, args.dataset_path / 'trusted_list',
                                              args.dataset_path / 'phishing')
                    run.log({"acc": acc})
                    modelHelper.save_model(model, args.output_dir, args.saved_model_name)

                if tot_count % args.lr_interval == 0:
                    args.lr = 0.99 * args.lr
                    K.set_value(full_model.optimizer.lr, args.lr)
                    logger.info("Learning rate changed to: " + str(args.lr))
                    run.log({'lr': args.lr})

    modelHelper.save_model(full_model, args.output_dir, args.new_saved_model_name)
    run.log_model(args.output_dir / f'{args.new_saved_model_name}.h5')
    logger.info("Training finished!")

    logger.info("Calculating embeddings for whitelist and phishing set")
    shared_model = full_model.layers[3]

    whitelist_emb = shared_model.predict(X_train_legit, batch_size=64)
    np.save(args.output_dir / 'whitelist_emb2', whitelist_emb)
    np.save(args.output_dir / 'whitelist_labels2', y_train_legit)
    run.save(str(args.output_dir / 'whitelist_emb2.npy'))
    run.save(str(args.output_dir / 'whitelist_labels2.npy'))

    phishing_emb = shared_model.predict(all_imgs_test, batch_size=64)
    np.save(args.output_dir / 'phishing_emb2', phishing_emb)
    np.save(args.output_dir / 'phishing_labels2', all_labels_test)
    run.save(str(args.output_dir / 'phishing_emb2.npy'))
    run.save(str(args.output_dir / 'phishing_labels2.npy'))
    logger.info("Embeddings saved to disk")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger()
    logger.info("VisualPhish - trainer")

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
        parser.add_argument('--lr', type=float, default=2e-5)  # 0.00002
        parser.add_argument('--output-dir', type=str, default=INTERIM_DATA_DIR / 'smallerSampleDataset')
        parser.add_argument('--saved-model-name', type=str, default='model')  # from first training
        parser.add_argument('--new-saved-model-name', type=str, default='model2')
        parser.add_argument('--save-interval', type=int, default=100)  # 2000
        parser.add_argument('--batch-size', type=int, default=16)  # TODO: change to 32
        parser.add_argument('--n-iter', type=int, default=200)  # p1: 21000, p2: 50000
        parser.add_argument('--lr-interval', type=int, default=250)  # p1: 100, p2: 250
        # hard examples training
        parser.add_argument('--num-sets', type=int, default=5)  # 100
        parser.add_argument('--iter-per-set', type=int, default=8)
        parser.add_argument('--hard-n-iter', type=int, default=30)

        args = parser.parse_args()
        run = wandb.init(
            project="VisualPhish smallerSampleDataset",
            group="visualphishnet",
            config=args,
            tags=["gabi", "phase-1"]
        )
        try:
            train_phase1(run, args)
            run.finish()
            run = wandb.init(
                project="VisualPhish smallerSampleDataset",
                group="visualphishnet",
                config=args,
                tags=["gabi", "phase-2"]
            )
            train_phase2(run, args)
        except Exception as e:
            logger.error(e)
        finally:
            run.finish()
