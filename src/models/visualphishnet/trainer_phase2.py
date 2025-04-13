import logging
from argparse import ArgumentParser

import numpy as np
from tools.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

from HardSubsetSampling import HardSubsetSampling
from triplet_sampling import TargetHelper, get_batch
import data
from keras.models import load_model
from keras import backend as K
from tqdm import tqdm
import wandb


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


def save_keras_model(model, output_dir, new_saved_model_name):
    # TODO: save artifact to wandb
    model.save(output_dir / f'{new_saved_model_name}.h5')
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

    full_model = load_model(args.output_dir / f"{args.saved_model_name}.h5",
                            custom_objects={'loss': custom_loss(args.margin)})

    from keras import optimizers
    optimizer = optimizers.Adam(lr=args.lr)
    full_model.compile(loss=custom_loss(args.margin), optimizer=optimizer)

    return full_model


def train(run, args):
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
    data_path_phish = args.dataset_path / 'phishing'

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

        data_path_trusted = args.dataset_path / 'trusted_list'

        # Read images legit (train)
        targets_trusted = open(data_path_trusted / 'targets.txt', 'r').read()
        all_imgs_train, all_labels_train, all_file_names_train = data.read_imgs_per_website(data_path_trusted,
                                                                                            targets_trusted,
                                                                                            args.legit_imgs_num,
                                                                                            args.reshape_size, 0)


        imgs_train_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(imgs_train_path, all_imgs_train)
        np.save(labels_train_path, all_labels_train)
        np.save(file_names_train_path, all_file_names_train)

        # Read images phishing (test)
        targets_phishing = open(data_path_phish / 'targets.txt', 'r').read()
        all_imgs_test, all_labels_test, all_file_names_test = data.read_imgs_per_website(data_path_phish,
                                                                                         targets_phishing,
                                                                                         args.phish_imgs_num,
                                                                                         args.reshape_size, 0)

        imgs_test_path.parent.mkdir(parents=True, exist_ok=True)
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
                run.log({"loss": loss_iteration})

                if tot_count % args.save_interval == 0:
                    save_keras_model(full_model, args.output_dir, args.new_saved_model_name)

                if tot_count % args.lr_interval == 0:
                    args.lr = 0.99 * args.lr
                    K.set_value(full_model.optimizer.lr, args.lr)
                    logger.info("Learning rate changed to: " + str(args.lr))
                    run.log({'lr': args.lr})

    save_keras_model(full_model, args.output_dir, args.new_saved_model_name)
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
        parser.add_argument('--lr', type=float, default=2e-5)  # 0.00002
        parser.add_argument('--output-dir', type=str, default=INTERIM_DATA_DIR / 'smallerSampleDataset')
        parser.add_argument('--saved-model-name', type=str, default='model')  # from first training
        parser.add_argument('--new-saved-model-name', type=str, default='model2')
        parser.add_argument('--save-interval', type=int, default=200) # 2000
        parser.add_argument('--batch-size', type=int, default=32) # TODO: change to 32
        parser.add_argument('--n-iter', type=int, default=20) # 50000
        parser.add_argument('--lr-interval', type=int, default=250) # 250
        # hard examples training
        parser.add_argument('--num-sets', type=int, default=5) # 100
        parser.add_argument('--iter-per-set', type=int, default=8)
        # parser.add_argument('--n_iter', type=int, default=30)

        args = parser.parse_args()
        run = wandb.init(
            project="VisualPhish smallerSampleDataset - phase2",
            notes=f"VisualPhish smallerSampleDataset - phase2",
            config=args
        )
        try:
            train(run, args)
        except Exception as e:
            logger.error(e)
        finally:
            run.finish()
