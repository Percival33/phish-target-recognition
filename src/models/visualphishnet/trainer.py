import gc
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend import clear_session, get_session, set_session
from tools.config import INTERIM_DATA_DIR, LOGS_DIR, PROCESSED_DATA_DIR, setup_logging

import DataHelper as data
import wandb
from HardSubsetSampling import HardSubsetSampling
from ModelHelper import ModelHelper
from RandomSampling import RandomSampling
from TargetHelper import TargetHelper
from triplet_sampling import get_batch_for_phase2


def train_phase1(run, args):
    logger.info("Trainer phase 1")
    (
        all_imgs_train,
        all_labels_train,
        all_file_names_train,
        all_imgs_test,
        all_labels_test,
        all_file_names_test,
    ) = data.read_or_load_imgs(args)
    logger.info("Images loaded")

    X_train_legit = all_imgs_train
    y_train_legit = all_labels_train

    idx_test, idx_train = data.read_or_load_train_test_idx(
        dirname=args.output_dir,
        all_imgs_test=all_imgs_test,
        all_labels_test=all_labels_test,
        phishing_test_size=args.phishing_test_size,
    )
    run.save(str(args.output_dir / "test_idx.npy"))
    run.save(str(args.output_dir / "train_idx.npy"))

    X_test_phish = all_imgs_test[idx_test, :]
    y_test_phish = all_labels_test[idx_test, :]

    X_train_phish = all_imgs_test[idx_train, :]
    y_train_phish = all_labels_test[idx_train, :]

    # create model
    modelHelper = ModelHelper()
    # with tf.profiler.experimental.Trace("model_loading", step_num=1):
    model = modelHelper.prepare_model(args.input_shape, args.new_conv_params, args.margin, args.lr)
    logger.debug("Model prepared")

    # order random array? -> po co?
    X_test_phish, y_test_phish = data.order_random_array(X_test_phish, y_test_phish, args.num_targets)
    X_train_phish, y_train_phish = data.order_random_array(X_train_phish, y_train_phish, args.num_targets)
    logger.debug("Phishing arrays ordered")

    # labels_start_end_train_phish, labels_start_end_test_phish
    labels_start_end_train_phish = data.targets_start_end(args.num_targets, y_train_phish)
    labels_start_end_test_phish = data.targets_start_end(args.num_targets, y_test_phish)
    # labels_start_end_train_legit
    labels_start_end_train_legit = data.all_targets_start_end(args.num_targets, y_train_legit, logger)
    logger.debug("Targets start and end calculated")

    targetHelper = TargetHelper(args.dataset_path / "phishing")
    randomSampling = RandomSampling(
        targetHelper,
        labels_start_end_train_phish,
        labels_start_end_test_phish,
        labels_start_end_train_legit,
    )
    logger.debug("Random sampling initialized")
    # training
    logger.info("Starting training process! - phase 1")

    targets_train = np.zeros([args.batch_size, 1])
    run.log({"lr": args.lr})

    # for i in tqdm(range(1, args.n_iter), desc="Training Iterations", position=0, leave=True):
    for i in range(1, args.n_iter):
        # with tf.profiler.experimental.Trace("next_batch", step_num=i):
        inputs = randomSampling.get_batch(
            targetHelper=targetHelper,
            X_train_legit=X_train_legit,
            y_train_legit=y_train_legit,
            X_train_phish=X_train_phish,
            labels_start_end_train_legit=labels_start_end_train_legit,
            batch_size=args.batch_size,
            num_targets=args.num_targets,
        )

        # with tf.profiler.experimental.Trace("training", step_num=i):
        loss_value = model.train_on_batch(inputs, targets_train)

        if i % 100 == 0:
            logger.info(f"Memory usage {modelHelper.get_model_memory_usage(args.batch_size, model)}")
            logger.info(f"Iteration: {i}. Loss: {loss_value}")

        run.log({"loss": loss_value})

        if i % args.save_interval == 0:
            # TODO: log model artifact if better accuracy
            # with tf.profiler.experimental.Trace("get_acc", step_num=i):
            if i % (3 * args.save_interval) == 0:
                testResults = modelHelper.get_embeddings(
                    model,
                    X_train_legit,
                    y_train_legit,
                    all_imgs_test,
                    all_labels_test,
                    train_idx=idx_train,
                    test_idx=idx_test,
                )
                acc = modelHelper.get_acc(
                    targetHelper,
                    testResults,
                    args.dataset_path / "trusted_list",
                    args.dataset_path / "phishing",
                    all_file_names_train,
                    all_file_names_test,
                )

                run.log({"acc": acc})
            # with tf.profiler.experimental.Trace("save_model", step_num=i):
            modelHelper.save_model(model, args.output_dir, args.saved_model_name)
            run.log_model(args.output_dir / f"{args.saved_model_name}.h5")

        if i % args.lr_interval == 0:
            args.lr = 0.99 * args.lr
            K.set_value(model.optimizer.lr, args.lr)
            run.log({"lr": args.lr})

    modelHelper.save_model(model, args.output_dir, args.saved_model_name)
    run.log_model(args.output_dir / f"{args.saved_model_name}.h5")
    logger.info("Training finished!")
    logger.info("Calculating embeddings for whitelist and phishing set")

    emb = modelHelper.get_embeddings(
        model,
        X_train_legit,
        y_train_legit,
        all_imgs_test,
        all_labels_test,
        train_idx=idx_train,
        test_idx=idx_test,
    )
    data.save_embeddings(emb, args.output_dir, run)
    acc = modelHelper.get_acc(
        targetHelper,
        emb,
        args.dataset_path / "trusted_list",
        args.dataset_path / "phishing",
        all_file_names_train,
        all_file_names_test,
    )
    run.log({"acc": acc})
    logger.info("Phase 1 has finished!")


def train_phase2(run, args):
    logger.info("Trainer phase 2")
    # TODO: log dataset hash
    # Initialize variables
    data_path_phish = args.dataset_path / "phishing"
    (
        all_imgs_train,
        all_labels_train,
        all_file_names_train,
        all_imgs_test,
        all_labels_test,
        all_file_names_test,
    ) = data.read_or_load_imgs(args)
    logger.info("Images loaded")

    X_train_legit = all_imgs_train
    y_train_legit = all_labels_train
    # Load the same train/split in phase 1
    idx_test = np.load(args.output_dir / "test_idx.npy")
    idx_train = np.load(args.output_dir / "train_idx.npy")

    # X_test_phish = all_imgs_test[idx_test, :]
    # y_test_phish = all_labels_test[idx_test, :]

    X_train_phish = all_imgs_test[idx_train, :]
    y_train_phish = all_labels_test[idx_train, :]

    labels_start_end_train_legit = data.all_targets_start_end(args.num_targets, y_train_legit, logger)
    targetHelper = TargetHelper(data_path_phish)

    modelHelper = ModelHelper()
    # with tf.profiler.experimental.Trace("model_loading2", step_num=1):
    full_model = modelHelper.load_trained_model(args.output_dir, args.saved_model_name, args.margin, args.lr)

    hard_subset_sampling = HardSubsetSampling()
    #########################################################################################
    n = 1  # number of wrong points

    # all training images
    X_train = np.concatenate([X_train_legit, X_train_phish])
    y_train = np.concatenate([y_train_legit, y_train_phish])

    # subset training
    X_train_new = np.zeros(
        [
            args.num_targets * 2 * n,
            X_train_legit.shape[1],
            X_train_legit.shape[2],
            X_train_legit.shape[3],
        ]
    )
    y_train_new = np.zeros([args.num_targets * 2 * n, 1])

    targets_train = np.zeros([args.batch_size, 1])

    logger.info("Starting training process! - phase 2")
    run.log({"lr": args.lr})
    total_iterations = 0
    # for k in tqdm(range(0, args.num_sets), desc="Sets"):
    for k in range(0, args.num_sets):
        logger.info(f"Starting a new set! - {k}")
        X_train_legit = all_imgs_train
        y_train_legit = all_labels_train

        fixed_set_idx = hard_subset_sampling.find_fixed_set_idx(
            labels_start_end_train_legit=labels_start_end_train_legit,
            num_target=args.num_targets,
        )
        fixed_set = X_train_legit[fixed_set_idx.astype(int), :, :, :]

        # for j in tqdm(range(0, args.iter_per_set), desc="Iterations of set"):
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
                X_train_phish=X_train_phish,
            )

            # for i in tqdm(range(1, args.hard_n_iter), desc="Hard Iterations"):
            for i in range(1, args.hard_n_iter):
                total_iterations += 1
                # with tf.profiler.experimental.Trace("next_batch2", step_num=tot_count):
                inputs = get_batch_for_phase2(
                    targetHelper=targetHelper,
                    X_train_legit=X_train_legit,
                    X_train_new=X_train_new,
                    labels_start_end_train=labels_start_end_train,
                    batch_size=args.batch_size,
                    train_fixed_set=fixed_set,
                    num_targets=args.num_targets,
                )

                # with tf.profiler.experimental.Trace("training2", step_num=tot_count):
                loss_iteration = full_model.train_on_batch(inputs, targets_train)

                if total_iterations % 100 == 0:
                    logger.info(f"Memory usage {modelHelper.get_model_memory_usage(args.batch_size, model)}")
                    logger.info(f"Set: {k} SetIteration: {j} Iteration: {i}. Loss: {loss_iteration}")
                run.log({"loss": loss_iteration})

                if total_iterations % (3 * args.save_interval) == 0:
                    # TODO: log model artifact if better accuracy
                    # with tf.profiler.experimental.Trace("get_acc2", step_num=i):
                    testResults = modelHelper.get_embeddings(
                        full_model,
                        X_train_legit,
                        y_train_legit,
                        all_imgs_test,
                        all_labels_test,
                        train_idx=idx_train,
                        test_idx=idx_test,
                    )
                    acc = modelHelper.get_acc(
                        targetHelper,
                        testResults,
                        args.dataset_path / "trusted_list",
                        args.dataset_path / "phishing",
                        all_file_names_train,
                        all_file_names_test,
                    )

                    run.log({"acc": acc})
                    # with tf.profiler.experimental.Trace("save_model2", step_num=i):
                    modelHelper.save_model(full_model, args.output_dir, args.new_saved_model_name)
                    run.log_model(args.output_dir / f"{args.new_saved_model_name}.h5")

                if total_iterations % args.lr_interval == 0:
                    args.lr = 0.99 * args.lr
                    K.set_value(full_model.optimizer.lr, args.lr)
                    logger.info(f"Learning rate changed to: {args.lr}")
                    run.log({"lr": args.lr})

            # del X_train_new
            # X_train_new
            # del y_train_new
            # del labels_start_end_train
            reset_keras(model)

    modelHelper.save_model(full_model, args.output_dir, args.new_saved_model_name)
    run.log_model(args.output_dir / f"{args.new_saved_model_name}.h5")
    logger.info("Training finished!")
    logger.info("Calculating embeddings for whitelist and phishing set")
    emb = modelHelper.get_embeddings(
        full_model,
        X_train_legit,
        y_train_legit,
        all_imgs_test,
        all_labels_test,
        train_idx=idx_train,
        test_idx=idx_test,
    )
    data.save_embeddings(emb, args.output_dir, run)

    acc = modelHelper.get_acc(
        targetHelper,
        emb,
        args.dataset_path / "trusted_list",
        args.dataset_path / "phishing",
        all_file_names_train,
        all_file_names_test,
    )
    run.log({"acc": acc})
    logger.info("Phase 2 has finished!")


def reset_keras(model):
    # Clear the Keras backend
    tf.keras.backend.clear_session()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass

    # Force garbage collection
    print(gc.collect())  # if it's done something you should see a number being outputted

    # In TF 2.x, GPU memory management is different
    # Use the following if you need explicit GPU memory management
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Restrict TensorFlow to only allocate as much GPU memory as needed
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)


if __name__ == "__main__":
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    setup_logging()
    logger = logging.getLogger(__name__)
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
        # Profiling parameters
        parser.add_argument("--logdir", type=Path, default=LOGS_DIR)
        parser.add_argument("--profile-batch", type=int, default=100)
        # Dataset parameters
        parser.add_argument(
            "--dataset-path",
            type=Path,
            default=INTERIM_DATA_DIR / "VisualPhish",
        )
        parser.add_argument("--reshape-size", default=[224, 224, 3])
        parser.add_argument("--phishing-test-size", default=0.4)
        parser.add_argument("--num-targets", type=int, default=155)
        parser.add_argument("--legit-imgs-num", type=int, default=9363)
        parser.add_argument("--phish-imgs-num", type=int, default=1195)
        # Model parameters
        parser.add_argument("--input-shape", default=[224, 224, 3])
        parser.add_argument("--margin", type=float, default=2.2)
        parser.add_argument("--new-conv-params", default=[5, 5, 512])
        # Training parameters
        parser.add_argument("--lr", type=float, default=2e-5)  # 0.00002
        parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR / "VisualPhish")
        parser.add_argument("--saved-model-name", type=str, default="model")  # from first training
        parser.add_argument("--new-saved-model-name", type=str, default="model2")
        parser.add_argument("--save-interval", type=int, default=2000)  # 2000
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--n-iter", type=int, default=21000)  # p1: 21000, p2: 50000
        parser.add_argument("--lr-interval", type=int, default=100)  # p1: 100, p2: 250
        # hard examples training
        parser.add_argument("--num-sets", type=int, default=75)
        parser.add_argument("--iter-per-set", type=int, default=8)
        parser.add_argument("--hard-n-iter", type=int, default=30)

        args = parser.parse_args()

        # run = wandb.init(
        #     project="VisualPhish",
        #     group="visualphishnet",
        #     config=args,
        #     tags=["jarvis", "phase-1"],
        # )
        try:
            # debug_dump_dir = args.logdir / "debug_dump"
            # debug_dump_dir.mkdir(parents=True, exist_ok=True)
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(
            #     log_dir=str(args.logdir), profile_batch=args.profile_batch
            # )
            # tf.debugging.experimental.enable_dump_debug_info(
            #     str(debug_dump_dir), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
            # )
            # options = tf.profiler.experimental.ProfilerOptions(
            #     host_tracer_level=3, python_tracer_level=1, device_tracer_level=1
            # )
            # tf.profiler.experimental.start(str(args.logdir), options=options)
            # train_phase1(run, args)
            # tf.profiler.experimental.stop()
            # run.finish()
            args.lr_interval = 250
            args.lr = 2e-5
            args.n_iter = 18_000
            run = wandb.init(
                project="VisualPhish",
                group="visualphishnet",
                notes="training params from paper"
                config=args,
                tags=["jarvis", "phase-2"],
            )
            artifact = run.use_artifact("jarcin/VisualPhish/run-t7jhr5z6-model.h5:v10", type="model")
            artifact_dir = artifact.download(args.output_dir)

            # tf.profiler.experimental.start(str(args.logdir), options=options)
            train_phase2(run, args)
            # tf.profiler.experimental.stop()
        except Exception as e:
            logger.error(e)
            tb = e.__traceback__
            while tb is not None:
                logger.error(f"File: {tb.tb_frame.f_code.co_filename}, Line: {tb.tb_lineno}")
                tb = tb.tb_next
            # run.save()
        finally:
            run.finish()
