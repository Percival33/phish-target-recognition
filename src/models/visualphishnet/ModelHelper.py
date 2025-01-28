import logging.config
import logging
import numpy as np
from keras import optimizers, Input, Model, backend as K
from keras.applications import VGG16
from keras.layers import Conv2D, GlobalMaxPooling2D, Lambda, Subtract, Reshape
from keras.models import load_model
from keras.regularizers import l2

import DataHelper as data
from Evaluate import Evaluate
from TargetHelper import TargetHelper
from tools.config import SRC_DIR


class ModelHelper:
    def __init__(self):
        logging.config.fileConfig(SRC_DIR / 'logging.conf')
        self.logger = logging.getLogger(__name__)

    def prepare_model(self, input_shape, new_conv_params, margin, lr):
        model = self.define_triplet_network(input_shape, new_conv_params)
        model.summary(print_fn=self.logger.debug)

        optimizer = optimizers.Adam(lr=lr)
        model.compile(loss=self.custom_loss(margin), optimizer=optimizer)
        return model

    def load_trained_model(self, output_dir, saved_model_name, margin, lr):
        model = load_model(output_dir / f"{saved_model_name}.h5",
                           custom_objects={'loss': self.custom_loss(margin)})

        optimizer = optimizers.Adam(lr=lr)
        model.compile(loss=self.custom_loss(margin), optimizer=optimizer)

        return model

    def get_embeddings(self, full_model, X_train_legit, y_train_legit, all_imgs_test, all_labels_test, test_idx,
                       train_idx) -> data.TrainResults:
        shared_model = full_model.layers[3]  # FIXME: dlaczego akurat 3???
        full_model.summary(print_fn=self.logger.debug)
        whitelist_emb = shared_model.predict(X_train_legit, batch_size=64)
        phishing_emb = shared_model.predict(all_imgs_test, batch_size=64)

        self.logger.info("Embeddings were calculated")

        return data.TrainResults(
            X_legit_train=whitelist_emb,
            y_legit_train=y_train_legit,
            X_phish=phishing_emb,
            y_phish=all_labels_test,
            phish_test_idx=test_idx,
            phish_train_idx=train_idx
        )

    def get_acc(self, targetHelper: TargetHelper, VPTrainResults: data.TrainResults, trusted_list_path, phishing_path):
        # TODO: enable using wandb artifacts
        self.logger.info("Preparing to calculate acc...")
        legit_file_names_targets = targetHelper.read_file_names(trusted_list_path, 'targets.txt')
        phish_file_names_targets = targetHelper.read_file_names(phishing_path, 'targets.txt')

        phish_train_file_names, phish_test_file_names = data.get_phish_file_names(phish_file_names_targets,
                                                                                  VPTrainResults.phish_train_idx,
                                                                                  VPTrainResults.phish_test_idx)

        evaluate = Evaluate(VPTrainResults, legit_file_names_targets, phish_train_file_names)

        n = 1  # Top-1 match
        correct = 0
        self.logger.info(f"Calculating acc with top-{n} match")
        for i in range(0, VPTrainResults.phish_test_idx.shape[0]):
            distances_to_train = evaluate.pairwise_distance[i, :]
            idx, values = evaluate.find_min_distances(np.ravel(distances_to_train), n)
            names_min_distance, only_names, min_distances = evaluate.find_names_min_distances(idx, values)
            found, found_idx = targetHelper.check_if_target_in_top(str(phish_test_file_names[i].name), only_names)
            self.logger.info(names_min_distance)

            if found == 1:
                correct += 1
        self.logger.info(f"Correct match percentage: {str(correct / VPTrainResults.phish_test_idx.shape[0])}")
        return correct / VPTrainResults.phish_test_idx.shape[0]

    def save_model(self, model, output_dir, model_name):
        # TODO: save artifact to wandb
        model.save(output_dir / f'{model_name}.h5')
        self.logger.info("Saved model to disk")

    @staticmethod
    def define_triplet_network(input_shape, new_conv_params):
        """
        Input_shape: shape of input images
        new_conv_params: dimension of the new convolution layer [spatial1,spatial2,channels]
        """

        # Define the tensors for the three input images
        anchor_input = Input(input_shape)
        positive_input = Input(input_shape)
        negative_input = Input(input_shape)

        # Use VGG as a base model
        base_model = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)

        x = base_model.output
        x = Conv2D(new_conv_params[2], (new_conv_params[0], new_conv_params[1]), activation='relu',
                   kernel_initializer='he_normal', kernel_regularizer=l2(2e-4))(x)
        x = GlobalMaxPooling2D()(x)
        model = Model(inputs=base_model.input, outputs=x)

        # Generate the encodings (feature vectors) for the two images
        encoded_a = model(anchor_input)
        encoded_p = model(positive_input)
        encoded_n = model(negative_input)

        mean_layer = Lambda(lambda x: K.mean(x, axis=1))

        square_diff_layer = Lambda(lambda tensors: K.square(tensors[0] - tensors[1]))
        square_diff_pos = square_diff_layer([encoded_a, encoded_p])
        square_diff_neg = square_diff_layer([encoded_a, encoded_n])

        square_diff_pos_l2 = mean_layer(square_diff_pos)
        square_diff_neg_l2 = mean_layer(square_diff_neg)

        # Add a diff layer
        diff = Subtract()([square_diff_pos_l2, square_diff_neg_l2])
        diff = Reshape((1,))(diff)

        # Connect the inputs with the outputs
        triplet_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=diff)

        # return the model
        return triplet_net

    @staticmethod
    def custom_loss(margin):
        def loss(y_true, y_pred):
            loss_value = K.maximum(y_true, margin + y_pred)
            loss_value = K.mean(loss_value, axis=0)
            return loss_value

        return loss
