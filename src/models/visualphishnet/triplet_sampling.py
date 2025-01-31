import numpy as np

# TODO: refactor to HardSubsetSampling.py


def pick_first_img_idx(labels_start_end, num_targets):
	random_target = -1
	while random_target == -1:
		random_target = np.random.randint(low=0, high=num_targets)
		if labels_start_end[random_target, 0] == -1:
			random_target = -1
	return random_target


def pick_pos_img_idx(labels_start_end_train, X_train_new, img_label):
	class_idx_start_end = labels_start_end_train[img_label, :]
	same_idx = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
	img = X_train_new[same_idx, :]
	return img


def pick_neg_img(labels_start_end_train, X_train_new, anchor_idx, num_targets):
	if anchor_idx == 0:
		targets = np.arange(1, num_targets)
	elif anchor_idx == num_targets - 1:
		targets = np.arange(0, num_targets - 1)
	else:
		targets = np.concatenate([np.arange(0, anchor_idx), np.arange(anchor_idx + 1, num_targets)])
	diff_target_idx = np.random.randint(low=0, high=num_targets - 1)
	diff_target = targets[diff_target_idx]

	class_idx_start_end = labels_start_end_train[diff_target, :]
	idx_from_diff_target = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
	img = X_train_new[idx_from_diff_target, :]

	return img, diff_target


def get_batch_for_phase2(
	targetHelper,
	X_train_legit,
	X_train_new,
	labels_start_end_train,
	batch_size,
	train_fixed_set,
	num_targets,
):
	# initialize 3 empty arrays for the input image batch
	h = X_train_legit.shape[1]
	w = X_train_legit.shape[2]
	triple = [np.zeros((batch_size, h, w, 3)) for i in range(3)]

	for i in range(0, batch_size):
		img_idx_pair1 = pick_first_img_idx(labels_start_end_train, num_targets)
		triple[0][i, :, :, :] = train_fixed_set[img_idx_pair1, :]
		img_label = img_idx_pair1

		# get image for the second: positive
		triple[1][i, :, :, :] = pick_pos_img_idx(
			labels_start_end_train=labels_start_end_train,
			X_train_new=X_train_new,
			img_label=img_label,
		)

		# get image for the third: negative from legit
		img_neg, label_neg = pick_neg_img(
			labels_start_end_train=labels_start_end_train,
			X_train_new=X_train_new,
			anchor_idx=img_label,
			num_targets=num_targets,
		)
		while targetHelper.check_if_same_category(img_label, label_neg) == 1:
			img_neg, label_neg = pick_neg_img(
				labels_start_end_train=labels_start_end_train,
				X_train_new=X_train_new,
				anchor_idx=img_label,
				num_targets=num_targets,
			)

		triple[2][i, :, :, :] = img_neg

	return triple
