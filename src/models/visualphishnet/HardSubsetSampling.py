import numpy as np


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
	def find_main_train(
		self,
		model,
		fixed_set,
		targets,
		X_train,
		y_train,
		X_train_new,
		y_train_new,
		X_train_legit,
		X_train_phish,
	):
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
		X_train_legit_last_layer, X_train_phish_last_layer, fixed_set_last_layer = self.predict_all_imgs(
			model,
			X_train_legit=X_train_legit,
			X_train_phish=X_train_phish,
			fixed_set=fixed_set,
		)
		pairwise_distance = self.compute_all_distances(
			fixed_set_last_layer, X_train_legit_last_layer, X_train_phish_last_layer
		)
		n = 1
		all_idx = self.find_index_for_all_set(y_train, pairwise_distance, n)
		X_train_new, y_train_new = self.find_next_training_set(
			X_train=X_train,
			y_train=y_train,
			X_train_new=X_train_new,
			y_train_new=y_train_new,
			all_idx=all_idx,
			n=n,
		)
		X_train_new, y_train_new = self.order_random_array(X_train_new, y_train_new, targets)
		labels_start_end_train = self.start_end_each_target(targets, y_train_new)
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

	@staticmethod
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

	@staticmethod
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

	# Compute L2 distance between embeddings
	@staticmethod
	def compute_distance_pair(layer1, layer2):
		diff = layer1 - layer2
		l2_diff = np.mean(diff**2)
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
		X_false_neg_idx = np.zeros(
			[
				n,
			]
		)
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
		website_random_idx = np.zeros(
			[
				num_target,
			]
		)
		for i in range(0, num_target):
			class_idx_start_end = labels_start_end_train_legit[i, :]
			website_random_idx[i] = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
		return website_random_idx

	# Get index of false positives (different-website examples with small distance) of one query image
	@staticmethod
	def find_n_false_positives(y_train, distances, n, test_label):
		count = 0
		X_false_pos_idx = np.zeros(
			[
				n,
			]
		)
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
