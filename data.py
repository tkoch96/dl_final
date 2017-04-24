import h5py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

class Data():
	"""
	- Accessing the image data stored in hdf5 format
	- Serving data to the Wasserstein GAN model
	"""

	def __init__(self, path, latent_dim, batch_size):
		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.load_font_file(path)

	def load_font_file(self, path):
		"""
		Loads the hdf5 file stored in path
		Returns a file object
		"""
		f = h5py.File(path, 'r')
		self.fonts = f['fonts']
		fonts_shape = f['fonts'].shape
		self.num_fonts, self.num_classes = fonts_shape[0], fonts_shape[1]
		self.latent_output_size = self.num_classes + self.latent_dim

	def randomize_labels(self):
		self.font_labels = np.random.randint(low=0, high=self.num_fonts, size=self.batch_size)
		self.char_labels = np.random.randint(low=0, high=self.num_classes, size=self.batch_size)

	def serve_real(self):
		"""
		- Serve self.batch_size amount of real font data to the critic
		- Serve self.batch_size amount of labels associated with the font data
		"""
		self.randomize_labels()
		batch_labels = np.zeros((self.batch_size, self.num_classes))
		batch_labels[np.arange(self.batch_size), self.char_labels] = 1
		images = np.array([self.fonts[self.font_labels[i]][self.char_labels[i]] for i in range(self.batch_size)], dtype=np.float32)
		return np.reshape(images, (self.batch_size, -1))/255, batch_labels

	def serve_latent(self):
		"""
		- Serve batch_size amount of latent variables to the generator
		- Need to have fake as an item in one_hot vector
		"""
		row_picker = np.arange(self.batch_size)
		one_hot = np.zeros((self.batch_size, self.num_classes))
		one_hot[row_picker, self.char_labels] = 1

		latent = np.random.uniform(size=(self.batch_size, self.latent_dim))
		feed_vectors = np.concatenate((one_hot, latent), axis=1)

		#labels = np.concatenate((one_hot, np.zeros((self.batch_size, 1))), axis=1)
		return feed_vectors, one_hot

	def serve_latent_orig(self):
		"""
		"""
		latent = np.random.uniform(size=(self.batch_size, self.latent_dim))
		one_hot = np.zeros((self.batch_size, self.num_classes))
		labels = np.concatenate((one_hot, np.zeros((self.batch_size, 1))), axis=1)
		return latent, labels

	def test(self):
		image = self.fonts[0][0]
		image = np.reshape(image, (64, 64, -1))
		image = np.tile(image, (1, 1, 3))
		print(image.shape)

		plt.imshow(image)
		plt.show()
