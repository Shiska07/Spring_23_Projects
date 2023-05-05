import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_images(generated_images, n_rows=1, n_cols=10):
    """
    Plot the images in a 1x10 grid
    :param generated_images:
    :return:
    """
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows * n_cols):
        ax[i].imshow(generated_images[i, :, :], cmap='gray')
        ax[i].axis('off')
    return f, ax


class GenerateSamplesCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate images from the generator model at the end of each epoch
    Uses the same noise vector to generate images at each epoch, so that the images can be compared over time
    """

    def __init__(self, generator, noise):
        self.generator = generator
        self.noise = noise

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("generated_images"):
            os.mkdir("generated_images")
        generated_images = self.generator(self.noise, training=False)
        generated_images = generated_images.numpy()
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.reshape((10, 28, 28))
        # plot images using matplotlib
        plot_images(generated_images)
        plt.savefig(os.path.join("generated_images", f"generated_images_{epoch}.png"))
        # close the plot to free up memory
        plt.close()


def build_discriminator():

    model = tf.keras.models.Sequential()

    # first conv2D layer
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(2, 2),
                            padding="same", input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # second conv2D layer
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # flatten and dense
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def build_generator():

    model = tf.keras.models.Sequential()

    # dense layer
    model.add(tf.keras.layers.Dense(7 * 7 * 8, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # reshape into 2D
    model.add(tf.keras.layers.Reshape((7, 7, 8)))

    model.add(tf.keras.layers.Conv2DTranspose(8, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # output is final image
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                     activation='tanh'))

    return model


class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn


    # calculates loss for the discriminator
    def get_discriminator_loss(self, desc_output_rl, desc_output_gen,):

        # concatenate output for real and fake images
        total_out = tf.concat([desc_output_rl, desc_output_gen], axis=0)

        # concatenate desired output for real(1) and fake(0) images
        target_out = tf.concat([tf.ones_like(desc_output_rl),
                                tf.zeros_like(desc_output_gen)], axis=0)

        # get total loss
        desc_loss = self.loss_fn(target_out, total_out)

        return desc_loss

    # calculates loss for the discriminator
    def get_generator_loss(self, desc_output_gen):

        # the generator wants output of the discriminator for fake images to be 1
        gen_loss = self.loss_fn(tf.ones_like(desc_output_gen), desc_output_gen)

        return gen_loss

    def train_step(self, data):

        batch_size = tf.shape(data)[0]

        # generates noise inputs randomly from the uniform distribution
        noise = tf.random.uniform([batch_size, 100])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_images = self.generator(noise, training=True)

            desc_output_rl = self.discriminator(data, training=True)
            desc_output_gen = self.discriminator(generated_images, training=True)

            g_loss = self.get_generator_loss(desc_output_gen)
            d_loss = self.get_discriminator_loss(desc_output_rl, desc_output_gen)

        gradients_of_generator = g_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}


def train_dcgan_mnist():
    tf.keras.utils.set_random_seed(5368)
    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # the images are in the range [0, 255], we need to rescale them to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = x_train[..., tf.newaxis].astype(np.float32)

    # plot 10 random images
    example_images = x_train[:10] * 127.5 + 127.5
    plot_images(example_images)

    plt.savefig("real_images.png")

    # build the discriminator and the generator
    discriminator = build_discriminator()
    generator = build_generator()

    # build the DCGAN
    dcgan = DCGAN(discriminator=discriminator, generator=generator)

    # compile the DCGAN
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [GenerateSamplesCallback(generator, tf.random.uniform([10, 100]))]
    # train the DCGAN
    dcgan.fit(x_train, epochs=50, batch_size=64, callbacks=callbacks, shuffle=True)

    # generate images
    noise = tf.random.uniform([16, 100])
    generated_images = generator(noise, training=False)
    plot_images(generated_images * 127.5 + 127.5, 4, 4)
    plt.savefig("generated_images.png")

    generator.save('generator.h5')


if __name__ == "__main__":
    train_dcgan_mnist()
