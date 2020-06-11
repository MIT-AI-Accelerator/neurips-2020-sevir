import horovod.tensorflow as hvd
import tensorflow as tf
from losses.gan_losses import generator_loss,discriminator_loss

# train_step based on : 
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
@tf.function
def train_step(generator, generator_optimizer,
                   discriminator, discriminator_optimizer,
                   input_image, target, epoch, summary_writer):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generator takes input images instead of random noise and generates sample output
        gen_output = generator(input_image, training=True)

        # train discriminator on actual input data
        disc_real_output = discriminator(input_image+[target], training=True)
        # train discriminator on generator output
        disc_generated_output = discriminator(input_image+[gen_output], training=True)

        # calculate loss from real data and generator data
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # get gradients
        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                    generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        # apply gradients
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                    generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                        discriminator.trainable_variables))

    # summary writer is for tensorboard
    if summary_writer is not None:
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    
    return (gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss)

