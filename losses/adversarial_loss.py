import numpy as np

def generator_loss(discriminator, generator, z):
    generated_data = generator(z)
    loss = -np.log(discriminator(generated_data))
    return np.mean(loss)

def discriminator_loss(discriminator, real_data, generator, z):
    generated_data = generator(z)
    real_loss = -np.log(discriminator(real_data))
    fake_loss = -np.log(1 - discriminator(generated_data))
    total_loss = real_loss + fake_loss
    return np.mean(total_loss)