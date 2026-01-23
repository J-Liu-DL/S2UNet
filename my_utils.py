import torch


def poisson_noise_generator(batch_imgs, scale=1000.0, clamp_output=True):
    scaled_imgs = batch_imgs * scale

    with torch.no_grad():
        noisy_scaled = torch.poisson(scaled_imgs)

    noisy_imgs = noisy_scaled / scale

    if clamp_output:
        noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)

    return noisy_imgs
