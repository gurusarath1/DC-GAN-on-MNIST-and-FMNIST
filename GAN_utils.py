import torch


def get_mixed_image_gradient(crit, real, fake, epsilon):

    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # take the gradient of outputs with respect to inputs.
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def get_gradient_penalty(gradient):

    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # penalty = (2norm(grad(mixed images)) - 1)^2
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


def get_gen_loss(crit_fake_pred):
    """
    Get generator's loss
    :param crit_fake_pred: Critic scores for fake images
    :return:
    """

    # loss = E[critic's Fake image scores]
    gen_loss = -1.0 * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gradient_penalty, c_lambda):
    """
    Get critic's loss
    :param crit_fake_pred: critic fake image scores
    :param crit_real_pred: critic real image scores
    :param gradient_penalty: norm of gradient of fake image (regularization term)
    :param c_lambda: hyperparameter to weight the 1-L continuity forcing
    :return:
    """

    # loss = E[critic's Fake image scores] - E[critic's Real image scores] + lambda * (2norm(grad(mixed images)) - 1)^2
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gradient_penalty

    return crit_loss
