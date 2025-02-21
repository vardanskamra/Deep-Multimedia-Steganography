import torch

def loss_function(cover, cover_pred, secret, secret_pred, beta=0.75):
  cover_loss = torch.nn.functional.mse_loss(cover, cover_pred)
  secret_loss = torch.nn.functional.mse_loss(secret, secret_pred)
  return cover_loss + beta * secret_loss

def normalized_correlation(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
    return numerator / denominator