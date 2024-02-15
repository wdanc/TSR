import torch

def cosine2_loss(x, mode=''):
    if mode == 'bchw':
        x = x.view(x.shape[0], x.shape[1], -1)
    elif mode =='bhwc':
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.shape[0], x.shape[1], -1)
    elif mode == 'bnc':
        x = x.permute(0, 2, 1)
    elif mode == 'bcn':
        pass
    else:
        raise ValueError('unsupported dimension mode')
    # gram = x @ x.transpose(-2, -1)
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    norm = norm @ norm.transpose(-2, -1) + 1e-8
    cosine = (x @ x.transpose(-2, -1)) / norm

    cosine = torch.triu(cosine, diagonal=1)
    cosine = cosine ** 2
    cosine = cosine.mean()
    return cosine
