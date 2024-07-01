# Adapted from https://github.com/GKLMIP/CL-XABSA.git
import numpy as np
import torch

class ConLoss(torch.nn.Module):

    def __init__(self, temperature = 0.07, base_temperature = 0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, batch, labels): 

        device = (torch.device('cuda') if batch.is_cuda else torch.device('cpu'))

        batch_size = batch.shape[0]

        #Creates a label matrix
        labels = labels.contiguous().view(-1,1)
        labels_ = []
        for i in labels.tolist():
            if i[0] == 0:
                labels_.append([1])
            elif i[0] == 1:
                labels_.append([2])
            elif i[0] == 2:
                labels_.append([3])
                
        labels_ = torch.Tensor(labels_)

        if labels_.shape[0] != batch_size:
            raise ValueError('Num of labels does not match the size of the batch')
        mask = torch.eq(labels_, labels_.T).float().to(device)

        contrast_count = batch.shape[1]
        contrast_feature = torch.cat(torch.unbind(batch, dim=1), dim=0).to(device)

        # print("Contrast feature", contrast_feature)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability  
        logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (mask.sum(1) + 1e-20)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        torch.cuda.empty_cache()
        del mask, logits_mask, logits, exp_logits, log_prob, mean_log_prob_pos, anchor_dot_contrast, logits_max

        return loss