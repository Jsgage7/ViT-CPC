import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class InfoNCELoss(nn.Module):
    "Modified from https://github.com/rschwarz15/CPCV2-PyTorch/blob/master/cpc_models/InfoNCE_Loss.py"
    def __init__(self, opt, in_channels, out_channels):
        super().__init__()
        self.opt = opt
        self.negative_samples = self.opt['num_negative_samples']
        self.k_predictions = self.opt['num_pred_steps']
        self.device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.prediction_modules = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.k_predictions)
        )

        self.contrast_loss = ExpNLLLoss()

        if 'weight_init' in self.opt and self.opt['weight_init'] == True:
            self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d,)):
                if module in self.prediction_modules:
                    self.makeDeltaOrthogonal(module.weight, nn.init.calculate_gain("Sigmoid"))

    def makeDeltaOrthogonal(self, weights, gain):
        rows = weights.size(0)
        cols = weights.size(1)
        weights.data.fill_(0)
        dim = max(rows, cols)
        a = torch.zeros((dim, dim)).normal_(0, 1)
        orthogonal_matrix, _ = torch.qr(a)
        d = torch.diag(orthogonal_matrix, 0).sign()
        diag_size = d.size(0)
        d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
        orthogonal_matrix.mul_(d_exp)
        mid1 = weights.size(2) // 2
        mid2 = weights.size(3) // 2
        with torch.no_grad():
            weights[:, :, mid1, mid2] = orthogonal_matrix[: weights.size(0), : weights.size(1)]
            weights.mul_(gain)


    def forward(self, z, c, skip_step=1):
        """Inputs:
            - z is the encoding from the encoder net,
            - c is the context from the autoregressive net"""
        # z.shape = torch.Size([64, 512, 7, 7])
        # c.shape = torch.Size([64, 512, 7, 7])

        batch_size = z.shape[0]
        total_loss = 0

        z.to(self.device)

        for k in range(1, self.k_predictions+ 1):
            #here a convolution operation (prediction_modules) is performed on z, using weights from the kth step. The contiguous is for memory management
            patches = self.prediction_modules[k - 1].forward(z[:, :, (k + skip_step):, :]).permute(2, 3, 0, 1).contiguous() 
            patches_shuf = patches.view(patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3])

            #random indicies are polled to produce the negative samples that are used for comparison
            rand_index = torch.randint(patches_shuf.shape[0], (patches_shuf.shape[0] * self.negative_samples, 1), dtype=torch.long, device=self.device)
            rand_index = rand_index.repeat(1, patches_shuf.shape[1])
            
            #this is then shuffled again to 'randomly' order the negative samples, then reshaped for comparison
            patches_shuf = torch.gather(patches_shuf, dim=0, index=rand_index, out=None)
            patches_shuf = patches_shuf.view(patches.shape[0], patches.shape[1], patches.shape[2], self.negative_samples, patches.shape[3]).permute(0, 1, 2, 4, 3)

            context = c[:, :, : -(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)

            #the context is compared via matrix multiplication to patches, the actual surrounding patch, and patches_shuf, which is a random negative sample
            similarity_actual = torch.matmul(context, patches.unsqueeze(-1)).squeeze(-2)
            similarity_shuf = torch.matmul(context, patches_shuf).squeeze(-2)

            #the probabilities are concatenated together, and softmax is applied, so it should maximize the similary score to actual surrounding patch
            similarity = torch.cat((similarity_actual, similarity_shuf), 3)
            similarity = similarity.permute(2, 3, 0, 1)
            similarity = torch.softmax(similarity, dim=1)

            true_f = torch.zeros((batch_size, similarity.shape[-2], similarity.shape[-1]), dtype=torch.long, device=self.device)
            total_loss += self.contrast_loss(input=similarity, target=true_f)

        total_loss /= self.k_predictions

        return total_loss


class ExpNLLLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ExpNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        x = torch.log(input + 1e-11)
        return F.nll_loss(x, target, weight=self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)

# class InfoNCELoss(nn.Module):
#     def __init__(self, num_samples, temperature=0.1):
#         super(InfoNCELoss, self).__init__()
#         self.num_samples = num_samples
#         self.temperature = temperature

#     def forward(self, features, labels):
#         batch_size = features.shape[0]

#         features = F.normalize(features, p=2, dim=-1)
#         pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
#         neg_mask = torch.logical_not(pos_mask)
#         pos_indices = pos_mask.nonzero(as_tuple=False)
#         neg_indices = neg_mask.nonzero(as_tuple=False)
#         num_pos_pairs = pos_indices.shape[0]
#         num_neg_pairs = neg_indices.shape[0]

#         # Sample negative samples
#         neg_indices = neg_indices[torch.randperm(num_neg_pairs)[:self.num_samples * num_pos_pairs]]
#         neg_features = features[neg_indices[:, 1]]

#         # Compute the dot products between positive and negative pairs
#         pos_dot_products = (features[pos_indices[:, 0]] * features[pos_indices[:, 1]]).sum(dim=-1)
#         neg_dot_products = (features.repeat(self.num_samples, 1)[neg_indices[:, 0]] * neg_features).sum(dim=-1)
#         print(len(pos_dot_products))
#         print(len(neg_dot_products))

#         # Compute the InfoNCE loss
#         logits = torch.cat([pos_dot_products.unsqueeze(1), neg_dot_products.view(batch_size, self.num_samples)], dim=1) / self.temperature
#         labels = torch.zeros(batch_size, dtype=torch.long)
#         loss = F.cross_entropy(logits, labels)

#         return loss
# class Model(nn.Module):
#     def __init__(self, num_features):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(num_features, num_features)

#     def forward(self, x):
#         x = self.fc(x)
#         return x
    
# if __name__ == '__main__':
#     #Testing on random problem to validate that the code works: IN PROGRESS
#     num_samples = 10
#     num_features = 5

#     data = torch.randn(10, num_features)
#     labels = torch.randint(0, 5, size=(10,))

#     info_nce_loss_fn = InfoNCELoss(num_samples=num_samples)

#     # Define the model and optimizer
#     model = Model(num_features=num_features)
#     optimizer = optim.Adam(model.parameters())

#     num_epochs = 100
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         # Forward pass
#         out = model(data)

#         # Compute the loss
#         #num_classes = torch.unique(labels).size(0)
#         #TODO, keep erroring on this line and not sure exactly why
#         loss = info_nce_loss_fn(out, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Print the loss
#         print('Epoch:', epoch + 1, 'Loss:', loss.item())