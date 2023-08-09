''' Simple regression head, largely adapted from LaneGCN.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionDecoder(nn.Module):
    ''' Decode the trajectories using the embeddings of actors
    '''
    def __init__(self, config) -> None:
        super(RegressionDecoder, self).__init__()
        self.config = config
        self.n_dim = 256
        self.num_modalities = self.config['num_mods'] # modalities
        self.num_preds = self.config['num_preds']     # steps

        # each branch of prediction has its own head
        preds = [
            nn.Sequential(
                LinearResidualBlock(self.n_dim, self.n_dim),
                nn.Linear(self.n_dim, 2 * self.num_preds)
            ) for _ in range(self.num_modalities)
        ]
        self.preds = nn.ModuleList(preds)
        self.single_pred = nn.Sequential(
            LinearResidualBlock(self.n_dim, self.n_dim),
            nn.Linear(self.n_dim, 2 * self.num_preds))

        # delta information is helpful for classification
        # use it to generate K different features for each branch
        self.delta_fusion = PredictionFusion(self.n_dim)
        self.classifier = nn.Sequential(
            LinearResidualBlock(self.n_dim, self.n_dim),
            nn.Linear(self.n_dim, 1)
        )
    
    def forward(self, actors, actor_idcs, actor_ctrs):
        # ========== Regression ========== #
        predictions = list()
        for i in range(self.num_modalities):
            predictions.append(self.preds[i](actors).unsqueeze(1))
        predictions = torch.cat(predictions, 1) # N * K * (steps * 2)
        predictions = predictions.view(predictions.shape[0], self.num_modalities, self.num_preds, 2)
        single_predictions = self.single_pred(actors).view(predictions.shape[0], self.num_preds, 2)

        # ========== Classification ========== #
        batch_size = len(actor_idcs)
        for i in range(batch_size):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            predictions[idcs] = predictions[idcs] + ctrs
            single_predictions[idcs] = single_predictions[idcs] + actor_ctrs[i].view(-1, 1, 2)
        
        pred_ctrs = predictions[:, :, -1].detach()
        actor_feats = self.delta_fusion(actors, torch.cat(actor_ctrs, dim=0), pred_ctrs)
        cls_scores = self.classifier(actor_feats).view(-1, self.num_modalities)
        # cls_scores = F.softmax(cls_scores, dim=1)

        # ========== Formatting for output ========== #
        cls_scores, cls_idcs = cls_scores.sort(dim=1, descending=True)
        row_idcs = torch.arange(len(cls_idcs)).long().to(cls_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, cls_idcs.size(1)).view(-1)
        cls_idcs = cls_idcs.view(-1)
        predictions = predictions[row_idcs, cls_idcs].view(cls_scores.shape[0], cls_scores.shape[1], -1, 2)
        
        result = {'confidence': [], 'prediction': [], 'single_prediction': []}
        for i in range(batch_size):
            idcs = actor_idcs[i]
            result['confidence'].append(cls_scores[idcs])
            result['prediction'].append(predictions[idcs])
            result['single_prediction'].append(single_predictions[idcs])
        return result


class LinearResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(LinearResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm1 = nn.GroupNorm(1, out_dim)
        self.norm2 = nn.GroupNorm(1, out_dim)

        if in_dim != out_dim:
            self.trans = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                nn.GroupNorm(1, out_dim)
            )
        else:
            self.trans = nn.Identity()
    
    def forward(self, x):
        feat = self.linear1(x)
        feat = F.relu(self.norm1(feat), inplace=True)
        feat = self.linear2(feat)
        feat = self.norm2(feat)

        feat = feat + self.trans(x)
        feat = F.relu(feat, inplace=True)
        return feat


class PredictionFusion(nn.Module):
    ''' Fuse the prediction information for classification
    '''
    def __init__(self, n_dim) -> None:
        super(PredictionFusion, self).__init__()
        self.n_dim = n_dim
        self.delta_embedding = nn.Sequential(
            nn.Linear(2, n_dim),
            # nn.GroupNorm(1, n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(n_dim, n_dim, bias=False),
            nn.GroupNorm(1, n_dim),
            nn.ReLU(inplace=True)
        )

        self.mapping = nn.Sequential(
            nn.Linear(2 * n_dim, n_dim, bias=False),
            nn.GroupNorm(1, n_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, actors, actor_ctrs, pred_ctrs):
        ''' Fuse the distance between actor and prediction for classification.
            Return shape : (N * K) * n_dim 
        '''
        num_modal = pred_ctrs.shape[1]
        delta = (actor_ctrs.unsqueeze(1) - pred_ctrs).view(-1, 2)
        delta_feats = self.delta_embedding(delta)
        actor_feats = actors.unsqueeze(1).repeat(1, num_modal, 1).view(-1, self.n_dim)

        actor_feats = torch.cat((delta_feats, actor_feats), dim=1)
        actor_feats = self.mapping(actor_feats)
        return actor_feats