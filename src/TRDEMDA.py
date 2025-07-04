import torch
import torch.nn as nn
import torch.nn.functional as F
from GAGCA import *
from DAFormer import DomainAdaptedTransformerEncoder
import numpy as np
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from Transfor import GTM_net

device = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result




class TASEMDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        input_dim = args.in_dim
        hidden_dim = args.hidden_dim
        latent_dim = args.latent_dim
        num_layers = args.num_layers
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        output_dim = args.output_dim
        drop_rate = args.drop_rate
        max_len = args.max_len

        self.FNN = nn.Linear(embed_dim, output_dim) 

        self.GAGCA = GAT_GCN_Model(input_dim, hidden_dim, latent_dim)
        self.DAFormer = DomainAdaptedTransformerEncoder(num_layers, output_dim, num_heads, max_len=max_len)

        self.mlp_prediction = MLP(output_dim, drop_rate) 
        #self.mlp_prediction = MLP(331, 0.2)
        self.RF = RandomForestClassifier(n_estimators=50, random_state=42)
        self.LR = LogisticRegression(max_iter=10)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.nb = GaussianNB()
    def vae_loss(self, x, x_recon, mu, logvar):
        # 重构损失（Binary Cross-Entropy）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
        
        # KL散度（正则项）
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def classification_loss(self, pred, label):
        gamma = 2.0
        alpha = 0.25
        bce = F.binary_cross_entropy(pred.float(), label.float(), reduction='none')
        pt = torch.exp(-bce)
        focal_loss = alpha * (1-pt)**gamma * bce
        return focal_loss.mean()
        
    def forward(self, args, x, edge_index, rel_matrix, train_model):
       
        output1 = self.GAGCA(x, edge_index, x)
        hidden_X = self.FNN(output1)
        
        # transfor = GTM_net(args, hidden_X.unsqueeze(0))
        # outtrans = transfor( hidden_X ,edge_index, rel_matrix)
        output2 = self.DAFormer(hidden_X.unsqueeze(0))
        outputs = F.leaky_relu(outtrans)
        if train_model:
            train_features_inputs, train_lable = train_features_choose(rel_matrix, outputs)
            train_mlp_result = self.mlp_prediction(train_features_inputs)
            return train_mlp_result, train_lable
        else:
            test_features_inputs, test_lable = test_features_choose(rel_matrix, outputs)
            test_mlp_result = self.mlp_prediction(test_features_inputs)
            return test_mlp_result, test_lable
        
    #  ---------------RF--------------------
        # if train_model:
        #     train_inputs, train_labels = train_features_choose(rel_matrix, outputs)
        #     # 转换为 numpy 格式
        #     train_inputs = train_inputs.detach().cpu().numpy()
        #     train_labels = train_labels[:,0].detach().cpu().numpy()
        #     self.RF.fit(train_inputs, train_labels)
        #     train_result = self.RF.predict_proba(train_inputs)[:,1]
            
        #     return torch.tensor(train_result, requires_grad=True), torch.tensor(train_labels, requires_grad=True)
        # else:
        #     test_inputs, test_labels = test_features_choose(rel_matrix, outputs)
        #     self.RF.fit( test_inputs, test_labels)
        #     rf_preds = self.RF.predict_proba(test_inputs)[:, 1]
        #     return torch.tensor(rf_preds), test_labels
        
        #  ---------------LR--------------------

        # if train_model:
        #     train_inputs, train_labels = train_features_choose(rel_matrix, outputs)
        #     # 转换为 numpy 格式
        #     train_inputs = train_inputs.detach().cpu().numpy()
        #     train_labels = train_labels[:,0].detach().cpu().numpy()
        #     self.LR.fit(train_inputs, train_labels)
        #     train_result = self.LR.predict_proba(train_inputs)[:,1]
        #     return torch.tensor(train_result, requires_grad=True), torch.tensor(train_labels, requires_grad=True)
        # else:
        #     test_inputs, test_labels = test_features_choose(rel_matrix, outputs)
        #     lr_preds = self.LR.predict_proba(test_inputs)[:, 1]
        #     return torch.tensor(lr_preds), test_labels
        
        #  ---------------KNN--------------------
        # 

        # if train_model:
        #     train_inputs, train_labels = train_features_choose(rel_matrix, outputs)
        #     train_inputs = train_inputs.detach().cpu().numpy()
        #     train_labels = train_labels[:,0].detach().cpu().numpy()
        #     self.knn.fit(train_inputs, train_labels)
        #     train_result = self.knn.predict_proba(train_inputs)[:,1]
        
        #     return torch.tensor(train_result, requires_grad=True), torch.tensor(train_labels, requires_grad=True)
        # else:
        #     test_inputs, test_labels = test_features_choose(rel_matrix, outputs)
        #     knn_preds = self.knn.predict_proba(test_inputs)[:, 1]
        #     return torch.tensor(knn_preds), test_labels
        
        #  ---------------Naive Bayes--------------------
        # 

        # if train_model:
        #     train_inputs, train_labels = train_features_choose(rel_matrix, outputs)
        #     train_inputs = train_inputs.detach().cpu().numpy()
        #     train_labels = train_labels[:,0].detach().cpu().numpy()
        #     self.nb.fit(train_inputs, train_labels)
        #     train_result = self.nb.predict_proba(train_inputs)[:,1]
        #     return torch.tensor(train_result, requires_grad=True), torch.tensor(train_labels, requires_grad=True)
        # else:
        #     test_inputs, test_labels = test_features_choose(rel_matrix, outputs)
        #     nb_preds = self.nb.predict_proba(test_inputs)[:, 1]
        #     return torch.tensor(nb_preds), test_labels


       


