import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size 
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        #I haven't initiated the initial hidden states, in that case as per documentation, it will be at default 0 initially.
        #make captions also the same size as embedded features
        embed = self.embedding(captions[:,:-1])
        # Stack the features and captions
        embedded_input = torch.cat((features.unsqueeze(1), embed), dim=1) # shape :(batch_size, caption length,embed_size)
        hidden_op, (h_1, c_1) = self.lstm(embedded_input) #didn't pass any initial hidden states so its automatically zero 
        output = self.fc(hidden_op)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            hidden_output, states = self.lstm(inputs, states)
            outputs = self.fc(hidden_output.squeeze(1))
            _, predicted = outputs.max(dim=1)                    # predicted: (1, 1)
            tokens.append(predicted.item())

            inputs = self.embedding(predicted)             # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)                   # inputs: (1, 1, embed_size)
        return tokens