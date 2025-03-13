import torch
import torch.nn as nn
import math

#From https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ConvTransNet(nn.Module):
	def __init__(self, use_abs = False):
		super(ConvTransNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 32, 5), #of histone mods, # of output channels, kernel size
			nn.LeakyReLU(0.2),
			nn.Conv1d(32, 64, 5),
			nn.LeakyReLU(0.2),
			nn.Conv1d(64, 5, 5),
		)

		self.pos_enc = PositionalEncoding(5)

		self.en = nn.TransformerEncoderLayer(d_model = 5, nhead = 5, batch_first = True)
		self.encoder = nn.TransformerEncoder(self.en, num_layers = 1)
		
		self.fc = nn.Linear(940, 1)

		self.use_abs = use_abs

	def forward(self, x):
		net = self.conv_net(x)
		#print(net.shape)
		#net = net.squeeze()
		#print(net.shape)
		net = torch.transpose(net, 1, 2)
		net = self.pos_enc(net)
		net = self.en(net)
		net = net.squeeze()
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		#print(net.shape)
		net = self.fc(net)

		if(self.use_abs):
			net = torch.abs(net)

		return net

class BetterConvTranNet(nn.Module):
	def __init__(self, use_abs = False):
		super(BetterConvTranNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 32, 5),
			nn.LeakyReLU(0.2),
			nn.Conv1d(32, 64, 5),
			nn.LeakyReLU(0.2),
			nn.Conv1d(64, 128, 5),
			nn.LeakyReLU(0.2),
			nn.Conv1d(128, 64, 5),
			nn.LeakyReLU(0.2),
			nn.Conv1d(64, 32, 5),
			nn.LeakyReLU(0.2),
			nn.Conv1d(32, 5, 5),
			

		)

		self.dropout = nn.Dropout(p=0.25)
		self.pos_enc = PositionalEncoding(5)

		self.en = nn.TransformerEncoderLayer(d_model = 5, nhead = 5, batch_first = True)
		self.encoder = nn.TransformerEncoder(self.en, num_layers = 5)

		self.fc = nn.Linear(880, 1)

		self.use_abs = use_abs

		
	
	def forward(self, x):
		net = self.conv_net(x)
		#net = net.squeeze(net)
		#net = net.dropout(net)
		net = torch.transpose(net, 1, 2)
		net = self.pos_enc(net)
		net = self.en(net)
		net = net.squeeze()
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		net = self.fc(net)

		if(self.use_abs):
			net = torch.abs(net)



		return net

class TransformerNoConv(nn.Module):
	def __init__(self, use_abs = False):
		super(TransformerNoConv, self).__init__()

		self.pos_enc = PositionalEncoding(d_model = 5)

		self.en = nn.TransformerEncoderLayer(
			d_model = 5, 
			nhead = 5, 
			batch_first = True, 
			dropout=0.3
		)

		self.encoder = nn.TransformerEncoder(self.en, num_layers = 1)

		self.fc = nn.Linear(1000, 1)

		self.use_abs = use_abs

	def forward(self, x):
		net = torch.transpose(x, 1, 2)
		net = self.pos_enc(net)
		#print(net)
		net = self.encoder(net)
		net = net.squeeze()
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		net = self.fc(net)

		if(self.use_abs):
			net = torch.abs(net)

		return net

class SimpleConvNet(nn.Module):
	def __init__(self, use_abs = False):
		super(SimpleConvNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 5, 5), #of histone mods, # of output channels, kernel size
			nn.Conv1d(5, 5, 5)
		)

		self.fc = nn.Linear(960, 1)

		self.use_abs = use_abs

	def forward(self, x):
		net = self.conv_net(x)
		#print(net.shape)
		net = net.squeeze()
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		#print(net.shape)
		net = self.fc(net)

		if(self.use_abs):
			net = torch.abs(net)

		return net

class BetterConvPoolNet(nn.Module):
	def __init__(self, use_abs = False):
		super(BetterConvPoolNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 32, 5), #of histone mods, # of output channels, kernel size
			nn.ReLU(),
			nn.Conv1d(32, 64, 5),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(64, 128, 5),
			nn.ReLU(),
			nn.Conv1d(128, 64, 5),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(64, 32, 5),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(32, 5, 5),
		)	

		self.dropout = nn.Dropout(p=0.25)
		
		self.fc = nn.Linear(80, 1)

		self.use_abs = use_abs

	def forward(self, x):
		net = self.conv_net(x)
		net = net.squeeze()
		net = self.dropout(net)
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		#print(net.shape)
		net = self.fc(net)

		if(self.use_abs):
			net = torch.abs(net)

		return net

class BetterConvNet(nn.Module):
	def __init__(self, use_abs = False):
		super(BetterConvNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 32, 5), #of histone mods, # of output channels, kernel size
			nn.ReLU(),
			nn.Conv1d(32, 64, 5),
			nn.ReLU(),
			nn.Conv1d(64, 128, 5),
			nn.ReLU(),
			nn.Conv1d(128, 64, 5), #of histone mods, # of output channels, kernel size
			nn.ReLU(),
			nn.Conv1d(64, 32, 5),
			nn.ReLU(),
			nn.Conv1d(32, 5, 5),
		)	

		self.dropout = nn.Dropout(p=0.25)
		
		self.fc = nn.Linear(880, 1)

		self.use_abs = use_abs

	def forward(self, x):
		net = self.conv_net(x)
		#print(net.shape)
		net = net.squeeze()
		net = self.dropout(net)
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		#print(net.shape)
		net = self.fc(net)

		if(self.use_abs):
			net = torch.abs(net)

		return net

