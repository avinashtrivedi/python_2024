from hmnet import ConvTransNet, BetterConvNet, BetterConvPoolNet, SimpleConvNet, TransformerNoConv
from torchinfo import summary

if __name__ == '__main__':
	
	#print(ConvTransNet())
	summary(ConvTransNet(), input_size = (1, 5, 200))
