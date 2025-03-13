import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	simple_conv_pccs = [0.505, 0.555, 0.493, 0.396, 0.170, 0.393, 0.326, 0.278, 0.270, 0.135]
	transformer_pccs = [0.697, 0.680, 0.631, 0.591, 0.462, 0.457, 0.456, 0.424, 0.447, 0.264] #TODO
	better_conv_pccs = [0.725, 0.711, 0.679, 0.591, 0.488, 0.494, 0.473, 0.475, 0.449, 0.282] #lr=0.0003
	single_net_pccs = [0.577, 0.532, 0.526, 0.473, 0.360, 0.380, 0.357, 0.220, 0.330, 0.091] #lr=0.0003
	conv_trans = [0.663, 0.589, 0.585, 0.363, -0.084, 0.379, 0.346, 0.016, 0.310, -0.069]

	conv_trans_1enc = [0.658, 0.665, 0.600, 0.341, 0.331, 0.415, 0.391, 0.289, 0.383, 0.132]

	trans_3_layer = [0.633, 0.660, 0.614, 0.270, 0.170, 0.422, 0.389, -0.011, 0.299, 0.206]
	labels = [
        "E123-E003",
        "E116-E003",
        "E123-E116",
        "E003-E005",
        "E003-E006",
        "E006-E007",
        "E005-E006",
        "E003-E004",
        "E004-E006",
        "E037-E038"]
	pccs = transformer_pccs
	ticks = np.arange(10) + 1

	plt.figure(figsize=(8,6))
	plt.xticks(ticks, labels, rotation = 45)
	plt.plot(ticks, pccs, linestyle='dashed', marker='o', label = "1-layer Transformer")
	plt.plot(ticks, simple_conv_pccs, linestyle='dashed', marker='o', label = "Simple CNN")
	plt.plot(ticks, better_conv_pccs, linestyle='dashed', marker='o', label="6-layer CNN")
	plt.plot(ticks, single_net_pccs, linestyle='dashed', marker = 'o', label = "SingleNet")
	plt.plot(ticks, conv_trans, linestyle='dashed', marker = 'o', label = "CNN-Transformer 3 Encoders")
	plt.plot(ticks, conv_trans_1enc, linestyle='dashed', marker = 'o', label = "CNN-Transformer 1 Encoder")
	plt.plot(ticks, trans_3_layer, linestyle='dashed', marker = 'o', label = "3-layer Transformer")
	plt.ylabel("Pearson Correlation Coefficient")
	plt.xlabel("Cell Pairs")
	plt.ylim(-0.1, 1)
	plt.subplots_adjust(bottom=0.25)
	plt.title("Model Performances")
	plt.legend()
	plt.show()
	
