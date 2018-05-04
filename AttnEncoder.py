import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttnEncoder(nn.Module):
	def __init__(self, pretrained_embeddings, lstm_hidden_dim, lstm_num_layers, linear1_hidden_dim):
		super(AttnEncoder, self).__init__()

		vocab_size, input_dim = pretrained_embeddings.shape
		self.embed = nn.Embedding(vocab_size, input_dim)
		self.embed.weight.data._copy(torch.from_numpy(pretrained_embeddings))
		self.embed.weight.requires_grad = False

		self.lstm_hidden_dim = lstm_hidden_dim
		self.lstm_num_layers = lstm_num_layers
		self.linear1_hidden_dim = linear1_hidden_dim

		self.lstm_i2h = nn.LSTM(input_dim, lstm_hidden_dim, lstm_num_layers)

		self.linear1 = nn.Linear(2 * lstm_hidden_dim, linear1_hidden_dim)

		self.linear2 = nn.Linear(linear1_hidden_dim, 1)

		self.linear_final = nn.Linear(2 * lstm_hidden_dim, 1)

	def forward(self, x_index):
		x = self.embed(x_index)

		batch_size, seq_len = x_index.shape
		# x.shape - (batch, length, input_dim)

		x_transpose = torch.transpose(x, 1, 0)

		hidden , (_, _) = self.lstm_i2h(x_transpose)

		# final hidden vector - dim - (batch, hidden_dim)
		h_N = hidden[-1]

		attn_energies = Variable(torch.zeros(batch_size, seq_len))

		for i in range(seq_len):
			attn_energies[i] = self.score(h_N, hidden[i])

		attn_weights = F.softmax(attn_energies, dim=1)

		context_vec = torch.bmm(attn_weights.unsqueeze(1), torch.transpose(hidden, 1, 0)).squeeze(1)

		out = F.sigmoid(self.linear_final(torch.cat((context_vec, h_N), dim=1)))

		return attn_weights, out.squeeze(1)


	def score(final_hidden, encoder_output):
		hidden_enc_out = torch.cat((final_hidden, encoder_output), dim=1)
		out_1 = F.tanh(self.linear1(hidden_enc_out))
		out_2 = self.linear2(out_1)
		return out_2.squeeze(1)



