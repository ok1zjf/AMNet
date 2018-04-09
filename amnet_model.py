__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '6.3'
__status__ = "Research"
__date__ = "30/1/2018"
__license__= "MIT License"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special.basic import bi_zeros
from torchvision import models
from torch.autograd import Variable
import numpy as np


class VGG16FC(nn.Module):
    def __init__(self):
        super(VGG16FC, self).__init__()
        model = models.vgg16(pretrained=True)
        self.core_cnn = nn.Sequential(*list(model.features.children())[:-7])  # to relu5_3`
        self.D=512
        return

    def forward(self, x):
        x = self.core_cnn(x)
        return x

class ResNet18FC(nn.Module):
    def __init__(self):
        super(ResNet18FC, self).__init__()
        self.core_cnn = models.resnet18(pretrained=True)
        self.D=256
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


class ResNet50FC(nn.Module):
    def __init__(self):
        super(ResNet50FC, self).__init__()
        self.core_cnn = models.resnet50(pretrained=True)
        self.D = 1024
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


class ResNet101FC(nn.Module):
    def __init__(self):
        super(ResNet101FC, self).__init__()
        self.core_cnn = models.resnet101(pretrained=True)
        self.D = 1024
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


#===============================================================================================

# Direct ResNet50 memorability estimation - no attention or RNN
class ResNet50FT(nn.Module):
    def __init__(self):
        super(ResNet50FT, self).__init__()
        self.core_cnn = models.resnet50(pretrained=True)
        self.avgpool = nn.AvgPool2d(7)
        expansion = 4
        self.fc = nn.Linear(512 * expansion, 1)
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        x = self.core_cnn.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        output_seq = x.unsqueeze(1)

        output = None
        alphas = None
        return output, output_seq, alphas


#===============================================================================================
class AMemNetModel(nn.Module):

    def __init__(self, core_cnn, hps, a_res = 14, a_vec_size=512):
        super(AMemNetModel, self).__init__()

        self.hps = hps
        self.use_attention = hps.use_attention
        #self.force_distribute_attention = hps.force_distribute_attention
        self.with_bn = True

        self.a_vec_size = a_vec_size    # D
        self.a_vec_num = a_res * a_res  # L

        self.seq_len = hps.seq_steps
        self.lstm_input_size = self.a_vec_size
        self.lstm_hidden_size = 1024  # H Also LSTM output
        self.lstm_layers = 1

        self.core_cnn = core_cnn

        self.inconv = nn.Conv2d(in_channels=core_cnn.D, out_channels=a_vec_size, kernel_size=(1,1), stride=1, padding=0, bias=True)
        if self.with_bn: self.bn1 = nn.BatchNorm2d(a_vec_size)


        # Layers for the h and c LSTM states
        self.hs1 = nn.Linear(in_features=self.a_vec_size, out_features=self.lstm_hidden_size)
        self.hc1 = nn.Linear(in_features=self.a_vec_size, out_features=self.lstm_hidden_size)

        # e layers
        self.e1 = nn.Linear(in_features=self.a_vec_size, out_features=self.a_vec_size, bias=False)

        # Context layers
        self.eh1 = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.a_vec_num)
        self.eh3 = nn.Linear(in_features=self.a_vec_size, out_features=1, bias=False)

        # LSTM
        self.rnn = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                        num_layers=self.lstm_layers, dropout=0.5, bidirectional=False)

        # Regression Network
        self.regnet1 = nn.Linear(in_features=self.lstm_hidden_size, out_features=512)
        self.regnet4 = nn.Linear(in_features=self.regnet1.out_features, out_features=1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.drop80 = nn.Dropout(0.80)

        if hps.torch_version_major == 0 and hps.torch_version_minor < 3:
            self.softmax = nn.Softmax()
        else:
            self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        if not self.use_attention:
            self.alpha = torch.Tensor(x.size(0), self.a_vec_num)
            self.alpha = Variable(self.alpha)
            if self.hps.use_cuda:
                self.alpha = self.alpha.cuda()

            nn.init.constant(self.alpha, 1)
            self.alpha = self.alpha / self.a_vec_num

        x = self.core_cnn(x)

        x = self.inconv(x)
        if self.with_bn: x = self.bn1(x)
        x = self.relu(x) # -> [B, D, Ly, Lx] [B, 512, 14, 14]
        x = self.drop80(x)

        a = x.view(x.size(0), self.a_vec_size, self.a_vec_num)  # [B, D, L]

        # Extract the annotation vector
        # Mean of each feature map
        af = a.mean(2) # [B, D]

        # Hidden states for the LSTM
        hs = self.hs1(af)  # [D->H]
        hs = self.tanh(hs)

        cs = self.hc1(af) # [D->H]
        cs = self.tanh(cs)

        e = a.transpose(2, 1).contiguous() # -> [B, L, D]
        e = e.view(-1, self.a_vec_size) # a=[B, L, D] -> (-> [B*L, D])
        e = self.e1(e) # [B*L, D] -> [B*L, D]
        e = self.relu(e)
        e = self.drop50(e)
        e = e.view(-1, self.a_vec_num, self.a_vec_size) # -> [B, L, D]
        e = e.transpose(2,1) # -> [B, D, L]

        # Execute the LSTM steps
        h = hs
        rnn_state = (hs.expand(self.lstm_layers, hs.size(0), hs.size(1)).contiguous(),
                     cs.expand(self.lstm_layers, cs.size(0), cs.size(1)).contiguous())

        steps = self.seq_len
        if steps == 0:
            steps = 1

        output_seq = [0] * steps
        alphas = [0] * steps

        for i in range(steps):

            if self.use_attention:

                # Dynamic part of the alpha map from the current hidden RNN state
                if 0:
                    eh = self.eh12(h)  # -> [H -> D]
                    eh = eh.view(-1, self.a_vec_size, 1) # [B, D, 1]
                    eh = e+eh # [B, D, L]  + [B, D, 1]  => adds the eh vec[D] to all positions [L] of the e tensor

                if 1:
                    eh = self.eh1(h)  # -> [H -> L]
                    eh = eh.view(-1, 1, self.a_vec_num)  # [B, 1, L]
                    eh = e+eh  # [B, D, L]  + [B, 1, L]

                eh = self.relu(eh)
                eh = self.drop50(eh)

                eh = eh.transpose(2, 1).contiguous()  # -> [B, L, D]
                eh = eh.view(-1, self.a_vec_size)  # -> [B*L, D]

                eh = self.eh3(eh)  # -> [B*L, 512] -> [B*L, 1]
                eh = eh.view(-1, self.a_vec_num)  # -> [B, L]


                alpha = self.softmax(eh) # -> [B, L]

            else:
                alpha = self.alpha

            alpha_a = alpha.view(alpha.size(0), self.a_vec_num, 1) # -> [B, L, 1]
            z = a.bmm(alpha_a) # ->[B, D, 1] scale the location feature vectors by the alpha mask and add them (matrix mul)
            # [D, L] * [L] = [D]

            z = z.view(z.size(0), self.a_vec_size)
            z = z.expand(1, z.size(0), z.size(1)) # Prepend a new, single dimension representing the sequence

            if self.seq_len == 0:
                z = z.squeeze(dim=0)
                h = self.drop50(z)

                out = self.regnet1(h)
                out = self.relu(out)
                out = self.drop50(out)
                out = self.regnet4(out)

                output_seq[0] = out
                alphas[0] = alpha.unsqueeze(1)

                break

            # Run RNN step
            self.rnn.flatten_parameters()
            h, rnn_state = self.rnn(z, rnn_state)
            h = h.squeeze(dim=0)  # remove the seqeunce dimension
            h = self.drop50(h)

            out = self.regnet1(h)
            out = self.relu(out)
            out = self.drop50(out)
            out = self.regnet4(out)

            # Store the output and the attention mask
            ind = i
            output_seq[ind] = out
            alphas[ind] = alpha.unsqueeze(1)


        output_seq = torch.cat(output_seq, 1)
        alphas = torch.cat(alphas, 1)

        output = None
        return output, output_seq, alphas


    def load_weights(self, state_dict, info=False):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # raise KeyError('unexpected key "{}" in state_dict'
                #                .format(name))
                if info:
                    print('Cannot load key "{}". It does not exist in the model state_dict. Ignoring...'.format(name))
                # print('unexpected key "{}" in state_dict. Ignoring...'.format(name))
                continue

            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                          name, own_state[name].size(), param.size()))
                raise

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))