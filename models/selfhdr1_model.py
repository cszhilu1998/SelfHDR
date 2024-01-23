import torch
from .base_model import BaseModel
from . import networks as N
import torch.optim as optim
from util.util import range_compressor_cuda
from .ahdrnet import AHDR
from .fshdr import FSHDR
from .hdr_transformer import HDRTransformer
from .sctnet import SCTNet


class SelfHDR1Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(SelfHDR1Model, self).__init__(opt)
        
        self.opt = opt
        self.loss_names = ['StruExpan', 'StruPrese', 'Total']
        self.visual_names = ['data_out', 'data_color_label', 'data_input0', 'data_input1', 'data_input2'] 

        self.model_names = ['SelfHDR']
        self.optimizer_names = ['SelfHDR_optimizer_%s' % opt.optimizer]

        if self.opt.network == 'AHDRNet':
            selfhdr = AHDR()
            opt.chop = False

        elif self.opt.network == 'FSHDR':
            selfhdr = FSHDR()
            opt.chop = False

        elif self.opt.network == 'HDR-Transformer':
            selfhdr = HDRTransformer()
            opt.chop = True

        elif self.opt.network == 'SCTNet':
            selfhdr = SCTNet()
            opt.chop = True

        self.netSelfHDR = N.init_net(selfhdr, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.opt.isTrain:
            self.optimizer_netSelfHDR = optim.Adam(self.netSelfHDR.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.weight_decay)
            self.optimizers = [self.optimizer_netSelfHDR]
            
            self.criterionL1 = N.init_net(N.L1MuLoss(), gpu_ids=opt.gpu_ids)

    def set_input(self, input):
        self.data_input0 = input['input0'].to(self.device)
        self.data_input1 = input['input1'].to(self.device)
        self.data_input2 = input['input2'].to(self.device)
        self.data_color_label = input['label'].to(self.device)
        self.data_expo = input['expo'].to(self.device, dtype=torch.float)
        self.image_name = input['fname']
    
    def forward(self):
        if self.isTrain or (not self.isTrain and not self.opt.chop):
            self.data_out = self.netSelfHDR(self.data_input0, self.data_input1, self.data_input2)
        else:
            N, C, H, W = self.data_input0.shape
            pad_w = 8 - W%8
            pad_h = 8 - H%8
            paddings = (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2)
            new_input0 = torch.nn.ReflectionPad2d(paddings)(self.data_input0)
            new_input1 = torch.nn.ReflectionPad2d(paddings)(self.data_input1)
            new_input2 = torch.nn.ReflectionPad2d(paddings)(self.data_input2)
            out = self.netSelfHDR(new_input0, new_input1, new_input2)
            self.data_out = out[:, :, pad_h//2:pad_h//2+H, pad_w//2:pad_w//2+W]
           
    def backward(self):
        in_rgb = self.data_input1[:, 3:6, ...]
        in_mask = (in_rgb < 0.5).float()
        in_mask = (in_mask * (2 * in_rgb) + (1 - in_mask) * (- 2 * in_rgb + 2))

        diff = range_compressor_cuda(self.data_color_label) * in_mask - \
               range_compressor_cuda(self.data_input1[:, 0:3, ...]) * in_mask
        label_mask = torch.mean(torch.abs(diff), 1, keepdim=True)
        label_mask[label_mask<5/255] = 0
        label_mask[label_mask>=5/255] = 1
        label_mask = 1 - label_mask

        self.loss_StruExpan = self.criterionL1(self.data_out, self.data_color_label, label_mask).mean()  # Structure-Expansion
        self.loss_StruPrese = self.criterionL1(self.data_out, self.data_input1[:, 0:3, ...], in_mask).mean() * 4  # Structure-Preserving
        self.loss_Total = self.loss_StruExpan + self.loss_StruPrese

        self.loss_Total.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer_netSelfHDR.zero_grad()
        self.backward()
        self.optimizer_netSelfHDR.step()
