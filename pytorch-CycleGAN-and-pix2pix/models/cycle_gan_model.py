import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pdb


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # define Generator networks with number of input/output image channels number of filters, model type, whether there is normilization, dropout, inititialization, etc;
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        # only need to define discriminators if training, pass in number of filters in first layer, model type, number of layers, norm, whether to use sigmoid, initialization, etc.
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        # load generator and discriminator values if continuing training
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        # if training
        if self.isTrain:

            # pooling is a buffer to store generated images (not relevant)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)  # loss for GAN
            self.criterionCycle = torch.nn.L1Loss()  # cycle loss
            self.criterionIdt = torch.nn.L1Loss()  # identity loss
            self.criterionMask = torch.nn.L1Loss()  # identity loss

            # optimizer for generators (both A and B)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            # Separate optimizers for discriminator A and B
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            # create list of optimizers and schedulers
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        # printing out network stats
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def normalize_0_1(self, tensor):
        return (tensor + 1) / 2.

    def normalize_neg_1_1(self, tensor):
        return (2. * tensor) - 1

    # specify which data is A and which is B
    # ensure to set to cuda for efficient calculation
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if self.opt.lambda_mask > 0.0 or self.opt.add_mask:
            input_A_mask = input['A_mask' if AtoB else 'B_mask']
            input_B_mask = input['B_mask' if AtoB else 'A_mask']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            if self.opt.lambda_mask > 0.0 or self.opt.add_mask:
                input_A_mask = input_A_mask.cuda(self.gpu_ids[0], async=True)
                input_B_mask = input_B_mask.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.input_B = input_B
        if self.opt.lambda_mask > 0.0 or self.opt.add_mask:
            self.input_A_mask = input_A_mask
            self.input_B_mask = input_B_mask
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if self.opt.lambda_mask > 0.0 or self.opt.add_mask:
            self.real_A_mask = Variable(self.input_A_mask[:, :3, :, :])
            self.real_B_mask = Variable(self.input_B_mask[:, :3, :, :])
            self.real_A_mask_alpha = Variable(self.input_A_mask[:, 3, :, :])
            self.real_B_mask_alpha = Variable(self.input_B_mask[:, 3, :, :])

    def test(self):

        self.forward()
        self.backward_G()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

    def backward_G(self):

        # local copies of mask and identity loss
        lambda_mask = self.opt.lambda_mask
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G should be identity if real image is fed.
            idt_A = self.netG_A(self.real_B)
            idt_B = self.netG_B(self.real_A)

            # identity loss
            if self.opt.isTrain:
                loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
                loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            else:
                loss_idt_B = 0
                loss_idt_A = 0

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A)) and D_B(G_B(B))
        fake_B = self.netG_A(self.real_A)
        fake_A = self.netG_B(self.real_B)

        # create copy of fakes for masking
        fake_B_mask = fake_B.clone() if lambda_mask > 0.0 else 0.0
        fake_A_mask = fake_A.clone() if lambda_mask > 0.0 else 0.0

        if self.opt.add_mask:

            # save copy of fake B for visualization
            self.fake_B_viz = fake_B.data
            self.fake_A_viz = fake_A.data

            # normalize into range 0,1 for pixel manipulation
            self.real_A_mask = self.normalize_0_1(self.real_A_mask)
            self.real_B_mask = self.normalize_0_1(self.real_B_mask)
            fake_B = self.normalize_0_1(fake_B)
            fake_A = self.normalize_0_1(fake_A)

            # invert alpha mask (all 0s become 1s and 1s become 0s)
            real_A_inverted = (self.real_A_mask_alpha.clone() - 1) * -1
            real_B_inverted = (self.real_B_mask_alpha.clone() - 1) * -1

            # combine real mask of person with fake backgrounds
            fake_B = self.real_A_mask + torch.mul(fake_B, real_A_inverted)
            fake_A = self.real_B_mask + torch.mul(fake_A, real_B_inverted)

            # normalize back into range -1,1 for proper visualization and gradient flow
            fake_B = self.normalize_neg_1_1(fake_B)
            fake_A = self.normalize_neg_1_1(fake_A)

        # predict real or fake
        pred_fake_A = self.netD_A(fake_B)
        pred_fake_B = self.netD_B(fake_A)

        # GAN loss
        if self.opt.isTrain:
            loss_G_A = self.criterionGAN(pred_fake_A, True)
            loss_G_B = self.criterionGAN(pred_fake_B, True)
        else:
            loss_G_A = 0
            loss_G_B = 0

        # cycle consistency
        rec_A = self.netG_B(fake_B)
        rec_B = self.netG_A(fake_A)

        # mask the image back on top during cycle step
        if self.opt.add_mask:

            # normalize into range 0,1 for pixel manipulation
            rec_A = self.normalize_0_1(rec_A)
            rec_B = self.normalize_0_1(rec_B)

            # pixel calculations
            rec_A = self.real_A_mask + torch.mul(rec_A, real_A_inverted)
            rec_B = self.real_B_mask + torch.mul(rec_B, real_B_inverted)

            # normalize back into range -1,1 for proper visualization and gradient flow
            rec_A = self.normalize_neg_1_1(rec_A)
            rec_B = self.normalize_neg_1_1(rec_B)
            self.real_A_mask = self.normalize_neg_1_1(self.real_A_mask)
            self.real_B_mask = self.normalize_neg_1_1(self.real_B_mask)

        # cycle loss
        if self.opt.isTrain:
            loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A
            loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B
        else:
            loss_cycle_A = 0
            loss_cycle_B = 0

        # mask loss
        if lambda_mask > 0.0:

            # normalize copies of fakes into range 0,1
            fake_B_mask = self.normalize_0_1(fake_B_mask)
            fake_A_mask = self.normalize_0_1(fake_A_mask)

            # mask out person in fake image using alpha channel of the real image mask
            fake_B_mask = torch.mul(fake_B_mask, self.real_A_mask_alpha)
            fake_A_mask = torch.mul(fake_A_mask, self.real_B_mask_alpha)

            # normalize masked fake copies into range -1,1 for visualization and gradient
            fake_B_mask = self.normalize_neg_1_1(fake_B_mask)
            fake_A_mask = self.normalize_neg_1_1(fake_A_mask)

            if self.opt.isTrain:
                # calculate the loss between the real and fake masks
                loss_mask_A = self.criterionMask(fake_B_mask, self.real_A_mask) * lambda_A * lambda_mask
                loss_mask_B = self.criterionMask(fake_A_mask, self.real_B_mask) * lambda_B * lambda_mask

            # store values for errors and visualization
            self.real_A_mask_data = self.real_A_mask.data
            self.real_B_mask_data = self.real_B_mask.data
            self.mask_A = fake_B_mask.data
            self.mask_B = fake_A_mask.data
            self.loss_mask_A = loss_mask_A.data[0]
            self.loss_mask_B = loss_mask_B.data[0]
        else:
            # store values for errors and visualization
            loss_mask_A = 0
            loss_mask_B = 0
            self.loss_mask_A = 0
            self.loss_mask_B = 0
            self.real_A_mask_data = self.real_A_mask.data
            self.real_B_mask_data = self.real_B_mask.data

        # combined loss
        if self.opt.isTrain:
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_mask_A + loss_mask_B
            loss_G.backward()
        else:
            loss_G = 0

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B)])
        if self.opt.lambda_identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        if self.opt.lambda_mask > 0.0:
            ret_errors['mask_A'] = self.loss_mask_A
            ret_errors['mask_B'] = self.loss_mask_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                                   ('real_B', real_B), ('fake_A', fake_A)])

        if self.opt.isTrain:
            ret_visuals['rec_A'] = rec_A
            ret_visuals['rec_B'] = rec_B
        if self.opt.isTrain and self.opt.lambda_identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        if (self.opt.lambda_mask > 0.0 or self.opt.add_mask):
            ret_visuals['mask_real_A'] = util.tensor2im(self.real_A_mask_data)
            ret_visuals['mask_real_B'] = util.tensor2im(self.real_B_mask_data)
        if self.opt.lambda_mask > 0.0:
            ret_visuals['mask_fake_B'] = util.tensor2im(self.mask_A)
            ret_visuals['mask_fake_A'] = util.tensor2im(self.mask_B)
        if self.opt.add_mask:
            ret_visuals['fake_B_noMask'] = util.tensor2im(self.fake_B_viz)
            ret_visuals['fake_A_noMask'] = util.tensor2im(self.fake_A_viz)

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
