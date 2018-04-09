__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '4.8'
__status__ = "Research"
__date__ = "23/1/2018"
__license__= "MIT License"

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import torch
from torchvision import transforms
from torch.autograd import Variable
import cv2

from lamem2 import  *
from pylogger import  *
from amnet_model import *
import amnet_model as amnet_model
from config import *
from utils import *


# ------------------------------------------------------------------------------------------

class PredictionResult():
    def __init__(self):
        self.rc = 0
        self.mse  = 0
        self.predictions = []
        self.targets = []
        self.outputs = []
        self.attention_masks = []
        self.inference_took = 0
        self.image_names = []

    def write_stdout(self):
        print(self.__str__())

        max_len=0
        for name in self.image_names:
            if len(name) > max_len: max_len = len(name)

        print('Id\tfilename'+(' '*(max_len-8+5))+'predicted\tGT')
        for i, (image, prediction, target) in enumerate(zip(self.image_names, self.predictions, self.targets)):
            print(str(i)+'\t'+image + (' '*(max_len-len(image)+5)) + str(round(prediction,3)) + '  \t' + str(round(target,3)) )

        return

    def write_csv(self, filename):
        with open(filename, 'wt') as f:
            for image, prediction, target in zip(self.image_names, self.predictions, self.targets):
                f.write(image+' '+str(prediction)+' '+str(target)+'\n')
        return


    def get_attention_maps(self, show=False):
        images = self.image_names
        att_maps = self.attention_masks

        num_images = len(att_maps)
        seq_len = self.outputs.shape[1]
        ares = int(np.sqrt(att_maps.shape[2]))

        amaps_imgs = []
        out_size = (224, 224)

        for b in range(num_images):

            # Read the source image and resize it to 224x224
            img = cv2.imread(images[b])
            img = cv2.resize(img, out_size)

            # Create an empty output image
            offset = 20
            canvas = np.zeros((224+offset*2+50, (224+offset*2)*(seq_len+1), 3), dtype=np.uint8)
            canvas[offset:224+offset, offset:224+offset,:] = img

            amaps = []

            # Get min/max pixel values across all attention maps
            att_max = 0
            att_min = 9999999
            local_norm = True
            for s in range(seq_len):
                img_alpha = att_maps[b,s]
                img_alpha = img_alpha.reshape((ares, ares))
                Min = img_alpha.min()
                Max = img_alpha.max()
                if att_max < Max: att_max = Max
                if att_min > Min: att_min = Min

            for s in range(seq_len):
                img_alpha = att_maps[b,s]
                img_alpha = img_alpha.reshape((ares, ares))

                # Normalize & convert to uint8
                if local_norm:
                    Min = img_alpha.min()
                    img_alpha -= Min
                    Max = img_alpha.max()
                    if (Max != 0):
                        img_alpha = img_alpha/Max
                else:
                    img_alpha_min = img_alpha - img_alpha.min()
                    if (att_max-att_min) > 0:
                        img_alpha_min = img_alpha_min / (att_max-att_min)
                    else:
                        print("Zero diff in alpha map!")
                    img_alpha = img_alpha_min - (img_alpha.min()-att_min)


                img_alpha = img_alpha * 255
                img_alpha = img_alpha.astype(np.uint8)

                # Scale to the source image dimensions
                heat_map_img = cv2.resize(img_alpha, out_size, interpolation=cv2.INTER_CUBIC)
                heat_map_img = cv2.applyColorMap(heat_map_img, cv2.COLORMAP_JET)

                alpha = 0.5
                beta = (1.0 - alpha)
                img_heat_map_blend = cv2.addWeighted(img, alpha, heat_map_img, beta, 0.0)
                # amaps.append(img_heat_map_blend)

                y_pos= (s+1) * (224+offset*2)
                canvas[offset:224 + offset, y_pos+offset:y_pos+224+offset, :] = img_heat_map_blend

            amaps.append(canvas)

            if show:
                cv2.imshow('Images with attention maps', canvas)
                cv2.waitKey(0)

            amaps_imgs.append(amaps)

        return amaps_imgs


    def write_attention_maps(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        att_maps = self.get_attention_maps()

        for att_map, source_image_filename in zip(att_maps, self.image_names):
            path, filename = os.path.split(source_image_filename)
            out_filename, ext = os.path.splitext(filename)
            out_filename += '_att'+ext
            out_filename = os.path.join(out_dir, out_filename)
            cv2.imwrite(out_filename, att_map[0])

        return

    def __str__(self):
        result = "Number of images: "+str(len(self.predictions))+"\n"
        result += "Spearman's Rank Correlation: "+ ('NA' if self.rc is None else str(self.rc) )+"\n"
        result += "MSE: "+('NA' if self.mse is None else str(self.mse) )+"\n"
        result += "Inference took: "+str(self.inference_took*1000000.0)+" us/per image"
        return result


class AMNet:

    def __init__(self):
        self.logger = None
        self.total_time = 0
        self.model = None
        self.lr = 0
        self.optimizer = None
        self.data_dir = 'data'
        self.show_delay = 0

        self.test_transform = None
        self.train_transform = None
        return

    def init(self, hps):
        self.hps = hps

        self.experiment_path = os.path.join(self.data_dir, hps.experiment_name)

        if hps.front_end_cnn == 'ResNet50FT':
            model = getattr(amnet_model, hps.front_end_cnn)()
        else:
            core_cnn = getattr(amnet_model, hps.front_end_cnn)()
            model = AMemNetModel(core_cnn, hps, a_res=14, a_vec_size=1024)

        rnd_seed = 12345
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        if hps.use_cuda:
            torch.cuda.set_device(hps.cuda_device)
            print("Curent CUDA device: ", torch.cuda.current_device())
            torch.cuda.manual_seed(rnd_seed)


        self.model = model
        self.init_transformations()
        self.load_checkpoint(self.hps.model_weights)
        return

    def get_experiment_path(self):
        return self.experiment_path

    def init_transformations(self):

        if self.hps.torchvision_version_major == 0 and self.hps.torchvision_version_minor < 2:
            _resize = transforms.Scale
            _rnd_resize_crop = transforms.RandomSizedCrop
        else:
            _resize = transforms.Resize
            _rnd_resize_crop = transforms.RandomResizedCrop

        self.train_transform = transforms.Compose([
            _resize([264, 264]),
            _rnd_resize_crop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.hps.img_mean, std=self.hps.img_std)
        ])

        # Test
        self.test_transform = transforms.Compose([
            _resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.hps.img_mean, std=self.hps.img_std)
        ])

        return


    def load_dataset(self, split='train_1', train=True, batch_size=512, dataset_root='datasets/lamem/',
                    drop_last=True, num_workers=8):

        dataset = LaMem2(os.path.join(dataset_root, self.hps.images_dir),
                         split_root = os.path.join(dataset_root, self.hps.splits_dir),
                         split=split,
                         transform=self.train_transform if train else self.test_transform)

        if len(dataset.data) < batch_size:
            batch_size = len(dataset.data)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last,
                                                  num_workers=num_workers)

        print("Loaded dataset:")
        print("\t", dataset.split_file)
        print("\ttrain: ", train)
        print("\tsamples: ", len(dataset.data))
        print("\tbatch size: ", batch_size)
        print("\tnum_workers: ", num_workers)
        return data_loader


    def save_checkpoint(self,  filename):
        dirs, _ = os.path.split(filename)
        os.makedirs(dirs, exist_ok=True)
        print('Saving checkpoint: ', filename)
        torch.save({'model': self.model.state_dict()}, filename)

    def load_checkpoint(self, filename):

        if filename.strip() == '':
            return False

        try:
            print('Loading checkpoint: ', filename)
            cpnt = torch.load(filename, map_location=lambda storage, loc: storage)
            self.experiment_path, filename = os.path.split(filename)
        except FileNotFoundError:
            print("Cannot open file: ", filename)
            self.model_weights_current = ''
            return False

        try:
            self.model.load_weights(cpnt['model'])
        except:
            self.model.load_state_dict(cpnt['model'])

        return True


    def postprocess(self, output, outputs):

        if self.hps.last_step_prediction:
            output = outputs[:,-1:]
        else:
            output = (outputs).sum(1)
            output = output / outputs.shape[1]

        output /= self.hps.target_scale
        output = output + self.hps.target_mean

        if self.hps.last_step_prediction:
            outputs[:] = 0
            outputs[:,-1:] = output
        else:
            outputs = (outputs / (outputs.shape[1] * self.hps.target_scale)) + self.hps.target_mean / outputs.shape[1]

        return output, outputs



    # Training
    # ================================================================
    def get_losses(self, output, outputs, alphas, target, criterion):
        batch_size = outputs.size(0)

        if self.hps.last_step_prediction:
            output = outputs[:, -1:]
        else:
            output = (outputs).sum(1)
            output = output / outputs.size(1)

        if output is not None:
            reg_loss = criterion(output, (target - self.hps.target_mean) * self.hps.target_scale)

        # ---------------------------------------
        # Attention maps loss
        # ---------------------------------------
        att_loss = None
        if alphas is not None:
            s = 1 / alphas.size(2)  # S/L
            att_loss = alphas
            att_loss = 0.2 - att_loss
            att_loss = att_loss.sum(1)  # along the sequence dimension
            att_loss = att_loss ** 2
            att_loss = att_loss.sum()  # along the locations
            att_loss = att_loss / batch_size  # - 1820
            att_loss = att_loss * self.hps.gamma

        # Get loss enforcing an ascending order of the memorabilies at each steps in the sequence
        # ---------------------------------------

        # Memorability-locations profile cost
        a = outputs * self.hps.mem_loc_w
        a = a ** 2

        mem_loc_loss = a.sum() / a.size(0)
        mem_loc_loss *= self.hps.omega

        return reg_loss, att_loss, mem_loc_loss



    def train_epoch(self, epoch, train_loader):
        params = self.hps
        self.model.train()

        # Do not fine tune the core_cnn (resnet), use it only as a features generator
        if self.hps.front_end_cnn != 'ResNet50FT':
            for p in self.model.core_cnn.parameters():
                p.requires_grad = False

        criterion = nn.MSELoss()

        if params.use_cuda:
            self.model.cuda()
            criterion = criterion.cuda()
            params.mem_loc_w = params.mem_loc_w.cuda()


        lr_ids = np.argwhere(np.array(params.lr_epochs) <= epoch)[-1][0]
        new_lr = params.lr[lr_ids]
        if new_lr != self.lr:
            print("Epoch: ", epoch, "  Setting new learning rate: ", new_lr)
            self.lr = new_lr
            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=params.l2_req)
            # optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9,   weight_decay=0.0001)


        predictions = []
        targets = []
        start_time = batch_time = time.time()
        for batch_idx, (data, target, _) in enumerate(train_loader):

            for val in target:
                targets.append(val)

            if params.use_cuda:
                data, target = data.cuda(), target.float().cuda()

            data, target = Variable(data), Variable(target)

            output, outputs, alphas = self.model(data)

            memity, _ = self.postprocess(output, outputs.cpu().data.numpy())
            for val in memity:
                predictions.append(val)

            self.optimizer.zero_grad()

            # Calculate losses for each step individually and
            # then averaged over entire batch
            reg_loss, att_loss, mem_loc_loss = self.get_losses(output, outputs, alphas, target, criterion)

            loss = reg_loss + mem_loc_loss
            if att_loss is not None:
                loss += att_loss

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                took = time.time() - batch_time
                batch_time = time.time()
                took_total = time.time() - self.total_time
                took_total_min = int(took_total // 60)
                took_total = took_total % 60

                total_samples = len(train_loader.dataset)

                print( '{:03d}:{:06.3f} - {} [{: 5d}/{: 5d} ({:.0f}%)]  \tLoss: {:.6f} ({:.6f},  {:.6f})  [{:.3f} sec]'.format(
                        took_total_min, took_total,
                        epoch, batch_idx * len(data), total_samples,
                        100. * batch_idx / len(train_loader),
                        loss.data[0], reg_loss.data[0], (att_loss.data[0] if att_loss is not None else 0), took))

            self.logger.write(train=True, epoch=epoch, epoch_samples=total_samples,
                              sample=(batch_idx * len(data)),
                              loss=loss.cpu().data.numpy()[0], lr=params.lr)

        # Finalize the training stage
        rc, mse = train_loader.dataset.getRankCorrelationWithMSE(predictions, gt=targets)
        took_total = time.time() - self.total_time
        took_total_min = int(took_total // 60)
        took_total = took_total % 60

        took_epoch = time.time() - start_time
        print("{:03d}:{:06.3f} - {} RC: {:.6f}  MSE: {:.6f}  [{:.3f} sec]".format(
            took_total_min, took_total, epoch, rc, mse, took_epoch))


        self.logger.write(train=True, epoch=epoch, epoch_samples=len(train_loader.dataset),
                          sample=(batch_idx * len(data)),
                          loss=loss.cpu().data.numpy()[0], lr=params.lr, src=rc)

        #print("--------------------------------------------------------------------")
        return


    def train(self):
        self.logger = Logger()
        self.logger.open(os.path.join(self.get_experiment_path(), 'train_log_'+str(self.hps.epoch_start) + '.csv'))
        self.load_checkpoint(os.path.join(self.get_experiment_path(), 'weights_'+str(self.hps.epoch_start) + '.pkl'))

        train_data_loader = self.load_dataset(split=self.hps.train_split, train=True, batch_size=self.hps.train_batch_size,
                                         dataset_root=self.hps.dataset_root)

        test_data_loader = self.load_dataset(split=self.hps.val_split, train=False, batch_size=self.hps.test_batch_size,
                                        dataset_root=self.hps.dataset_root)

        self.total_time = time.time()
        for epoch in range(self.hps.epoch_start + 1, self.hps.epoch_start + self.hps.epoch_max):

            print("--------------------------------------------------------------------------------------")
            self.train_epoch(epoch, train_data_loader)
            self.save_checkpoint(os.path.join(self.get_experiment_path(), 'weights_'+str(epoch) + '.pkl'))

            rc, mse, test_loss = self.eval_model(test_data_loader)
            self.logger.write(train=False, epoch=epoch, epoch_samples=None, sample=None, loss=test_loss, lr=None, src=rc, mse=mse)

        return


# Evaluation
    def eval_model(self, test_loader):

        self.model.eval()
        criterion = nn.MSELoss()

        if self.hps.use_cuda:
            self.model.cuda()
            criterion = criterion.cuda()
            self.hps.mem_loc_w = self.hps.mem_loc_w.cuda()

        test_att_loss = 0
        test_reg_loss = 0
        test_mem_loc_loss = 0

        predictions = []
        targets = []
        batches = 0
        img_inference_took_avg = 0
        for data, target, _ in test_loader:

            for val in target:
                targets.append(val)

            target = target.float()
            if self.hps.use_cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data, volatile=True), Variable(target)

            # print(batches, len(predictions))
            batch_inference_start = time.time()
            output, outputs, alphas = self.model(data)
            batch_inference_took = time.time() - batch_inference_start

            img_inference_took = batch_inference_took / data.size(0)
            img_inference_took_avg += img_inference_took

            reg_loss, att_loss, mem_loc_loss = self.get_losses(output, outputs, alphas, target, criterion)

            test_att_loss += att_loss.cpu().data.numpy()[0] if att_loss is not None else 0
            test_reg_loss += reg_loss.cpu().data.numpy()[0]
            test_mem_loc_loss += mem_loc_loss.cpu().data.numpy()[0]

            batches += 1

            memity, _ = self.postprocess(output, outputs.cpu().data.numpy())

            for val in memity:
                predictions.append(val)

        rc, mse = test_loader.dataset.getRankCorrelationWithMSE(predictions, gt=targets)

        test_reg_loss /= batches
        test_att_loss /= batches
        test_mem_loc_loss /= batches

        test_loss = test_reg_loss + test_att_loss + test_mem_loc_loss
        img_inference_took_avg /= batches

        print('\nValidation: avg_loss: {:.4f} ({:.4f},  {:.4f})    RC: {:.6f}  MSE: {:.6f}  image_inference: {:.3f} us\n'.format(
                test_loss, test_reg_loss, test_att_loss, rc, mse, img_inference_took_avg*1000000.0))

        return rc, mse, test_loss



    def eval_models(self, model_weights, splits):
        avg_rc = 0
        avg_mse = 0
        for i in range(len(splits)):
            print("----------------------------------------------------")
            if len(model_weights) == 1:
                self.load_checkpoint(model_weights[0])
            else:
                self.load_checkpoint(model_weights[i])

            test_data_loader = self.load_dataset(split=splits[i], train=False, batch_size=self.hps.test_batch_size,
                                            dataset_root=self.hps.dataset_root, drop_last=False)

            rc, mse,_ = self.eval_model(test_data_loader)
            avg_rc += rc
            avg_mse += mse

        avg_rc /= len(splits)
        avg_mse /= len(splits)
        print(" AVG RC/MSE: ", avg_rc, ' / ', avg_mse)

        print("Done")
        return

    def predict(self, test_loader):

        self.model.eval()

        if self.hps.use_cuda:
            self.model.cuda()

        pr = PredictionResult()

        predictions = []
        targets = []
        output = None
        outputs = None
        alphas = None
        img_names = []

        batches = 0
        img_inference_took = 0

        for data, target, names in test_loader:

            for val in target:
                targets.append(val)

            img_names += names

            target = target.float()
            if self.hps.use_cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data, volatile=True), Variable(target)

            # print(batches, len(predictions))
            batch_inference_start = time.time()
            output_, outputs_, alphas_ = self.model(data)
            batch_inference_took = time.time() - batch_inference_start
            img_inference_took += batch_inference_took / data.size(0)
            batches += 1

            outputs_ = outputs_.cpu().data.numpy()
            output_ = None if output_ is None else output_.cpu().data.numpy()
            alphas_ = alphas_.cpu().data.numpy()

            memity, outputs_ = self.postprocess(output_, outputs_)

            for val in memity:
                predictions.append(val)

            # Append results overl all batches
            output = output_ if output is None else np.concatenate(output, output_)
            outputs = outputs_ if outputs is None else np.concatenate(outputs, outputs_)
            alphas = alphas_ if alphas is None else np.concatenate(alphas, alphas_)

        if test_loader.dataset.valid_labels:
            rc, mse = test_loader.dataset.getRankCorrelationWithMSE(predictions, gt=targets)
        else:
            rc, mse = None, None


        pr.rc = rc
        pr.mse  = mse
        pr.image_names = img_names
        pr.predictions = predictions
        pr.targets = targets
        pr.outputs = outputs
        pr.attention_masks = alphas
        pr.inference_took = img_inference_took / batches

        return pr



    def predict_memorability(self, images_path):

        # Use the data.Dataset class to simplify preprocesing and batch generation on multicore architectures
        dataset = LaMem2(split=images_path, # Load all images if the split points to a directory otherwise expects
                         transform=self.test_transform)

        batch_size = self.hps.test_batch_size
        if len(dataset.data) < batch_size:
            batch_size = len(dataset.data)
            print("Reducing batch size from ", self.hps.test_batch_size, "to",batch_size)

        num_workers = 8
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                                  num_workers=num_workers)

        pr = self.predict(loader)
        return pr



