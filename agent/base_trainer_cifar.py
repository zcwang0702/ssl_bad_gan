import datetime
import json
import logging
import math
import os
import sys

import torch
import torch.optim as optim

sys.path.append('../')

import graph.loss.loss as module_loss
import graph.metric.metric as module_metric
import graph.model.model as module_arch
from dataloader.data_loaders import get_ssl_loaders
from utils.util import ensure_dir, get_instance
from utils.visualization import WriterTensorboardX


def build_optimizer(model, optim_name, config):
    optim_config = config['optimizer']
    min_lr = optim_config['args']['min_lr']
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optim, optim_config['type'][optim_name])(trainable_params, **optim_config['args'][optim_name])

    scheduler_config = config['scheduler']
    if scheduler_config['type'] == 'LambdaLR':
        lr_lambda = lambda epoch: max(min_lr,
                                      min(3. * (1. - float(epoch) / float(config['trainer']['max_epochs'])), 1.))
        scheduler = get_instance(optim.lr_scheduler, 'scheduler', config, optimizer, lr_lambda)
    else:
        scheduler = get_instance(optim.lr_scheduler, 'scheduler', config, optimizer)

    return optimizer, scheduler


def model_parallel(model, device, device_ids):
    # The module must have its parameters and buffers on device_ids[0] before running DataParallel module.
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)  # used for displaying logging and warning info

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(self.config['n_gpu'])

        # dataloader
        self.labeled_loader, self.unlabeled_loader, self.dev_loader = get_ssl_loaders(self.config)

        # build model architecture
        self.dis = module_arch.Discriminator(self.config)
        self.gen = module_arch.Generator(self.config)
        self.enc = module_arch.Encoder(self.config)

        # print('dis: ===\n', self.dis, '\ngen: ===\n', self.gen, '\nenc: ===\n', self.enc, )

        # model parallel using muti-gpu
        self.dis = model_parallel(self.dis, self.device, self.device_ids)
        self.gen = model_parallel(self.gen, self.device, self.device_ids)
        self.enc = model_parallel(self.enc, self.device, self.device_ids)

        # build optimizer, learning rate scheduler.
        self.dis_optimizer, self.dis_scheduler = build_optimizer(self.dis, 'dis', self.config)
        self.gen_optimizer, self.gen_scheduler = build_optimizer(self.gen, 'gen', self.config)
        self.enc_optimizer, self.enc_scheduler = build_optimizer(self.enc, 'enc', self.config)

        # construct dictionary of loss and evaluation metric
        self.loss_dict = {name: getattr(module_loss, name) for name in self.config['losses']}
        self.metric_dict = {name: getattr(module_metric, name) for name in self.config['metrics']}

        self.train_logger = train_logger  # used for saving logging info

        trainer_config = self.config['trainer']
        self.max_epochs = trainer_config['max_epochs']
        self.save_period = trainer_config['save_period']
        self.val_period = trainer_config['val_period']
        self.verbosity = trainer_config['verbosity']
        self.monitor = trainer_config.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = trainer_config.get('early_stop', math.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(trainer_config['save_dir'], self.config['name'], start_time)
        # setup visualization writer instance
        writer_dir = os.path.join(trainer_config['log_dir'], self.config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, trainer_config['tensorboardX'])

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """ 
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)

            return func

        # use data to init model for the first time
        images = []
        for i in range(int(500 / self.config['data_loader']['args']['train_batch_size'])):
            lab_images, _ = self.labeled_loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        self.gen.apply(func_gen(True))
        noise = torch.Tensor(images.size(0), self.config['model']['noise_size']).uniform_().to(self.device)
        gen_images = self.gen(noise)
        self.gen.apply(func_gen(False))

        self.dis.apply(func_gen(True))
        logits = self.dis(images.to(self.device))
        self.dis.apply(func_gen(False))

    def train(self):
        """
        Full training logic
        """
        self._param_init()

        for epoch in range(self.start_epoch, self.max_epochs + 1):

            result = self._train_epoch(epoch)
            result.update({'epoch': epoch})
            # save logged information into log dict
            if self.train_logger is not None:
                self.train_logger.add_entry(result)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if epoch % self.val_period == 0 and self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and result[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and result[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = result[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    # display best mnt metric
                    self.logger.info('Best %s: %f' % (self.mnt_metric, self.mnt_best))
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'exp_name': self.config['name'],
            'epoch': epoch,
            'logger': self.train_logger,
            'dis_state_dict': self.dis.state_dict(),
            'dis_optimizer': self.dis_optimizer.state_dict(),
            'dis_scheduler': self.dis_scheduler.state_dict(),
            'gen_state_dict': self.gen.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'gen_scheduler': self.gen_scheduler.state_dict(),
            'enc_state_dict': self.enc.state_dict(),
            'enc_optimizer': self.enc_optimizer.state_dict(),
            'enc_scheduler': self.enc_scheduler.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load model params from checkpoint.
        if checkpoint['config']['name'] != self.config['name']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        self.dis.load_state_dict(checkpoint['dis_state_dict'])
        self.gen.load_state_dict(checkpoint['gen_state_dict'])
        self.enc.load_state_dict(checkpoint['enc_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        self.enc_optimizer.load_state_dict(checkpoint['enc_optimizer'])

        # load scheduler state from checkpoint only when scheduler type is not changed
        if checkpoint['config']['scheduler']['type'] != self.config['scheduler']['type']:
            self.logger.warning('Warning: Scheduler type given in config file is different from that of checkpoint. ' + \
                                'Scheduler parameters not being resumed.')
        self.dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])
        self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
        self.enc_scheduler.load_state_dict(checkpoint['enc_scheduler'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
