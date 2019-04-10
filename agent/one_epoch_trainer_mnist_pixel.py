import torch
import torch.nn.functional as F

from base_trainer_mnist_pixel import BaseTrainer
from utils.visualization import visualize_generated_img


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, resume, config, train_logger=None):
        super(Trainer, self).__init__(resume, config, train_logger)

        self.iter_per_epoch = len(self.unlabeled_loader)
        self.log_step = int(self.iter_per_epoch * config['trainer']['log_step_ratio'])  # log per #num iterations

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.dis.train()
        self.gen.train()

        self.dis_scheduler.step()
        self.gen_scheduler.step()

        self.loss_name_list = ['lab_loss', 'unl_loss', 'd_loss', 'fm_loss', 'g_loss']
        self.metric_name_list = ['average_loss', 'error_rate', 'incorrect', 'unl_acc', 'gen_acc', 'max_unl_acc',
                                 'max_gen_acc']

        for batch_idx in range(len(self.unlabeled_loader) // self.config['data_loader']['args']['dev_batch_size']):
            # train Dis
            lab_images, lab_labels = self.labeled_loader.next()
            lab_images, lab_labels = lab_images.to(self.device), lab_labels.to(self.device)

            unl_images, _ = self.unlabeled_loader.next()
            unl_images = unl_images.to(self.device)

            noise = torch.Tensor(unl_images.size(0), self.config['model']['noise_size']).uniform_().to(self.device)
            gen_images = self.gen(noise)

            lab_logits = self.dis(lab_images)
            unl_logits = self.dis(unl_images)
            gen_logits = self.dis(gen_images.detach())

            # Standard classification loss
            lab_loss = self.loss_dict['d_criterion'](lab_logits, lab_labels)

            # GAN true-fake loss: sumexp(logits) is seen as the input to the sigmoid
            # use 11-dimensional vector to represent 10-dimensional vector
            unl_logsumexp = self.loss_dict['log_sum_exp'](unl_logits)
            gen_logsumexp = self.loss_dict['log_sum_exp'](gen_logits)

            true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
            fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
            unl_loss = true_loss + fake_loss

            d_loss = lab_loss + unl_loss

            # update dis scheduler and optimizer
            self.dis_optimizer.zero_grad()
            d_loss.backward()
            self.dis_optimizer.step()

            # train Gen and Enc
            unl_images, _ = self.unlabeled_loader2.next()
            unl_images = unl_images.to(self.device)
        
            noise = torch.Tensor(unl_images.size(0), self.config['model']['noise_size']).uniform_().to(self.device)
            gen_images = self.gen(noise)

            # Feature matching loss
            unl_feat = self.dis(unl_images, feat=True)
            gen_feat = self.dis(gen_images, feat=True)
            # fm_loss = torch.mean((torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)) ** 2)
            fm_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))

            #pixelcnn loss
            noise = torch.Tensor(30, self.config['model']['noise_size']).uniform_().to(self.device)
            gen_images = self.gen(noise)
            gen_images = (gen_images - 0.5) / 0.5
            gen_images = gen_images.view(-1, 1, 28, 28)
            logits = self.pixelcnn(gen_images)
            log_probs = - self.loss_dict['discretized_mix_logistic_loss_c1'](gen_images.permute(0, 2, 3, 1),
                                                                         logits.permute(0, 2, 3, 1), sum_all=False)
            p_loss = torch.max(log_probs - self.ploss_th, torch.FloatTensor(log_probs.size()).fill_(0.0).to(self.device))
            non_zero_cnt = float((p_loss > 0).sum().data.cpu().item())
            if non_zero_cnt > 0:
                p_loss = p_loss.sum() / non_zero_cnt * self.config['trainer']['p_loss_weight']
            else:
                p_loss = 0

            # Generator loss
            g_loss = fm_loss + p_loss

            # update gen scheduler and optimizer
            self.gen_optimizer.zero_grad()
            g_loss.backward()
            self.gen_optimizer.step()

            # log inter loss
            if batch_idx % self.config['trainer']['log_write_iteration'] == 0:
                self.writer.set_step((epoch - 1) * self.iter_per_epoch + batch_idx, mode='train')

                for loss_name in self.loss_name_list:
                    loss_value = eval('%s.item()' % loss_name)
                    self.writer.add_scalars('%s' % loss_name, loss_value)

        # epoch logging
        train_average_loss, train_error_rate, train_incorrect = \
            self.metric_dict['eval_classification'](self.dis, self.gen, self.labeled_loader, self.device)

        train_unl_acc, train_gen_acc, train_max_unl_acc, train_max_gen_acc = \
            self.metric_dict['eval_true_fake'](self.dis, self.gen, self.labeled_loader, self.device, self.config, 10)

        # write epoch metric to logger and tensorboard
        log = {}
        self.writer.set_step(epoch, mode='train')
        for metric_name in self.metric_name_list:
            metric_value = eval('train_%s' % metric_name)
            log.update({'train_%s' % metric_name: metric_value})
            self.writer.add_scalars('%s' % metric_name, metric_value)

        # visualize generated images
        if epoch % self.config['trainer']['vis_period'] == 0:
            visualize_generated_img(self.gen, epoch, self.writer, self.device, self.config)

        # display epoch logging on the screen
        if self.verbosity >= 2 and epoch % self.config['trainer']['log_display_period'] == 0:
            self.logger.info('Train Epoch %d: %s' % (epoch, str(log)))

        if self.verbosity >= 2 and epoch % self.config['trainer']['val_period'] == 0:
            val_log = self._valid_epoch(epoch)
            # add val logging
            log = {**log, **val_log}

        # return log info to base_trainer.train()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        val_average_loss, val_error_rate, val_incorrect = \
            self.metric_dict['eval_classification'](self.dis, self.gen, self.dev_loader, self.device)

        val_unl_acc, val_gen_acc, val_max_unl_acc, val_max_gen_acc = \
            self.metric_dict['eval_true_fake'](self.dis, self.gen, self.dev_loader, self.device, self.config)

        self.writer.set_step(epoch, mode='val')
        val_log = {}
        for metric_name in self.metric_name_list:
            metric_value = eval('val_%s' % metric_name)
            val_log.update({'val_%s' % metric_name: metric_value})
            self.writer.add_scalars('%s' % metric_name, metric_value)

        # display epoch logging on the screen
        if self.verbosity >= 2 and epoch % self.config['trainer']['log_display_period'] == 0:
            self.logger.info('Val Epoch %d: %s' % (epoch, str(val_log)))

        return val_log
