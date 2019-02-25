import torch
import torch.nn.functional as F

from base_trainer_cifar import BaseTrainer
from utils.visualization import visualize_generated_img


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, resume, config, train_logger=None):
        super(Trainer, self).__init__(resume, config, train_logger)

        self.iter_per_epoch = int(len(self.unlabeled_loader) // self.unlabeled_loader.batch_size + 1)
        self.log_step = int(self.iter_per_epoch * config['trainer']['log_step_ratio'])  # log per #num iterations

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.dis.train()
        self.gen.train()

        self.loss_name_list = ['lab_loss', 'unl_loss', 'd_loss', 'fm_loss', 'vi_loss', 'g_loss']
        self.metric_name_list = ['average_loss', 'error_rate', 'unl_acc', 'gen_acc', 'max_unl_acc', 'max_gen_acc']

        for batch_idx, (unl_images, _) in enumerate(self.unlabeled_loader.get_iter()):

            # train Dis
            self.dis_optimizer.zero_grad()
            lab_images, lab_labels = self.labeled_loader.next()
            lab_images, lab_labels = lab_images.to(self.device), lab_labels.to(self.device)

            unl_images = unl_images.to(self.device)

            noise = torch.Tensor(unl_images.size(0), self.config['model']['noise_size']).uniform_().to(self.device)
            gen_images = self.gen(noise)

            lab_logits = self.dis(lab_images)
            unl_logits = self.dis(unl_images)
            gen_logits = self.dis(gen_images.detach())

            # Standard classification loss
            lab_loss = self.loss_dict['d_criterion'](lab_logits, lab_labels)

            # # Conditional entropy loss
            # ent_loss = self.config['trainer']['ent_weight'] * self.loss_dict['entropy'](unl_logits)

            # GAN true-fake loss: sumexp(logits) is seen as the input to the sigmoid
            # use 11-dimensional vector to represent 10-dimensional vector
            unl_logsumexp = self.loss_dict['log_sum_exp'](unl_logits)
            gen_logsumexp = self.loss_dict['log_sum_exp'](gen_logits)

            true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
            fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
            unl_loss = true_loss + fake_loss

            d_loss = lab_loss + unl_loss

            # update dis scheduler and optimizer
            self.dis_scheduler.step()
            d_loss.backward()
            self.dis_optimizer.step()

            # train Gen and Enc
            self.gen_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()
            noise = torch.Tensor(unl_images.size(0), self.config['model']['noise_size']).uniform_().to(self.device)
            gen_images = self.gen(noise)

            # Entropy loss via variational inference
            mu, log_sigma = self.enc(gen_images)
            vi_loss = self.loss_dict['gaussian_nll'](mu, log_sigma, noise)

            # Feature matching loss
            unl_feat = self.dis(unl_images, feat=True)
            gen_feat = self.dis(gen_images, feat=True)
            fm_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))

            # # Entropy loss via feature pull-away term
            # nsample = gen_feat.size(0)
            # gen_feat_norm = gen_feat / gen_feat.norm(p=2, dim=1).reshape([-1, 1]).expand_as(gen_feat)
            # cosine = torch.mm(gen_feat_norm, gen_feat_norm.t())
            # mask = (torch.ones(cosine.size()) - torch.diag(torch.ones(nsample))).to(self.device)
            # pt_loss = self.config['trainer']['pt_weight'] * torch.sum((cosine * mask) ** 2) / (nsample * (nsample - 1))

            # Generator loss
            g_loss = fm_loss + self.config['trainer']['vi_weight'] * vi_loss

            # update gen scheduler and optimizer
            self.gen_scheduler.step()
            self.enc_scheduler.step()
            g_loss.backward()
            self.gen_optimizer.step()
            self.enc_optimizer.step()

            # log inter loss
            self.writer.set_step((epoch - 1) * self.iter_per_epoch + batch_idx, mode='train')

            for loss_name in self.loss_name_list:
                loss_value = eval('%s.item()' % loss_name)
                self.writer.add_scalar('%s' % loss_name, loss_value)

                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # epoch logging
        train_average_loss, train_error_rate = \
            self.metric_dict['eval_classification'](self.dis, self.gen, self.labeled_loader, self.device)

        train_unl_acc, train_gen_acc, train_max_unl_acc, train_max_gen_acc = \
            self.metric_dict['eval_true_fake'](self.dis, self.gen, self.labeled_loader, self.device, self.config, 10)

        # write epoch metric to logger and tensorboard
        log = {}
        self.writer.set_step(epoch, mode='train')
        for metric_name in self.metric_name_list:
            metric_value = eval('train_%s' % metric_name)
            log.update({'train_%s' % metric_name: metric_value})
            self.writer.add_scalar('%s' % metric_name, metric_value)

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

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        val_average_loss, val_error_rate = \
            self.metric_dict['eval_classification'](self.dis, self.gen, self.dev_loader, self.device)

        val_unl_acc, val_gen_acc, val_max_unl_acc, val_max_gen_acc = \
            self.metric_dict['eval_true_fake'](self.dis, self.gen, self.dev_loader, self.device, self.config)

        self.writer.set_step(epoch, mode='val')
        val_log = {}
        for metric_name in self.metric_name_list:
            metric_value = eval('val_%s' % metric_name)
            val_log.update({'val_%s' % metric_name: metric_value})
            self.writer.add_scalar('%s' % metric_name, metric_value)

        # display epoch logging on the screen
        if self.verbosity >= 2 and epoch % self.config['trainer']['log_display_period'] == 0:
            self.logger.info('Val Epoch %d: %s' % (epoch, str(val_log)))

        return val_log
