from .base_model import *


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        model_config = config['model']
        print('===> Init Discriminator for {}'.format(model_config['type']))

        self.noise_size = model_config['noise_size']
        self.num_label = model_config['num_label']

        if model_config['type'] in ['svhn', 'prostate']:
            n_filter_1, n_filter_2 = 64, 128
        else:
            n_filter_1, n_filter_2 = 96, 192

        # Assume X is of size [batch x 3 x 32 x 32]
        self.core_net = nn.Sequential(

            nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) if model_config['type'] in ['svhn', 'prostate'] \
                else nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.2)),

            WN_Conv2d(3, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5) if model_config['type'] in ['svhn', 'prostate'] else nn.Dropout(0.5),

            WN_Conv2d(n_filter_1, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5) if model_config['type'] in ['svhn', 'prostate'] else nn.Dropout(0.5),

            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        self.out_net = WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)

    def forward(self, X, feat=False):
        if X.dim() == 2:
            X = X.view(X.size(0), 3, 32, 32)

        if feat:
            return self.core_net(X)
        else:
            return self.out_net(self.core_net(X))


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        model_config = config['model']

        self.noise_size = model_config['noise_size']

        if not model_config['larger_gen']:
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 4 * 4 * 512, bias=False), nn.BatchNorm1d(4 * 4 * 512), nn.ReLU(),
                Expression(lambda tensor: tensor.view(tensor.size(0), 512, 4, 4)),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128, 3, 5, 2, 2, 1, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )
        else:
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 2 * 2 * 1024, bias=False), nn.BatchNorm1d(2 * 2 * 1024), nn.ReLU(),
                Expression(lambda tensor: tensor.view(tensor.size(0), 1024, 2, 2)),
                nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128, 3, 5, 1, 2, 0, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )

    def forward(self, noise):
        output = self.core_net(noise)

        return output


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        model_config = config['model']

        self.noise_size = model_config['noise_size']
        self.image_size = model_config['image_size']

        self.core_net = nn.Sequential(
            nn.Conv2d(3, 128, 5, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2, 2, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0), 512 * 4 * 4)),
        )

        if model_config['output_params']:
            self.core_net.add_module(str(len(self.core_net._modules)),
                                     WN_Linear(4 * 4 * 512, self.noise_size * 2, train_scale=True, init_stdv=0.1))
            self.core_net.add_module(str(len(self.core_net._modules)), Expression(lambda x: torch.chunk(x, 2, 1)))
        else:
            self.core_net.add_module(str(len(self.core_net._modules)),
                                     WN_Linear(4 * 4 * 512, self.noise_size, train_scale=True, init_stdv=0.1))

    def forward(self, inputs):

        output = self.core_net(inputs)

        return output


class Generator_mnist(nn.Module):
    def __init__(self, config):
        super(Generator_mnist, self).__init__()

        model_config = config['model']

        self.noise_size = model_config['noise_size']
        self.image_size = model_config['image_size']

        self.core_net = nn.Sequential(
            nn.Linear(self.noise_size, 500, bias=False), nn.BatchNorm1d(500), nn.Softplus(),
            nn.Linear(500, 500, bias=False), nn.BatchNorm1d(500), nn.Softplus(),
            WN_Linear(500, self.image_size, train_scale=True), nn.Sigmoid()
        )

    def forward(self, noise):
        output = self.core_net(noise)

        return output


class Discriminator_mnist(nn.Module):
    def __init__(self, config):
        super(Discriminator_mnist, self).__init__()

        model_config = config['model']
        self.image_size = model_config['image_size']
        self.num_label = model_config['num_label']

        self.feat_net = nn.Sequential(
            GaussianNoise(0.3), WN_Linear(self.image_size, 1000), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear(1000, 500), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear(500, 250), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear(250, 250), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear(250, 250), nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            GaussianNoise(0.5),
            WN_Linear(250, self.num_label, train_scale=True)
        )

    def forward(self, X, feat=False):
        if X.dim() == 4:
            X = X.view(X.size(0), -1)

        if feat:
            return self.feat_net(X)
        else:
            return self.out_net(self.feat_net(X))
