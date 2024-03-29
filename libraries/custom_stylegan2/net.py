# modified from https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import math
import random
import sys
from typing import Optional, Tuple

import kornia
import torch
from torch import nn
from torch.nn import functional as F

sys.path.append("libraries/stylegan2_pytorch")
from libraries.stylegan2_pytorch.op import FusedLeakyReLU, fused_leaky_relu
from libraries.stylegan2_pytorch.model import PixelNorm, Upsample, Blur, ModulatedConv2d, Generator


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.empty_like(image[:, :1]).normal_()

        return image + self.weight * noise


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel // groups, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel // groups * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualConv1d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True,
            bias_init=0, c=1, w=1, init="normal", lr_mul=1
    ):
        super().__init__()
        if init == "normal":
            weight = torch.randn(out_channel, in_channel // groups, kernel_size).div_(lr_mul)
        elif init == "uniform":
            weight = torch.FloatTensor(out_channel, in_channel // groups, kernel_size).uniform_(-1, 1).div_(lr_mul)
        else:
            raise ValueError()
        self.weight = nn.Parameter(weight)
        self.scale = w * c ** 0.5 / math.sqrt(in_channel / groups * kernel_size) * lr_mul

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

        else:
            self.bias = None

        self.in_channel = in_channel
        self.out_channel = out_channel

    @property
    def memory_cost(self):
        return self.out_channel

    @property
    def flops(self):
        f = 2 * self.in_channel * self.out_channel // self.groups - self.out_channel
        if self.bias is not None:
            f += self.out_channel
        return f

    def forward(self, input):
        out = F.conv1d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, w=1
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (w / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

        self.in_dim = in_dim
        self.out_dim = out_dim

    @property
    def memory_cost(self):
        return self.out_dim

    @property
    def flops(self):
        f = 2 * self.in_dim * self.out_dim - self.out_dim
        if self.bias is not None:
            f += self.out_dim
        return f

    def forward(self, input):
        if self.activation is not None:
            out = F.linear(input, self.weight * self.scale)
            assert self.bias is not None
            bias = self.bias * self.lr_mul
            out = fused_leaky_relu(out, bias)

        else:
            bias = None if self.bias is None else self.bias * self.lr_mul
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv1d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            groups=1,
            demodulate=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel // groups, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input: torch.Tensor, style: torch.Tensor):
        batch, in_channel, height = input.shape

        style = self.modulation(style).view(batch, self.groups, in_channel // self.groups, 1)
        if self.groups > 1:
            style = torch.repeat_interleave(style, self.out_channel // self.groups, dim=1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            # demod = torch.rsqrt(weight.pow(2).sum([2, 3]) + 1e-8)
            # weight = weight * demod.view(batch, self.out_channel, 1, 1)

            weight = weight.view(batch, self.out_channel, -1)
            weight = F.normalize(weight, dim=-1)

        weight = weight.view(
            batch * self.out_channel, in_channel // self.groups, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height)
        out = F.conv1d(input, weight, padding=self.padding, groups=batch * self.groups)
        _, _, height = out.shape
        out = out.view(batch, self.out_channel, height)

        return out


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4, size2=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size2))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            use_noise=True,
            conv_1d=False,
            groups=1,
    ):
        super().__init__()
        self.use_noise = use_noise

        if conv_1d:
            self.conv = ModulatedConv1d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                groups=groups,
                demodulate=demodulate,
            )
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1))
        else:
            self.conv = ModulatedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                upsample=upsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate,
            )
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.noise = NoiseInjection()
        # self.activate = ScaledLeakyReLU(0.2)
        # self.activate = FusedLeakyReLU(out_channel)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input, style, noise: Optional[torch.Tensor] = None):
        out = self.conv(input, style)
        if self.use_noise:
            out = self.noise(out, noise=noise)
        out = out + self.bias
        out = self.activate(out) * 2 ** 0.5

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1],
                 out_channel=3):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
            last_channel=3,
            crop_background=False,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim
        self.crop_background = crop_background

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        size2 = 8 if crop_background else 4
        self.input = ConstantInput(self.channels[4], size2=size2)
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, out_channel=last_channel)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, out_channel=last_channel))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        self.cropper = kornia.augmentation.RandomCrop((self.size, self.size), resample='NEAREST')

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if self.crop_background:
            if self.training:
                image = self.cropper(image)
            else:
                image = image[:, :, :, self.size // 2: self.size * 3 // 2]
        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, config, size, in_dim=3, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(in_dim, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        # minibatch standard deviation
        self.minibatch_std = config.minibatch_std
        self.stddev_group = 4
        self.stddev_feat = 1

        if self.minibatch_std:
            in_channel += 1

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input, ddp=False, world_size=1):
        out = self.convs(input)
        batch, channel, height, width = out.shape

        if self.minibatch_std:
            group = min(batch, self.stddev_group)
            stddev = out.view(
                group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
            )
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
            if ddp:
                torch.distributed.all_reduce(stddev)
                stddev /= world_size
            stddev = stddev.repeat(group, 1, height, width)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class PretrainedStyleGAN(nn.Module):
    def __init__(self):
        super(PretrainedStyleGAN, self).__init__()
        import kornia
        size = 256
        latent = 512
        n_mlp = 8
        channel_multiplier = 2
        device = "cuda"
        ckpt = "__stylegan2_pytorch/stylegan2-church-config-f.pt"
        g_ema = Generator(
            size, latent, n_mlp, channel_multiplier=channel_multiplier
        ).to(device)
        checkpoint = torch.load(ckpt)

        g_ema.load_state_dict(checkpoint["g_ema"])
        g_ema.input.input = nn.Parameter(g_ema.input.input[:, :, 1:-1].data)

        self.size = 128
        self.gen = g_ema
        self.cropper = kornia.augmentation.RandomCrop((self.size, self.size), resample='NEAREST')
        self.n_latent = g_ema.n_latent

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor], inject_index):
        z = torch.cat(z, dim=1)
        sample, _ = self.gen([z], inject_index=inject_index)
        if self.training:
            sample = self.cropper(sample)
        else:
            sample = sample[:, :, :, self.size // 2: self.size * 3 // 2]
        return sample, None
