"""
model.py

Pix2Pix-style architecture for IR -> RGB colorization.

Generator: U-Net
    Input:  (B, 4, H, W)  -- thermal, NIR, SWIR1, SWIR2
    Output: (B, 3, H, W)  -- RGB, values in [-1, 1] (tanh output)

Discriminator: PatchGAN
    Input:  concatenated (IR, RGB) pair -> (B, 7, H, W)
    Output: (B, 1, H/16, W/16) -- a grid of real/fake scores, one per patch,
            rather than a single real/fake score for the whole image.
            This encourages locally realistic texture instead of just
            globally plausible colors.
"""

import torch
import torch.nn as nn


def down_block(in_ch, out_ch, norm=True):
    """Downsampling block: Conv -> (InstanceNorm) -> LeakyReLU. Halves H and W."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_ch, out_ch, dropout=False):
    """Upsampling block: ConvTranspose -> InstanceNorm -> ReLU -> (Dropout).
    Doubles H and W."""
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """U-Net generator: encoder downsamples to a bottleneck, decoder upsamples
    back up, with skip connections between matching encoder/decoder levels so
    fine spatial detail (edges, boundaries) survives the compression.

    Designed for 256x256 input tiles (8 downsampling steps: 256 -> 1).
    """

    def __init__(self, in_channels=4, out_channels=3, base_filters=64):
        super().__init__()
        f = base_filters

        # Encoder (downsampling path)
        self.enc1 = down_block(in_channels, f, norm=False)   # 256 -> 128
        self.enc2 = down_block(f, f * 2)                     # 128 -> 64
        self.enc3 = down_block(f * 2, f * 4)                 # 64 -> 32
        self.enc4 = down_block(f * 4, f * 8)                 # 32 -> 16
        self.enc5 = down_block(f * 8, f * 8)                 # 16 -> 8
        self.enc6 = down_block(f * 8, f * 8)                 # 8 -> 4
        self.enc7 = down_block(f * 8, f * 8)                 # 4 -> 2
        self.bottleneck = down_block(f * 8, f * 8, norm=False)  # 2 -> 1

        # Decoder (upsampling path), with skip connections
        self.dec7 = up_block(f * 8, f * 8, dropout=True)         # 1 -> 2
        self.dec6 = up_block(f * 8 * 2, f * 8, dropout=True)     # 2 -> 4
        self.dec5 = up_block(f * 8 * 2, f * 8, dropout=True)     # 4 -> 8
        self.dec4 = up_block(f * 8 * 2, f * 8)                   # 8 -> 16
        self.dec3 = up_block(f * 8 * 2, f * 4)                   # 16 -> 32
        self.dec2 = up_block(f * 4 * 2, f * 2)                   # 32 -> 64
        self.dec1 = up_block(f * 2 * 2, f)                       # 64 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        b = self.bottleneck(e7)

        d7 = self.dec7(b)
        d6 = self.dec6(torch.cat([d7, e7], dim=1))
        d5 = self.dec5(torch.cat([d6, e6], dim=1))
        d4 = self.dec4(torch.cat([d5, e5], dim=1))
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        out = self.final(torch.cat([d1, e1], dim=1))
        return out


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator. Takes the IR input concatenated with
    an RGB image (real or generated) and outputs a grid of real/fake scores
    -- one score per local patch of the image, rather than one score for the
    whole image. This pushes the generator toward realistic local texture."""

    def __init__(self, in_channels=4, rgb_channels=3, base_filters=64):
        super().__init__()
        f = base_filters
        total_in = in_channels + rgb_channels

        self.model = nn.Sequential(
            down_block(total_in, f, norm=False),      # 256 -> 128
            down_block(f, f * 2),                      # 128 -> 64
            down_block(f * 2, f * 4),                   # 64 -> 32
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),  # -> patch scores
        )

    def forward(self, ir, rgb):
        x = torch.cat([ir, rgb], dim=1)
        return self.model(x)


if __name__ == "__main__":
    # Dummy batch test: confirm shapes flow correctly before touching real data.
    batch_size = 2
    ir_input = torch.randn(batch_size, 4, 256, 256)
    real_rgb = torch.randn(batch_size, 3, 256, 256)

    gen = UNetGenerator(in_channels=4, out_channels=3)
    disc = PatchGANDiscriminator(in_channels=4, rgb_channels=3)

    fake_rgb = gen(ir_input)
    print("Generator output shape:", fake_rgb.shape)
    assert fake_rgb.shape == (batch_size, 3, 256, 256), "Generator output shape mismatch!"

    disc_out_real = disc(ir_input, real_rgb)
    disc_out_fake = disc(ir_input, fake_rgb)
    print("Discriminator output shape (real):", disc_out_real.shape)
    print("Discriminator output shape (fake):", disc_out_fake.shape)

    n_params_gen = sum(p.numel() for p in gen.parameters())
    n_params_disc = sum(p.numel() for p in disc.parameters())
    print(f"Generator params: {n_params_gen:,}")
    print(f"Discriminator params: {n_params_disc:,}")

    print("\nAll shape checks passed.")