"""
PyTorch implementation of mJNet — exact 1-to-1 match of the TensorFlow reference.

Axis convention
~~~~~~~~~~~~~~~
TensorFlow layout : (batch, H, W, T, C)   — "channels last"
PyTorch layout    : (batch, C, T, H, W)   — "channels first"

Mapping rules
~~~~~~~~~~~~~
TF  kernel_size = (3,3,1)             →  PyTorch (T=1, H=3, W=3)
TF  size_two    = (2,2,1)             →  PyTorch (T=1, H=2, W=2)
TF  conv_1 kernel (3, 3, T_full)      →  PyTorch (T_full, 3, 3) + padding='same'
TF  AveragePooling3D((1,1,T))         →  PyTorch avg_pool3d  kernel=(T,1,1)
TF  MaxPooling3D((1,1,k))             →  PyTorch max_pool3d  kernel=(k,1,1)
TF  concat  axis=3  (T axis)          →  PyTorch torch.cat   dim=2  (T axis)
TF  concat  axis=-1 (C axis)          →  PyTorch torch.cat   dim=1  (C axis)
TF  Conv3DTranspose size_two strides  →  PyTorch ConvTranspose3d stride=size_two
TF  UpSampling3D    size_two          →  PyTorch F.interpolate scale_factor=size_two

Key insight for the standard (non-longJ, non-v2) decoder path:
  • TF conv_1 with padding='same' and kernel (3,3,T) **preserves** T in the output.
  • TF AveragePooling3D((1,1,T)) then collapses T → 1.
  • TF Conv3DTranspose(strides=(2,2,1)) upsamples H,W ×2 but keeps T=1.
  • TF concatenates on axis=3 (T) → T becomes 1+1=2.
  • TF MaxPooling3D((1,1,2)) then reduces T from 2 → 1 again.
  This pattern repeats for both skip connections.  We replicate it exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Building block
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv3D → Activation → (optional) BatchNorm.

    Mirrors TF: Conv3D(activation=...) → [LeakyReLU if v2] → [BN if batch]
    Order: Conv → ReLU/LeakyReLU → BN  (same as TF reference)
    """

    def __init__(self, in_ch, out_ch, kernel_size=(1, 3, 3),
                 batch_norm=True, v2=False):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size) if isinstance(kernel_size, tuple) \
                  else kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=padding, bias=True)
        self.act = nn.LeakyReLU(0.33) if v2 else nn.ReLU()
        self.bn = nn.BatchNorm3d(out_ch) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# ---------------------------------------------------------------------------
#  mJNet
# ---------------------------------------------------------------------------
class mJNet(nn.Module):
    """
    Exact PyTorch port of the TensorFlow ``mJNet`` function.

    Input : (batch, 1, T, H, W)       — 1-channel 3-D CTP patch
    Output: (batch, num_classes, H, W) — per-pixel raw logits
    """

    def __init__(self,
                 num_classes: int = 1,
                 num_timepoints: int = 30,
                 batch_norm: bool = True,
                 dropout: bool = False,
                 dropout_rates: dict = None,
                 longJ: bool = False,
                 v2: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.T = num_timepoints
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.longJ = longJ
        self.v2 = v2

        # Dropout rates (TF defaults)
        if dropout_rates is None:
            dropout_rates = {
                'long.1': 0.5, 'long.2': 0.5, 'long.3': 0.5,
                '1': 0.5, '2': 0.5, '3': 0.5,
                '3.1': 0.5, '3.2': 0.5, '3.3': 0.5, '3.4': 0.5,
                '4': 0.5, '5': 0.5,
            }
        self.dr = dropout_rates

        # Temporal max-pool factors for longJ
        self.mp_long = {'long.1': 2, 'long.2': 3, 'long.3': 5}

        # Channel list — exact copy from TF reference
        if v2:
            ch = [16, 16, 32, 32, 64, 64, -1, 64, 64, 128, 128,
                  128, 128, 128, 128, 128, 128, -1, 128, 128, -1, 64, 32]
            ch = [c // 2 if c > 0 else c for c in ch]
        else:
            ch = [16, 32, 16, 32, 16, 32, 16, 32, 64, 64, 128,
                  128, 256, -1, -1, -1, -1, 128, 128, 64, 64, 32, 16]
        self.ch = ch

        # Kernel aliases  (PyTorch order: T, H, W)
        self.k = (1, 3, 3)            # TF (3,3,1)
        self.s = (1, 2, 2)            # TF (2,2,1) — pool / up-conv stride

        self._build()
        self._init_weights()

    # ------------------------------------------------------------------ helpers
    def _cb(self, ic, oc, ks=None):
        """Shortcut for ConvBlock with current settings."""
        return ConvBlock(ic, oc, kernel_size=(ks or self.k),
                         batch_norm=self.batch_norm, v2=self.v2)

    # ------------------------------------------------------------------ build
    def _build(self):
        ch = self.ch

        # ==========  Temporal collapse  ==========
        if self.longJ:
            self.conv_01_1 = self._cb(1, ch[0])
            self.conv_01_2 = self._cb(ch[0], ch[1])
            self.conv_02_1 = self._cb(ch[1], ch[2])
            self.conv_02_2 = self._cb(ch[2], ch[3])
            self.conv_03_1 = self._cb(ch[3], ch[4])
            self.conv_03_2 = self._cb(ch[4], ch[5])
            if self.dropout:
                self.drop_long1 = nn.Dropout3d(self.dr['long.1'])
            enc_in = ch[5]
        else:
            # TF: Conv3D(ch[6], kernel=(3, 3, T), padding='same')
            # padding='same' keeps all dims unchanged → output T = input T
            self.conv_1 = nn.Conv3d(
                1, ch[6],
                kernel_size=(self.T, 3, 3),
                padding='same',  # TF padding='same' — handles even/odd T correctly
                bias=True,       # TF Conv3D always has bias
            )
            if self.batch_norm:
                self.conv_1_bn = nn.BatchNorm3d(ch[6])
            if self.dropout:
                self.drop_1 = nn.Dropout3d(self.dr['1'])
            enc_in = ch[6]

        # ==========  Encoder  ==========
        self.conv_2_1 = self._cb(enc_in, ch[7])
        self.conv_2_2 = self._cb(ch[7], ch[8])
        if self.dropout:
            self.drop_2 = nn.Dropout3d(self.dr['2'])

        self.conv_3_1 = self._cb(ch[8], ch[9])
        self.conv_3_2 = self._cb(ch[9], ch[10])
        if self.dropout:
            self.drop_3 = nn.Dropout3d(self.dr['3'])

        self.conv_4_1 = self._cb(ch[10], ch[11])
        self.conv_4_2 = self._cb(ch[11], ch[12])

        # ==========  V2 residual bottleneck  ==========
        if self.v2:
            if self.dropout:
                self.drop_3_1 = nn.Dropout3d(self.dr['3.1'])
                self.drop_3_2 = nn.Dropout3d(self.dr['3.2'])
                self.drop_3_3 = nn.Dropout3d(self.dr['3.3'])
                self.drop_3_4 = nn.Dropout3d(self.dr['3.4'])
            k3 = (3, 3, 3)
            self.conv_4_1_res = self._cb(ch[12], ch[13], ks=k3)
            self.conv_5_1_res = self._cb(ch[13], ch[14], ks=k3)
            self.conv_6_1_res = self._cb(ch[12] + ch[14], ch[15], ks=k3)
            self.conv_7_1_res = self._cb(ch[15], ch[16], ks=k3)
            in5 = ch[12] + ch[10]       # C-concat of up_02 and conv_3
        else:
            # TF: Conv3DTranspose(ch[17], size_two, strides=size_two)
            self.upconv_1 = nn.ConvTranspose3d(
                ch[12], ch[17], kernel_size=self.s, stride=self.s, padding=0)
            # TF concats upconv_1 and conv_3 on axis=3 (T dim) → T: 1+1=2
            # Both branches must have the same channel count for T-concat.
            # ch[17]=128, ch[10]=128 in the standard config.
            in5 = ch[17]               # concat on T, so C stays the same

        # ==========  Decoder stage 1  ==========
        self.conv_5_1 = self._cb(in5, ch[18])
        self.conv_5_2 = self._cb(ch[18], ch[19])

        if not self.v2:
            if self.dropout:
                self.drop_4 = nn.Dropout3d(self.dr['4'])
            self.upconv_2 = nn.ConvTranspose3d(
                ch[19], ch[20], kernel_size=self.s, stride=self.s, padding=0)
            in6 = ch[20]              # T-concat again

        if self.v2:
            in6 = (ch[12] + ch[10]) * 2   # C-concat of up_03 and addconv_2

        # ==========  Decoder stage 2  ==========
        self.conv_6_1 = self._cb(in6, ch[21])
        self.conv_6_2 = self._cb(ch[21], ch[22])

        if self.dropout and not self.v2:
            self.drop_5 = nn.Dropout3d(self.dr['5'])

        # ==========  Output head (raw logits, no activation)  ==========
        self.conv_out = nn.Conv3d(ch[22], self.num_classes, kernel_size=1)

    # --------------------------------------------------------------- weights
    def _init_weights(self):
        """Xavier-uniform (= glorot_uniform in TF) for all conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------------------------------------------- forward
    def forward(self, x):
        """
        Args
            x: (batch, 1, T, H, W)
        Returns
            (batch, num_classes, H, W)  — raw logits
        """

        # ==================  Temporal collapse  ==================
        if self.longJ:
            c = self.conv_01_1(x)
            c = self.conv_01_2(c)
            c = F.max_pool3d(c, kernel_size=(self.mp_long['long.1'], 1, 1))
            c = self.conv_02_1(c)
            c = self.conv_02_2(c)
            c = F.max_pool3d(c, kernel_size=(self.mp_long['long.2'], 1, 1))
            c = self.conv_03_1(c)
            c = self.conv_03_2(c)
            c = F.max_pool3d(c, kernel_size=(self.mp_long['long.3'], 1, 1))
            if self.dropout:
                c = self.drop_long1(c)
            pool_drop_1 = c                                  # (B, C, 1, H, W)
        else:
            # Conv with 'same' padding → output T = input T
            c = self.conv_1(x)
            c = F.leaky_relu(c, 0.33) if self.v2 else F.relu(c)
            if self.batch_norm:
                c = self.conv_1_bn(c)
            # TF: AveragePooling3D((1,1,T)) → collapses T to 1
            pool_drop_1 = F.avg_pool3d(c, kernel_size=(c.shape[2], 1, 1))
            if self.dropout:
                pool_drop_1 = self.drop_1(pool_drop_1)
            # (B, ch[6], 1, H, W)

        # ==================  Encoder block 2  ==================
        conv_2 = self.conv_2_1(pool_drop_1)
        conv_2 = self.conv_2_2(conv_2)                       # (B, ch[8], 1, H, W)
        pool_drop_2 = F.max_pool3d(conv_2, kernel_size=self.s)
        if self.dropout:
            pool_drop_2 = self.drop_2(pool_drop_2)           # (B, ch[8], 1, H/2, W/2)

        # ==================  Encoder block 3  ==================
        conv_3 = self.conv_3_1(pool_drop_2)
        conv_3 = self.conv_3_2(conv_3)                       # (B, ch[10], 1, H/2, W/2)
        pool_drop_3 = F.max_pool3d(conv_3, kernel_size=self.s)
        if self.dropout:
            pool_drop_3 = self.drop_3(pool_drop_3)           # (B, ch[10], 1, H/4, W/4)

        # ==================  Bottleneck block 4  ==================
        conv_4 = self.conv_4_1(pool_drop_3)
        conv_4 = self.conv_4_2(conv_4)                       # (B, ch[12], 1, H/4, W/4)

        # ==================  Decoder  ==================
        if self.v2:
            # --- V2 residual path (concat on CHANNELS, dim=1) ---
            pd31 = F.max_pool3d(conv_4, kernel_size=self.s)
            if self.dropout:
                pd31 = self.drop_3_1(pd31)
            c41 = self.conv_4_1_res(pd31)
            if self.dropout:
                c41 = self.drop_3_2(c41)
            c51 = self.conv_5_1_res(c41)
            add1 = pd31 + c51
            up01 = F.interpolate(add1, scale_factor=(1, 2, 2), mode='nearest')
            conc1 = torch.cat([up01, conv_4], dim=1)         # C-concat
            if self.dropout:
                conc1 = self.drop_3_3(conc1)
            c61 = self.conv_6_1_res(conc1)
            if self.dropout:
                c61 = self.drop_3_4(c61)
            c71 = self.conv_7_1_res(c61)
            add2 = conv_4 + c71
            up02 = F.interpolate(add2, scale_factor=(1, 2, 2), mode='nearest')
            up_1 = torch.cat([up02, conv_3], dim=1)          # C-concat
        else:
            # --- Standard path (concat on T, dim=2) ---
            # TF: Conv3DTranspose → concat with conv_3 on axis=3 (T)
            up_t1 = self.upconv_1(conv_4)                    # (B, ch[17], 1, H/2, W/2)
            up_1 = torch.cat([up_t1, conv_3], dim=2)         # T-concat → T=2
            # (B, ch[17], 2, H/2, W/2)

        # ==================  Decoder stage 1  ==================
        conv_5 = self.conv_5_1(up_1)
        conv_5 = self.conv_5_2(conv_5)

        if self.v2:
            addconv_5 = torch.cat([conv_5, conv_5], dim=1)
            while addconv_5.shape[1] < up_1.shape[1]:
                addconv_5 = torch.cat([addconv_5, addconv_5], dim=1)
            addconv_5 = addconv_5[:, :up_1.shape[1]]
            add3 = up_1 + addconv_5
            up03 = F.interpolate(add3, scale_factor=(1, 2, 2), mode='nearest')
            addconv_2 = torch.cat([conv_2, conv_2], dim=1)
            while addconv_2.shape[1] < up03.shape[1]:
                addconv_2 = torch.cat([addconv_2, addconv_2], dim=1)
            addconv_2 = addconv_2[:, :up03.shape[1]]
            up_2 = torch.cat([up03, addconv_2], dim=1)       # C-concat
        else:
            # TF: MaxPooling3D((1,1,2)) → pools T: 2→1
            pool_drop_4 = F.max_pool3d(conv_5, kernel_size=(2, 1, 1))
            if self.dropout:
                pool_drop_4 = self.drop_4(pool_drop_4)
            # TF: Conv3DTranspose → concat with conv_2 on axis=3 (T)
            up_t2 = self.upconv_2(pool_drop_4)               # (B, ch[20], 1, H, W)
            up_2 = torch.cat([up_t2, conv_2], dim=2)         # T-concat → T=2
            # (B, ch[20], 2, H, W)

        # ==================  Decoder stage 2  ==================
        conv_6 = self.conv_6_1(up_2)
        conv_6 = self.conv_6_2(conv_6)

        if not self.v2:
            # TF: MaxPooling3D((1,1,2)) → pools T: 2→1
            conv_6 = F.max_pool3d(conv_6, kernel_size=(2, 1, 1))
            if self.dropout:
                conv_6 = self.drop_5(conv_6)
            # (B, ch[22], 1, H, W)

        # ==================  Output  ==================
        out = self.conv_out(conv_6)                          # (B, num_classes, 1, H, W)
        out = out.squeeze(2)                                 # (B, num_classes, H, W)
        return out


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------
def create_mjnet(input_shape=(30, 16, 16),
                 num_classes=1,
                 batch_norm=True,
                 dropout=False,
                 dropout_rates=None,
                 longJ=False,
                 v2=False):
    """
    Create an mJNet model.

    Args:
        input_shape: (num_timepoints, H, W)
        num_classes: output channels (1 = binary, >1 = categorical)
        batch_norm:  use BatchNorm after every conv
        dropout:     use Dropout3d at designated locations
        dropout_rates: dict of per-layer rates
        longJ:       progressive temporal down-sampling variant
        v2:          residual-connection variant
    """
    return mJNet(
        num_classes=num_classes,
        num_timepoints=input_shape[0],
        batch_norm=batch_norm,
        dropout=dropout,
        dropout_rates=dropout_rates,
        longJ=longJ,
        v2=v2,
    )


# ---------------------------------------------------------------------------
#  Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for label, kwargs, T, HW in [
        ("Standard mJNet (16×16, T=30, 4-class)", dict(longJ=False, v2=False), 30, 16),
        ("Standard mJNet (16×16, T=40, 4-class)", dict(longJ=False, v2=False), 40, 16),
        ("LongJ   mJNet (16×16, T=30, 4-class)",  dict(longJ=True,  v2=False), 30, 16),
    ]:
        print(f"--- {label} ---")
        try:
            model = create_mjnet(
                input_shape=(T, HW, HW), num_classes=4,
                batch_norm=True, **kwargs,
            ).to(device)
            x = torch.randn(2, 1, T, HW, HW, device=device)
            y = model(x)
            nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Input:  {tuple(x.shape)}")
            print(f"  Output: {tuple(y.shape)}")
            print(f"  Params: {nparams:,}\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
