from models.resnet import ResNetBase, get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr
from models.modules.resnet_block import BasicBlock, Bottleneck, BasicBlockIN, BottleneckIN, BasicBlockLN

from MinkowskiEngine import MinkowskiReLU
import MinkowskiEngine.MinkowskiOps as me
import MinkowskiEngine as ME
import torch

import torch.nn as nn

class Res16UNetBase(ResNetBase):
  BLOCK = None
  PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  INIT_DIM = 32
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(Res16UNetBase, self).__init__(in_channels, out_channels, config, D)

  def network_initialization(self, in_channels, out_channels, config, D):
    # Setup net_metadata
    dilations = self.DILATIONS
    bn_momentum = config['bn_momentum']

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    # Output of the first conv concated to conv6
    self.inplanes = self.INIT_DIM
    self.conv0p1s1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config['conv1_kernel_size'], 1),
        stride=1,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

    self.conv1p1s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        dilation=dilations[0],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv2p2s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        dilation=dilations[1],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv3p4s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        dilation=dilations[2],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv4p8s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        dilation=dilations[3],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
        
    self.relu = MinkowskiReLU(inplace=True)

    # add a classification head here
    self.clf_conv0 = conv(
        256,
        512,
        kernel_size=3,
        stride=2,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.clf_bn0 = get_norm(self.NORM_TYPE, 512, D, bn_momentum=bn_momentum)
    self.clf_conv1 = conv(
        512,
        512,
        kernel_size=3,
        stride=2,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.clf_bn1 = get_norm(self.NORM_TYPE, 512, D, bn_momentum=bn_momentum)

    self.final_linear = nn.Conv1d(512, config['clf_num_labels'], kernel_size=1, bias=True)


  def forward(self, x):
    # downsample
    out = self.conv0p1s1(x)
    out = self.bn0(out)
    out_p1 = self.relu(out)

    out = self.conv1p1s2(out_p1)
    out = self.bn1(out)
    out = self.relu(out)
    out_b1p2 = self.block1(out)

    out = self.conv2p2s2(out_b1p2)
    out = self.bn2(out)
    out = self.relu(out)
    out_b2p4 = self.block2(out)

    out = self.conv3p4s2(out_b2p4)
    out = self.bn3(out)
    out = self.relu(out)
    out_b3p8 = self.block3(out)
    
    
    out = self.conv4p8s2(out_b3p8)
    out = self.bn4(out)
    out = self.relu(out)
    out = self.block4(out)
    # pixel_dist=16
    
    #######################
    # classification head
    #######################
    clf=self.clf_conv0(out)
    clf=self.clf_bn0(clf)
    clf=self.clf_conv1(clf)
    clf=self.clf_bn1(clf)
    clf=self.relu(clf)

    batch_size = len(clf.decomposed_coordinates)
    pooled_feats = []
    for ii in range(batch_size):
      fea = clf.features_at(ii)
      coo = clf.coordinates_at(ii)
      pooled_feats.append(fea.max(0)[0][None,:])
    pooled_feats = torch.cat(pooled_feats,0)  # [N, 512]
    pooled_feats = pooled_feats.transpose(0,1).unsqueeze(0)  #[1, 512, N]
    out = self.final_linear(pooled_feats) #[1,21,N]
    out = out.squeeze(0).transpose(0,1)

    # clf=self.clf_glob_avg(clf)
    # print('yes')
    # clf=self.clf_conv2(clf)

    
    return out


class Res16UNet14(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
  LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
  PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


# Experimentally, worse than others
class Res16UNetLN14(Res16UNet14):
  NORM_TYPE = NormType.SPARSE_LAYER_NORM
  BLOCK = BasicBlockLN


class Res16UNetTemporalBase(Res16UNetBase):
  """
  Res16UNet that can take 4D independently. No temporal convolution.
  """
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(Res16UNetTemporalBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class Res16UNetTemporal14(Res16UNet14, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal18(Res16UNet18, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal34(Res16UNet34, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal50(Res16UNet50, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal101(Res16UNet101, Res16UNetTemporalBase):
  pass


class Res16UNetTemporalIN14(Res16UNetTemporal14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class Res16UNetTemporalIN18(Res16UNetTemporal18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class Res16UNetTemporalIN34(Res16UNetTemporal34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class Res16UNetTemporalIN50(Res16UNetTemporal50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class Res16UNetTemporalIN101(Res16UNetTemporal101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class STRes16UNetBase(Res16UNetBase):

  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(STRes16UNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class STRes16UNet14(STRes16UNetBase, Res16UNet14):
  pass


class STRes16UNet18(STRes16UNetBase, Res16UNet18):
  pass


class STRes16UNet34(STRes16UNetBase, Res16UNet34):
  pass


class STRes16UNet50(STRes16UNetBase, Res16UNet50):
  pass


class STRes16UNet101(STRes16UNetBase, Res16UNet101):
  pass


class STRes16UNet18A(STRes16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class STResTesseract16UNetBase(STRes16UNetBase):
  CONV_TYPE = ConvType.HYPERCUBE


class STResTesseract16UNet18A(STRes16UNet18A, STResTesseract16UNetBase):
  pass
