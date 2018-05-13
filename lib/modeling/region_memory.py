from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python.modeling import initializers
from caffe2.python import core, workspace
from core.config import cfg

import utils.blob as blob_utils
import utils.c2 as c2_utils

# initialize the memory
def init(model):
    mem = model.create_param(
            param_name='mem_00/spatial',
            initializer=initializers.Initializer("GaussianFill", std=cfg.MEM.STD),
            shape=[cfg.MEM.C, cfg.MEM.INIT_H, cfg.MEM.INIT_W])
    return model.ResizeMemoryInit()

def init_normalizer(model):
    norm = model.create_param(
            param_name='mem_00/spatial_normalizer',
            initializer=initializers.Initializer("ConstantFill", value=1.),
            shape=[1, cfg.MEM.INIT_H, cfg.MEM.INIT_W])
    return model.ResizeNormalizerInit()

def add_loss(model, cls_score, loss_scale=1.0):
    cls_score_name = c2_utils.UnscopeGPUName(cls_score._name)
    cls_prob_name = cls_score_name.replace('cls_score','cls_prob')
    loss_cls_name = cls_score_name.replace('cls_score','loss_cls')
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        [cls_score, 'labels_int32'], 
        [cls_prob_name, loss_cls_name],
        scale=model.GetLossScale() * loss_scale
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls])
    model.AddLosses([loss_cls])
    accuracy_cls_name = cls_score_name.replace('cls_score', 'accuracy_cls')
    model.Accuracy([cls_prob_name, 'labels_int32'], accuracy_cls_name)
    model.AddMetrics(accuracy_cls_name)

    return loss_gradients, cls_prob

def _mem_roi_align(model, mem, name_scope):
    mem_name = c2_utils.UnscopeGPUName(mem._name)
    mem_crop = model.RoIFeatureTransform(
        mem_name,
        name_scope + '/mem_crop',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.MEM.CROP_SIZE,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=cfg.MEM.SCALE
    )
    return mem_crop

def _norm_roi_align(model, norm):
    norm_name = c2_utils.UnscopeGPUName(norm._name)
    norm_crop = model.RoIFeatureTransform(
        norm_name,
        'norm_crop',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.MEM.CROP_SIZE,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=cfg.MEM.SCALE
    )
    return norm_crop

def _ctx_roi_align(model, ctx, name_scope):
    ctx_name = c2_utils.UnscopeGPUName(ctx._name)
    ctx_crop = model.RoIFeatureTransform(
        ctx_name,
        name_scope + '/ctx_crop',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.MEM.CROP_SIZE,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=cfg.MEM.SCALE
    )
    return ctx_crop

def _roi_align(model, conv_feats, spatial_scales):
    conv_crop = model.RoIFeatureTransform(
        conv_feats,
        'conv_crop',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.MEM.CROP_SIZE,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scales
    )
    return conv_crop

def _affine(model, conv_crop, dim, logits, name_scope, reuse):
    init_weight = ('GaussianFill', {'std': cfg.MEM.IN_STD})
    init_bias_scale = ('ConstantFill', {'value': cfg.MEM.IN_R})
    init_bias_offset = ('ConstantFill', {'value': 0.})
    cls_pred_dim = model.num_classes

    scaler_name = name_scope + '/affine/scaler'
    offset_name = name_scope + '/affine/offset'
    if not reuse:
        scaler = model.FC(logits,
                            scaler_name,
                            cls_pred_dim,
                            dim,
                            weight_init=init_weight,
                            bias_init=init_bias_scale)

        offset = model.FC(logits,
                        offset_name,
                        cls_pred_dim,
                        dim,
                        weight_init=init_weight,
                        bias_init=init_bias_offset)
    else:
        scaler_weight_name = 'mem_01/affine/scaler_w'
        scaler_bias_name = 'mem_01/affine/scaler_b'
        scaler = model.FCShared(logits, 
                          scaler_name,
                          cls_pred_dim, 
                          dim,
                          weight=scaler_weight_name,
                          bias=scaler_bias_name)
        
        offset_weight_name = 'mem_01/affine/offset_w'
        offset_bias_name = 'mem_01/affine/offset_b'
        offset = model.FCShared(logits, 
                          offset_name,
                          cls_pred_dim, 
                          dim,
                          weight=offset_weight_name,
                          bias=offset_bias_name)

    if cfg.MEM.IN_ACT == 'tanh':
        scaler = model.Tanh(scaler, scaler)
    elif cfg.MEM.IN_ACT == 'none':
        pass
    else:
        raise NotImplementedError

    # then try to combine them together
    scaled_name = name_scope + '/affine/scaled'
    blobs_in = [c2_utils.UnscopeGPUName(conv_crop._name), scaler_name]

    blobs_out = [scaled_name]
    scaled = model.MulConvFC(blobs_in, blobs_out)
    result_name = name_scope + '/affine/result'
    blobs_in = [c2_utils.UnscopeGPUName(scaled._name), offset_name]

    blobs_out = [result_name]
    result = model.SumConvFC(blobs_in, blobs_out)
    result = model.Relu(result, result)

    if 'gpu_0' in result._name:
        model.AddSummaryHistogram(conv_crop._name)
        model.AddSummaryHistogram(logits._name)
        model.AddSummaryHistogram(scaler._name)
        model.AddSummaryHistogram(offset._name)
        model.AddSummaryHistogram(result._name)

    return result

def _input_features(model, conv_crop, dim, logits, name_scope, reuse):
    if cfg.MEM.IN == 'film':
        input_feat = _affine(model, conv_crop, dim, logits, name_scope, reuse)
    elif cfg.MEM.IN == 'mlp':
        comb_feat = _bottomtop(model, dim, conv_crop, logits, name_scope, reuse)
        input_feat = _input(model, comb_feat, name_scope, reuse)
    else:
        raise NotImplementedError

    return input_feat

def _inplace_update(model, mem_crop, input_crop, dim, name_scope, reuse):
    input_init = ('GaussianFill', {'std': cfg.MEM.U_STD})
    mem_init = ('GaussianFill', {'std': cfg.MEM.U_STD * cfg.MEM.FM_R})
    input_gate_init = ('GaussianFill', {'std': cfg.MEM.U_STD / cfg.MEM.VG_R})
    mem_gate_init = ('GaussianFill', {'std': cfg.MEM.U_STD * cfg.MEM.FM_R / cfg.MEM.VG_R})
    bias_init = ('ConstantFill', {'value': 0.})
    mconv = cfg.MEM.CONV
    mpad = (mconv - 1) // 2

    p_input_name = name_scope + '/inplace/input_p'
    p_reset_name = name_scope + '/inplace/reset_p'
    p_update_name = name_scope + '/inplace/update_p'

    m_input_name = name_scope + '/inplace/input_m'
    m_reset_name = name_scope + '/inplace/reset_m'
    m_update_name = name_scope + '/inplace/update_m'

    input_name = name_scope + '/inplace/input'
    reset_name = name_scope + '/inplace/reset'
    update_name = name_scope + '/inplace/update'

    mem_crop_name = c2_utils.UnscopeGPUName(mem_crop._name)
    mult_mem_name = name_scope + '/inplace/mult_mem'
    next_crop_raw_name = name_scope + '/next_crop_raw'
    next_crop_name = name_scope + '/next_crop'

    if not reuse:
        p_input = model.Conv(input_crop,
                            p_input_name,
                            dim,
                            cfg.MEM.C,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight_init=input_init,
                            bias_init=bias_init)
        p_reset = model.Conv(input_crop,
                            p_reset_name,
                            dim,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight_init=input_gate_init,
                            bias_init=bias_init)
        p_update = model.Conv(input_crop,
                            p_update_name,
                            dim,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight_init=input_gate_init,
                            bias_init=bias_init)
        m_reset = model.Conv(mem_crop,
                            m_reset_name,
                            cfg.MEM.C,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight_init=mem_gate_init,
                            no_bias=True)
        m_update = model.Conv(mem_crop,
                            m_update_name,
                            cfg.MEM.C,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight_init=mem_gate_init,
                            no_bias=True)
        reset = model.net.Sum([p_reset_name, m_reset_name], reset_name)
        reset = model.net.Sigmoid(reset, reset_name)
        blobs_in = [mem_crop_name, reset_name]
        blobs_out = [mult_mem_name]
        mult_mem = model.MulConvGate(blobs_in, blobs_out)
        m_input = model.Conv(mult_mem,
                            m_input_name,
                            cfg.MEM.C,
                            cfg.MEM.C,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight_init=mem_init,
                            no_bias=True)
    else:
        p_input_weight_name = 'mem_01/inplace/input_p_w'
        p_input_bias_name = 'mem_01/inplace/input_p_b'
        p_reset_weight_name = 'mem_01/inplace/reset_p_w'
        p_reset_bias_name = 'mem_01/inplace/reset_p_b'
        p_update_weight_name = 'mem_01/inplace/update_p_w'
        p_update_bias_name = 'mem_01/inplace/update_p_b'

        m_input_weight_name = 'mem_01/inplace/input_m_w'
        m_reset_weight_name = 'mem_01/inplace/reset_m_w'
        m_update_weight_name = 'mem_01/inplace/update_m_w'

        p_input = model.ConvShared(input_crop,
                            p_input_name,
                            dim,
                            cfg.MEM.C,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight=p_input_weight_name,
                            bias=p_input_bias_name)
        p_reset = model.ConvShared(input_crop,
                            p_reset_name,
                            dim,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight=p_reset_weight_name,
                            bias=p_reset_bias_name)
        p_update = model.ConvShared(input_crop,
                            p_update_name,
                            dim,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight=p_update_weight_name,
                            bias=p_update_bias_name)
        m_reset = model.ConvShared(mem_crop,
                            m_reset_name,
                            cfg.MEM.C,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight=m_reset_weight_name,
                            no_bias=True)
        m_update = model.ConvShared(mem_crop,
                            m_update_name,
                            cfg.MEM.C,
                            1,
                            mconv,
                            stride=1,
                            pad=mpad,
                            weight=m_update_weight_name,
                            no_bias=True)
        reset = model.net.Sum([p_reset_name, m_reset_name], reset_name)
        reset = model.net.Sigmoid(reset, reset_name)
        blobs_in = [mem_crop_name, reset_name]
        blobs_out = [mult_mem_name]
        mult_mem = model.MulConvGate(blobs_in, blobs_out)
        m_input = model.ConvShared(mult_mem,
                                    m_input_name,
                                    cfg.MEM.C,
                                    cfg.MEM.C,
                                    mconv,
                                    stride=1,
                                    pad=mpad,
                                    weight=m_input_weight_name,
                                    no_bias=True)

    input = model.net.Sum([p_input_name, m_input_name], input_name)
    if cfg.MEM.ACT == 'tanh':
        input = model.Tanh(input, input)
    elif cfg.MEM.ACT == 'relu':
        input = model.Relu(input, input)
    else:
        raise NotImplementedError
    update = model.net.Sum([p_update_name, m_update_name], update_name)
    update = model.net.Sigmoid(update, update_name)
    next_crop_raw = model.net.Sub([input_name, mem_crop_name], next_crop_raw_name)
    blobs_in = [next_crop_raw_name, update_name]
    blobs_out = [next_crop_name]
    next_crop = model.MulConvGate(blobs_in, blobs_out)

    if 'gpu_0' in p_input._name:
        model.AddSummaryHistogram(p_input._name)
        model.AddSummaryHistogram(m_input._name)
        model.AddSummaryHistogram(p_reset._name)
        model.AddSummaryHistogram(m_reset._name)
        model.AddSummaryHistogram(p_update._name)
        model.AddSummaryHistogram(m_update._name)

        model.AddSummaryHistogram(input._name)
        model.AddSummaryHistogram(reset._name)
        model.AddSummaryHistogram(update._name)
        model.AddSummaryHistogram(mem_crop._name)
        model.AddSummaryHistogram(next_crop_raw._name)

    return next_crop

def _assemble(model, mem, norm_diff, next_crop):
    # assemble back, inverse roi operation
    rois = core.ScopedBlobReference('rois')
    mem_diff = model.InvRoIAlign(rois, mem, next_crop)
    normalized_diff = model.DivConvNorm(mem_diff, norm_diff)

    if 'gpu_0' in mem_diff._name:
        model.AddSummaryHistogram(mem_diff._name)
        model.AddSummaryHistogram(norm_diff._name)
        model.AddSummaryHistogram(normalized_diff._name)

    return normalized_diff

def update(model, mem, norm_diff, conv_crop, dim, cls_score, cls_prob, iter, reuse):
    # make sure everything is feature, no back propagation
    assert cls_score._name.endswith('_nb')
    # assert cls_prob._name.endswith('_nb')
    name_scope = 'mem_%02d' % iter
    # do something with the scale
    # then, get the spatial features
    mem_crop = _mem_roi_align(model, mem, name_scope)
    input_crop = _input_features(model, conv_crop, dim, cls_score, name_scope, reuse)
    next_crop = _inplace_update(model, mem_crop, input_crop, dim, name_scope, reuse)
    mem_diff = _assemble(model, mem, norm_diff, next_crop)
    mem = model.net.Sum([mem_diff, mem], name_scope + '/values')

    return mem

def _bottomtop(model, dim, conv_crop, logits, name_scope, reuse):
    init_conv = ('GaussianFill', {'std': cfg.MEM.STD})
    init_fc = ('GaussianFill', {'std': cfg.MEM.STD * cfg.MEM.FP_R})
    init_bias = ('ConstantFill', {'value': 0.})

    conv_out_name = name_scope + '/input/01_conv_out'
    fc_out_name = name_scope + '/input/01_fc_out'
    if not reuse:
        conv_out = model.Conv(conv_crop,
                            conv_out_name,
                            dim,
                            cfg.MEM.C,
                            1,
                            stride=1,
                            pad=0,
                            weight_init=init_conv,
                            bias_init=init_bias)

        fc_out = model.FC(logits, 
                          fc_out_name,
                          model.num_classes, 
                          cfg.MEM.C,
                          weight_init=init_fc,
                          no_bias=True)
    else:
        conv_weight_name = 'mem_01/input/01_conv_out_w'
        conv_bias_name = 'mem_01/input/01_conv_out_b'
        conv_out = model.ConvShared(conv_crop,
                            conv_out_name,
                            dim,
                            cfg.MEM.C,
                            1,
                            stride=1,
                            pad=0,
                            weight=conv_weight_name,
                            bias=conv_bias_name)
        
        fc_weight_name = 'mem_01/input/01_fc_out_w'
        fc_out = model.FCShared(logits, 
                          fc_out_name,
                          model.num_classes, 
                          cfg.MEM.C,
                          weight=fc_weight_name,
                          no_bias=True)

    # then try to combine them together
    sum_out_name = name_scope + '/input/01_out'
    blobs_in = [conv_out_name, fc_out_name]
    blobs_out = [sum_out_name]
    sum_out = model.SumConvFC(blobs_in, blobs_out)
    sum_out = model.Relu(sum_out, sum_out)

    if 'gpu_0' in sum_out._name:
        model.AddSummaryHistogram(conv_crop._name)
        model.AddSummaryHistogram(logits._name)
        model.AddSummaryHistogram(conv_out._name)
        model.AddSummaryHistogram(fc_out._name)
        model.AddSummaryHistogram(sum_out._name)

    return sum_out

def _input(model, comb_feat, name_scope, reuse):
    num_layers = cfg.MEM.IN_L
    in_conv = cfg.MEM.IN_CONV
    in_pad = (in_conv - 1) // 2
    init_weight = ('XavierFill', {})
    init_bias = ('ConstantFill', {'value': 0.})
    blob_in = comb_feat

    for i in range(2, num_layers+1):
        blob_out_name = name_scope + '/input/%02d' % i
        if not reuse:
            blob_out = model.Conv(blob_in,
                                blob_out_name,
                                cfg.MEM.C,
                                cfg.MEM.C,
                                in_conv,
                                stride=1,
                                pad=in_pad,
                                weight_init=init_weight,
                                bias_init=init_bias)
        else:
            conv_weight_name = 'mem_01/input/%02d_w' % i
            conv_bias_name = 'mem_01/input/%02d_b' % i
            blob_out = model.ConvShared(blob_in,
                                    blob_out_name,
                                    cfg.MEM.C,
                                    cfg.MEM.C,
                                    in_conv,
                                    stride=1,
                                    pad=in_pad,
                                    weight=conv_weight_name,
                                    bias=conv_bias_name)
        blob_in = model.Relu(blob_out, blob_out)
        if 'gpu_0' in blob_in._name:
            model.AddSummaryHistogram(blob_in._name)

    return blob_in

def _build_context_cls(model, mem, name_scope, reuse):
    num_layers = cfg.MEM.CT_L
    cconv = cfg.MEM.CT_CONV
    cpad = (cconv - 1) // 2
    init_weight = ('XavierFill', {})
    init_bias = ('ConstantFill', {'value': 0.})
    if cfg.MEM.AT_MIN:
        # minimal design
        min_iter = 1
    else:
        min_iter = 0

    bl_in = mem
    for nconv in range(1, num_layers+1):
        suffix = '/context/cls_n{}'.format(nconv)
        if not reuse:
            bl_out = model.Conv(
                bl_in,
                name_scope + suffix,
                cfg.MEM.C,
                cfg.MEM.C,
                cconv,
                stride=1,
                pad=cpad,
                weight_init=init_weight,
                bias_init=init_bias)
        else:
            bl_out = model.ConvShared(
                bl_in,
                name_scope + suffix,
                cfg.MEM.C,
                cfg.MEM.C,
                cconv,
                stride=1,
                pad=cpad,
                weight='mem_%02d/context/cls_n%d_w' % (min_iter, nconv),
                bias='mem_%02d/context/cls_n%d_b' % (min_iter, nconv))
        bl_in = model.Relu(bl_out, bl_out)

    return bl_in

def _build_context_res_cls(model, mem_cls, name_scope, reuse):
    num_layers = cfg.MEM.CT_L
    cconv = cfg.MEM.CT_CONV
    cpad = (cconv - 1) // 2
    init_weight = ('XavierFill', {})
    init_bias = ('ConstantFill', {'value': 0.})
    if cfg.MEM.AT_MIN:
        # minimal design
        min_iter = 1
    else:
        min_iter = 0

    # first do a convolution
    if not reuse:
        bl_in = model.Conv(mem_cls,
                            name_scope + '/context/cls_res_n1',
                            cfg.MEM.C,
                            cfg.MEM.C,
                            cconv,
                            stride=1,
                            pad=cpad,
                            weight_init=init_weight,
                            bias_init=init_bias)
    else:
        bl_in = model.ConvShared(
                mem_cls,
                name_scope + '/context/cls_res_n1',
                cfg.MEM.C,
                cfg.MEM.C,
                cconv,
                stride=1,
                pad=cpad,
                weight='mem_%02d/context/cls_res_n1_w' % (min_iter),
                bias='mem_%02d/context/cls_res_n1_b' % (min_iter))
    # then have resnet connections
    for nconv in range(2, num_layers+1, 2):
        if not reuse:
            bl_out1 = model.Relu(bl_in, name_scope + '/context/cls_res_n{}_relu'.format(nconv-1))
            bl_out1 = model.Conv(bl_out1,
                                name_scope + '/context/cls_res_n{}'.format(nconv),
                                cfg.MEM.C,
                                cfg.MEM.C,
                                cconv,
                                stride=1,
                                pad=cpad,
                                weight_init=init_weight,
                                bias_init=init_bias)
            bl_out2 = model.Relu(bl_out1, bl_out1)
            bl_out2 = model.Conv(bl_out2,
                                name_scope + '/context/cls_res_n{}'.format(nconv+1),
                                cfg.MEM.C,
                                cfg.MEM.C,
                                cconv,
                                stride=1,
                                pad=cpad,
                                weight_init=init_weight,
                                bias_init=init_bias)
        else:
            bl_out1 = model.Relu(bl_in, name_scope + '/context/cls_res_n{}_relu'.format(nconv-1))
            bl_out1 = model.ConvShared(bl_out1,
                                name_scope + '/context/cls_res_n{}'.format(nconv),
                                cfg.MEM.C,
                                cfg.MEM.C,
                                cconv,
                                stride=1,
                                pad=cpad,
                                weight='mem_%02d/context/cls_res_n%d_w' % (min_iter, nconv),
                                bias='mem_%02d/context/cls_res_n%d_b' % (min_iter, nconv))
            bl_out2 = model.Relu(bl_out1, bl_out1)
            bl_out2 = model.ConvShared(bl_out2,
                                name_scope + '/context/cls_res_n{}'.format(nconv+1),
                                cfg.MEM.C,
                                cfg.MEM.C,
                                cconv,
                                stride=1,
                                pad=cpad,
                                weight='mem_%02d/context/cls_res_n%d_w' % (min_iter, nconv+1),
                                bias='mem_%02d/context/cls_res_n%d_b' % (min_iter, nconv+1))
        bl_in = model.net.Sum([bl_in, bl_out2], name_scope + '/context/cls_res_r{}'.format(nconv+1))

    return bl_in

def _add_roi_2mlp_head(model, mem_cls, name_scope, reuse):
    init_weight = ('GaussianFill', {'std': cfg.MEM.STD})
    init_bias = ('ConstantFill', {'value': 0.})
    if cfg.MEM.AT_MIN:
        # minimal design
        min_iter = 1
    else:
        min_iter = 0

    roi_size = cfg.MEM.CROP_SIZE
    ctx_crop = _ctx_roi_align(model, mem_cls, name_scope)

    bl_in = ctx_crop
    dim_in = cfg.MEM.C * roi_size * roi_size
    dim_out =  cfg.MEM.FC_C

    for nf in range(cfg.MEM.FC_L):
        suffix = '/fc{}'.format(nf+6)
        if not reuse:
            bl_out = model.FC(bl_in,
                            name_scope + suffix,
                            dim_in, dim_out,
                            weight_init=init_weight,
                            bias_init=init_bias)
        else:
            bl_out = model.FCShared(bl_in,
                            name_scope + suffix,
                            dim_in, dim_out,
                            weight='mem_%02d/fc%d_w' % (min_iter, nf+6),
                            bias='mem_%02d/fc%d_b' % (min_iter, nf+6))
        bl_in = model.Relu(bl_out, bl_out)
        dim_in = dim_out

    return c2_utils.UnscopeGPUName(bl_in._name), dim_out

def _add_outputs(model, blob_fc7, dim_fc7, cls_score_base, name_scope, reuse):
    init_weight = ('GaussianFill', {'std': cfg.MEM.STD})
    init_bias = ('ConstantFill', {'value': 0.})
    if cfg.MEM.AT_MIN:
        # minimal design
        min_iter = 1
    else:
        min_iter = 0

    if not reuse:
        cls_score = model.FC(
                blob_fc7,
                name_scope + '/cls_score',
                dim_fc7,
                model.num_classes,
                weight_init=init_weight,
                bias_init=init_bias)
    else:
        cls_score = model.FCShared(
                blob_fc7,
                name_scope + '/cls_score',
                dim_fc7,
                model.num_classes,
                weight='mem_%02d/cls_score_w' % min_iter,
                bias='mem_%02d/cls_score_b' % min_iter)

    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        cls_prob = model.Softmax(name_scope + '/cls_score', 
                                name_scope + '/cls_prob', 
                                engine='CUDNN')
    else:
        cls_prob = None 

    return cls_score, cls_prob


def _add_attends(model, blob_fc7, dim_fc7, name_scope, reuse):
    init_weight_attend = ('GaussianFill', {'std': cfg.MEM.AT_STD})
    if cfg.MEM.AT_MIN:
        # minimal design
        min_iter = 1
        init_bias_attend = ('ConstantFill', {'value': cfg.MEM.AT_R})
    else:
        min_iter = 0
        init_bias_attend = ('ConstantFill', {'value': 0.})

    # then add attention
    # if cfg.MEM.AT_MIN:
    if not reuse:
        cls_attend = model.FC(
                blob_fc7,
                name_scope + '/cls_attend',
                dim_fc7,
                1,
                weight_init=init_weight_attend,
                bias_init=init_bias_attend)
    else:
        cls_attend = model.FCShared(
                blob_fc7,
                name_scope + '/cls_attend',
                dim_fc7,
                1,
                weight='mem_%02d/cls_attend_w' % min_iter,
                bias='mem_%02d/cls_attend_b' % min_iter)
    # else:
    #     if not reuse:
    #         cls_attend = model.FC(
    #                 blob_fc7,
    #                 name_scope + '/cls_attend',
    #                 dim_fc7,
    #                 1,
    #                 weight_init=init_weight_attend,
    #                 no_bias=True)
    #     else:
    #         cls_attend = model.FCShared(
    #                 blob_fc7,
    #                 name_scope + '/cls_attend',
    #                 dim_fc7,
    #                 1,
    #                 weight='mem_%02d/cls_attend_w' % min_iter,
    #                 no_bias=True)

    return cls_attend


def init_attenton_prediction(model, mem):
    name_scope = 'mem_00'
    # add context to the network
    if cfg.MEM.CT_L:
        mem_cls = _build_context_cls(model, mem, name_scope, False)
    else:
        mem_cls = mem

    blob_fc7, dim_fc7 = _add_roi_2mlp_head(model, mem_cls, name_scope, False)
    cls_attend = _add_attends(model, blob_fc7, dim_fc7, name_scope, False)

    return cls_attend

def prediction(model, mem, cls_score_base, iter, reuse):
    # implement the most basic version of prediction
    name_scope = 'mem_%02d' % iter
    # a hack to fix the reusing issue
    if cfg.MEM.AT_MIN:
        reuse_this = reuse
        reuse_cls = reuse
    elif iter == 1:
        reuse_this = True
        reuse_cls = False
    else:
        reuse_this = True
        reuse_cls = True

    # add context to the network
    if cfg.MEM.CT_L:
        if cfg.MEM.CT == 'plain':
            mem_cls = _build_context_cls(model, mem, name_scope, reuse_this)
        elif cfg.MEM.CT == 'res':
            mem_cls = _build_context_res_cls(model, mem, name_scope, reuse_this)
        else:
            raise NotImplementedError
    else:
        mem_cls = mem

    blob_fc7, dim_fc7 = _add_roi_2mlp_head(model, mem_cls, 
                                            name_scope, 
                                            reuse_this)

    cls_score, cls_prob = _add_outputs(model, 
                                        blob_fc7, 
                                        dim_fc7, 
                                        cls_score_base,
                                        name_scope, 
                                        reuse_cls)

    cls_attend = _add_attends(model, 
                            blob_fc7, 
                            dim_fc7, 
                            name_scope, 
                            reuse_this)

    return cls_score, cls_prob, cls_attend

def combine(model, cls_score_list, cls_attend_list):
    num_preds = len(cls_score_list)
    for cls_score in cls_score_list:
        assert cls_score._name.endswith('_nb')
    if cfg.MEM.AT_MIN:
        assert len(cls_attend_list) == num_preds - 1
    else:
        assert len(cls_attend_list) == num_preds

    if num_preds > 1:
        if cfg.MEM.AT_MIN:
            cls_attend_final = model.ConcatAttentionRegion(cls_attend_list)
        else:
            cls_attend_final = model.ConcatAttentionRegionNormal(cls_attend_list)
    else:
        cls_attend_final = cls_attend_list[0]

    cls_weight_final = model.Softmax(cls_attend_final, 'final/cls_weight', engine='CUDNN')
    cls_score_final = model.ReduceWithAttentionRegion(cls_score_list, cls_weight_final)

    return cls_score_final

def add_final_prob(model, cls_preds_final):
    cls_prob = model.Softmax('final/cls_score', 
                            'final/cls_prob', 
                            engine='CUDNN')
