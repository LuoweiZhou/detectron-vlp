#include "retinanet_detection_op.h"

using std::pair;
using std::make_pair;
using std::sort;
using std::priority_queue;
using std::vector;
using ::exp;

namespace caffe2 {

const float EXP_UP = 4.1351666;

// to compare two values
template <typename T>
struct _compare_value {
  bool operator()(
      const pair<T, int>& lhs,
      const pair<T, int>& rhs) {
    return (lhs.first > rhs.first);
  }
};

float _clip_min(const float f, const float m) {
    return (f > m) ? (f) : (m);
}

float _clip_max(const float f, const float m) {
    return (f < m) ? (f) : (m);
}

// ignore the weights here as they are not used anyway
void _bbox_transform_clip(const float x1, const float y1, const float x2, const float y2,
                        const float dx, const float dy, float dw, float dh,
                        const float height_max, const float width_max,
                        float* xx1, float* yy1, float* xx2, float* yy2, bool* valid) {
    const float width = x2 - x1 + 1.;
    const float height = y2 - y1 + 1.;
    const float ctr_x = x1 + 0.5 * width;
    const float ctr_y = y1 + 0.5 * height;

    dw = (dw > EXP_UP) ? EXP_UP : dw;
    dh = (dh > EXP_UP) ? EXP_UP : dh;

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;

    // including clipping
    *xx1 = _clip_max(_clip_min(pred_ctr_x - 0.5 * pred_w, 0.), width_max);
    *yy1 = _clip_max(_clip_min(pred_ctr_y - 0.5 * pred_h, 0.), height_max);
    *xx2 = _clip_max(_clip_min(pred_ctr_x + 0.5 * pred_w - 1., 0.), width_max);
    *yy2 = _clip_max(_clip_min(pred_ctr_y + 0.5 * pred_h - 1., 0.), height_max); 

    pred_w = (*xx2) - (*xx1) + 1.;
    pred_h = (*yy2) - (*yy1) + 1.;

    if (pred_w < 1. || pred_h < 1.)
        *valid = false;
    else
        *valid = true;
}

// modified nms, will only return the top_n boxes 
int _nms_cpu(float* boxes_arrow,
            vector<pair<float, int>>& index_list,
            const int num,
            const float threshold,
            const int top_n) {
    int cnt = 0;
    for (int i=0; i<num; i++) {
        const int ind = index_list[i].second;
        const int bp = ind * 7;
        // if not suppressed
        if (boxes_arrow[bp + 6] < 1.) {
            // leave it untouched
            const float x1A = boxes_arrow[bp];
            const float y1A = boxes_arrow[bp+1];
            const float x2A = boxes_arrow[bp+2];
            const float y2A = boxes_arrow[bp+3];
            const float areaA = boxes_arrow[bp + 5]; 
            // suppress others
            for (int j=i+1; j<num; j++) {
                const int jnd = index_list[j].second;
                const int bbp = jnd * 7;
                const float x1B = boxes_arrow[bbp];
                const float y1B = boxes_arrow[bbp+1];
                const float x2B = boxes_arrow[bbp+2];
                const float y2B = boxes_arrow[bbp+3];
                const float areaB = boxes_arrow[bbp + 5];

                const float xx1 = (x1A > x1B) ? x1A : x1B;
                const float yy1 = (y1A > y1B) ? y1A : y1B;
                const float xx2 = (x2A < x2B) ? x2A : x2B;
                const float yy2 = (y2A < y2B) ? y2A : y2B;

                float w = xx2 - xx1 + 1.;
                w = (w > 0.) ? w : 0.;
                float h = yy2 - yy1 + 1.;
                h = (h > 0.) ? h : 0.;
                const float inter = w * h;
                const float iou = inter / (areaA + areaB - inter);

                if (iou >= threshold) {
                    boxes_arrow[bbp + 6] = 1.;
                }
            }
            cnt ++;
        }
        // enough boxes
        if (cnt == top_n) {
            // suppress all the rest
            for (int j=i+1; j<num; j++) {
                const int jnd = index_list[j].second;
                boxes_arrow[jnd * 7 + 6] = 1.;
            }
            break;
        }
    }
    return cnt;
}

int _build_heap(priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>>* PQ,
                const float* cls_prob, const int num_probs, const int top_n, const float threshold) {
    for (int i=0; i<num_probs; i++) {
        const float prob = cls_prob[i];
        if (prob <= threshold)
            continue;
        if (PQ->size() < top_n || prob > PQ->top().first) {
            PQ->push(make_pair(prob, i));
        }
        if (PQ->size() > top_n) {
            PQ->pop();
        }
    }
    return PQ->size();
}

void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

template <>
int RetinanetDetectionOp<float, CPUContext>::_detection_for_one_image(
                                              vector<const float*>& cls_preds,
                                              vector<const float*>& cls_probs,
                                              vector<const float*>& box_preds,
                                              vector<const float*>& anchors,
                                              const float* im_info_pointer,
                                              vector<int>& heights,
                                              vector<int>& widths,
                                              const int im,
                                              const int num_classes,
                                              const int num_levels,
                                              float* roi_pointer,
                                              float* feat_pointer,
                                              const int num_rois) {

    Tensor<CPUContext> boxes_per_level[num_levels];
    Tensor<CPUContext> feats_per_level[num_levels];
    int per_level_counts[num_levels];
    for (int i=0; i<num_levels; i++)
        per_level_counts[i] = 0;

    Tensor<CPUContext> boxes_per_class[num_classes];
    Tensor<CPUContext> feats_per_class[num_classes];
    int per_class_counts[num_classes];
    for (int i=0; i<num_classes; i++)
        per_class_counts[i] = 0;

    Tensor<CPUContext> all_boxes;
    Tensor<CPUContext> all_feats;
    vector<pair<float, int>> all_scores;

    const float im_height = im_info_pointer[im*3];
    const float im_width = im_info_pointer[im*3 + 1];

    for (int cnt=0; cnt<num_levels; cnt++) {
        const int lvl = k_min_ + cnt;
        const float stride = pow(2., lvl);
        const float* anchor = anchors[cnt];
        const int height = heights[cnt];
        const int width = widths[cnt];
        const int pixel = height * width;

        const int num_probs = A_ * num_classes * pixel;
        const int offset_cls = im * num_probs;
        const float* cls_pred = cls_preds[cnt] + offset_cls;
        const float* cls_prob = cls_probs[cnt] + offset_cls;
        const int offset_box = im * A_ * 4 * pixel;
        const float* box_pred = box_preds[cnt] + offset_box;
        // first find the top n largest predictions, we build a heap
        priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>> PQ;
        const float threshold = (cnt == num_levels - 1) ? 0.0 : thresh_;
        const int num_top = _build_heap(&PQ, cls_prob, num_probs, top_n_, threshold);
        // x1, y1, x2, y2, score, c
        boxes_per_level[cnt].Resize(num_top, 6);
        float* boxes_pointer = boxes_per_level[cnt].mutable_data<float>();
        feats_per_level[cnt].Resize(num_top, num_classes);
        float* feats_pointer = feats_per_level[cnt].mutable_data<float>();

        int num_valid = 0;
        for (int i=0; i<num_top; i++) {
            auto& pqelm = PQ.top();
            const float cls_prob_val = pqelm.first;
            const int index = pqelm.second;
            PQ.pop();
            int ind = index;
            const int w = ind % width;
            ind /= width;
            const int h = ind % height;
            ind /= height;
            const int c = ind % num_classes;
            const int a = ind / num_classes;
            // anchor pointer
            const int ap = a * 4;
            // box pointer
            const int bp = (a * 4 * height + h) * width + w;

            const float x = w * stride;
            const float y = h * stride;

            const float x1 = x + anchor[ap];
            const float y1 = y + anchor[ap+1];
            const float x2 = x + anchor[ap+2];
            const float y2 = y + anchor[ap+3];

            const float dx = box_pred[bp];
            const float dy = box_pred[bp + pixel];
            const float dw = box_pred[bp + 2 * pixel];
            const float dh = box_pred[bp + 3 * pixel];

            float xx1, yy1, xx2, yy2;
            bool valid;
            _bbox_transform_clip(x1, y1, x2, y2, dx, dy, dw, dh, 
                                im_height-1., im_width-1.,
                                &xx1, &yy1, &xx2, &yy2, &valid);
            if (valid) {
                const int bbp = num_valid * 6;
                boxes_pointer[bbp] = xx1;
                boxes_pointer[bbp + 1] = yy1;
                boxes_pointer[bbp + 2] = xx2;
                boxes_pointer[bbp + 3] = yy2;
                boxes_pointer[bbp + 4] = cls_prob_val;
                boxes_pointer[bbp + 5] = static_cast<float>(c);

                per_class_counts[c] ++;

                // also copy the features, note that is starts from class 0
                const int fp = num_valid * num_classes;
                const int ffp = (a * num_classes * height + h) * width + w;
                for (int j=0, jj=0; j<num_classes; j++, jj+=pixel) {
                    feats_pointer[fp + j] = cls_pred[ffp + jj];
                }
                num_valid ++;
            }
        }
        per_level_counts[cnt] = num_valid;
    }

    int num_total = 0;
    for (int cls=0; cls<num_classes; cls++) {
        const int current_cls_count = per_class_counts[cls];
        // no boxes
        if (!current_cls_count)
            continue;
        // get boxes and scores for each class
        // x1, y1, x2, y2, score, area, suppressed
        boxes_per_class[cls].Resize(current_cls_count, 7);
        float* boxes_arrow = boxes_per_class[cls].mutable_data<float>();
        feats_per_class[cls].Resize(current_cls_count, num_classes);
        float* feats_arrow = feats_per_class[cls].mutable_data<float>();
        vector<pair<float, int>> index_list;
        int num_current = 0;
        for (int cnt=0; cnt<num_levels; cnt++) {
            const float* boxes_pointer = boxes_per_level[cnt].data<float>();
            const float* feats_pointer = feats_per_level[cnt].data<float>();
            const int num_valid = per_level_counts[cnt];
            for (int i=0; i<num_valid; i++) {
                const int bbp = i * 6;
                if (static_cast<int>(boxes_pointer[bbp + 5]) == cls) {
                    // copy the stuff
                    const int bp = num_current * 7; 
                    _copy(5, boxes_pointer + bbp, boxes_arrow + bp);
                    index_list.push_back(make_pair(boxes_arrow[bp + 4], num_current));
                    // (y2 - y1 + 1) * (x2 - x1 + 1)
                    boxes_arrow[bp + 5] = (boxes_arrow[bp + 3] - boxes_arrow[bp + 1] + 1.) *
                                          (boxes_arrow[bp + 2] - boxes_arrow[bp] + 1.);
                    // everything not suppressed
                    boxes_arrow[bp + 6] = 0.;
                    const int ffp = i * num_classes;
                    const int fp = num_current * num_classes;
                    _copy(num_classes, feats_pointer + ffp, feats_arrow + fp);
                    num_current ++;
                }
            }
        }
        DCHECK_EQ(num_current, current_cls_count);
        // first sort the entries
        sort(index_list.begin(), index_list.end(), _compare_value<float>());
        // do nms
        const int num_kepted = _nms_cpu(boxes_arrow, 
                                        index_list, 
                                        current_cls_count,
                                        nms_,
                                        dpi_);
        num_total += num_kepted;
    }

    // merge all the boxes
    // x1, y1, x2, y2
    all_boxes.Resize(num_total, 4);
    float* all_boxes_pointer = all_boxes.mutable_data<float>();
    all_feats.Resize(num_total, num_classes);
    float* all_feats_pointer = all_feats.mutable_data<float>();
    int num = 0;
    for (int cls=0; cls<num_classes; cls++) {
        const int current_cls_count = per_class_counts[cls];
        // no boxes
        if (!current_cls_count)
            continue;
        const float* boxes_arrow = boxes_per_class[cls].data<float>();
        const float* feats_arrow = feats_per_class[cls].data<float>();
        for (int i=0; i<current_cls_count; i++) {
            const int bp = i * 7;
            const int fp = i * num_classes;
            if (boxes_arrow[bp + 6] < 1.) {
                // copy everything
                const int bbp = num * 4;
                _copy(4, boxes_arrow + bp, all_boxes_pointer + bbp);
                all_scores.push_back(make_pair(boxes_arrow[bp + 4], num));
                const int ffp = num * num_classes;
                _copy(num_classes, feats_arrow + fp, all_feats_pointer + ffp);
                num ++;
            }
        }
    }
    DCHECK_EQ(num, num_total);
    // sort the scores
    sort(all_scores.begin(), all_scores.end(), _compare_value<float>());
    // get the top dpi detections and store them
    // im, x1, y1, x2, y2
    float* current_roi_pointer = roi_pointer + num_rois * 5;
    float* current_feat_pointer = feat_pointer + num_rois * num_classes;
    int num_to_update = (num_total > dpi_) ? dpi_ : num_total;
    for (int i=0; i<num_to_update; i++) {
        const int ind = all_scores[i].second;
        const int bp = ind * 4;
        const int fp = ind * num_classes;
        const int bbp = i * 5;
        const int ffp = i * num_classes;
        current_roi_pointer[bbp] = static_cast<float>(im);
        _copy(4, all_boxes_pointer + bp, current_roi_pointer + bbp + 1);
        _copy(num_classes, all_feats_pointer + fp, current_feat_pointer + ffp);
    }
    // clean up memory

    for (int cnt=0; cnt<num_levels; cnt++) {
        boxes_per_level[cnt].FreeMemory();
        feats_per_level[cnt].FreeMemory();
    }

    for (int cls=0; cls<num_classes; cls++) {
        boxes_per_class[cls].FreeMemory();
        feats_per_class[cls].FreeMemory();
    }
    all_boxes.FreeMemory();
    all_feats.FreeMemory();

    return num_to_update;
}

template <>
bool RetinanetDetectionOp<float, CPUContext>::RunOnDevice() {
    // get inputs
    const int num_inputs = InputSize();
    DCHECK_EQ(num_inputs % 4, 1);
    const int num_levels = num_inputs / 4;
    DCHECK_EQ(num_levels, k_max_ - k_min_ + 1);

    auto& im_info = Input(0);
    const int num_images = im_info.dim32(0);
    DCHECK_EQ(im_info.dim32(1), 3);
    auto& X = Input(1);
    const int num_classes = X.dim32(1) / A_;

    vector<const float*> cls_preds;
    vector<const float*> cls_probs;
    vector<const float*> box_preds;
    vector<const float*> anchors;
    vector<int> heights;
    vector<int> widths;

    const float* im_info_pointer = im_info.data<float>();

    for (int i=1; i<num_inputs; i+=4) {
        auto& cls_pred = Input(i);
        heights.push_back(cls_pred.dim32(2));
        widths.push_back(cls_pred.dim32(3));
        cls_preds.push_back(cls_pred.data<float>());

        auto& cls_prob = Input(i+1);
        cls_probs.push_back(cls_prob.data<float>());
        auto& box_pred = Input(i+2);
        box_preds.push_back(box_pred.data<float>());
        auto& anchor = Input(i+3);
        anchors.push_back(anchor.data<float>());
    }

    Tensor<CPUContext> roi_vecs;
    Tensor<CPUContext> feat_vecs;
    roi_vecs.Resize(num_images * dpi_, 5);
    feat_vecs.Resize(num_images * dpi_, num_classes);
    float* roi_pointer = roi_vecs.mutable_data<float>();
    float* feat_pointer = feat_vecs.mutable_data<float>();

    int num_rois = 0;
    for (int im=0; im<num_images; im++) {
        int num_current_rois = _detection_for_one_image(cls_preds,
                                                        cls_probs,
                                                        box_preds,
                                                        anchors,
                                                        im_info_pointer,
                                                        heights,
                                                        widths,
                                                        im,
                                                        num_classes,
                                                        num_levels,
                                                        roi_pointer,
                                                        feat_pointer,
                                                        num_rois);
        num_rois += num_current_rois;
    }

    // dump to outputs
    auto* rois = Output(0);
    auto* feats = Output(1);
    rois->Resize(num_rois, 5);
    feats->Resize(num_rois, num_classes);

    _copy(num_rois * 5, roi_pointer, rois->mutable_data<float>());
    _copy(num_rois * num_classes, feat_pointer, feats->mutable_data<float>());

    roi_vecs.FreeMemory();
    feat_vecs.FreeMemory();

    return true;
}

REGISTER_CPU_OPERATOR(RetinanetDetection, RetinanetDetectionOp<float, CPUContext>);

// Input: im_info, cls_preds, cls_probs, bbox_preds, cell_anchors
// Output: rois, feats
OPERATOR_SCHEMA(RetinanetDetection)
    .NumInputs(5, INT_MAX)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Produce RetinaNet Detection Results.
)DOC")
    .Arg(
        "k_min",
        "(int) The finest level.")
    .Arg(
        "k_max",
        "(int) The coarsest level.")
    .Arg(
        "A",
        "(int) Product of scale and aspect ratio per location.")
    .Arg(
        "top_n",
        "(int) Number of top values to consider at each level.")
    .Arg(
        "nms",
        "(float) NMS threshold.")
    .Arg(
        "dpi",
        "(int) Number of detections per image.")
    .Input(
        0,
        "im_info",
        "Image information of shape (N, 3).")
    .Output(
        0,
        "rois",
        "2D tensor of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Output(
        1,
        "feats",
        "2D tensor of shape (R, F) specifying R RoI features.");

SHOULD_NOT_DO_GRADIENT(RetinanetDetection);

} // namespace caffe2