#include "select_bottom_n_single_op.h"

using std::pair;
using std::make_pair;
using std::priority_queue;

namespace caffe2 {

namespace {
// to compare two values
template <typename T>
struct _compare_value {
  bool operator()(
      const pair<T, int>& lhs,
      const pair<T, int>& rhs) {
    return (lhs.first < rhs.first);
  }
};

int _build_heap(priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>>* PQ,
                const float* cls_prob, const int num_probs, const int top_n) {
    for (int i=0; i<num_probs; i++) {
        const float prob = cls_prob[i];
        if (PQ->size() < top_n || prob > PQ->top().first) {
            PQ->push(make_pair(prob, i));
        }
        if (PQ->size() > top_n) {
            PQ->pop();
        }
    }
    return PQ->size();
}

}

template <>
bool SelectBottomNSingleOp<float, CPUContext>::RunOnDevice() {
    auto& X = Input(0);  // Input data
    DCHECK_EQ(X.ndim(), 4);
    const int num_images = X.dim32(0);
    DCHECK_LT(im_, num_images);
    const int num_probs = X.dim32(1) * X.dim32(2) * X.dim32(3);
    const float* Xp = X.data<float>() + im_ * num_probs;
    auto* Yi = Output(0);
    auto* Yv = Output(1);

    if (num_probs <= top_n_) {
        // just select everything
        Yi->Resize(num_probs);
        Yv->Resize(num_probs);

        // copy index
        TIndex* yi_data = Yi->mutable_data<TIndex>();
        for (int i=0; i<num_probs; i++) {
            yi_data[i] = static_cast<TIndex>(i);
        }
        // copy values
        context_.Copy<float, CPUContext, CPUContext>(num_probs, Xp, Yv->mutable_data<float>());
        return true;
    }

    Yi->Resize(top_n_);
    Yv->Resize(top_n_);
    TIndex* yi_data = Yi->mutable_data<TIndex>();
    float* yv_data = Yv->mutable_data<float>();

    priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>> PQ;
    _build_heap(&PQ, Xp, num_probs, top_n_);
    for (int i=0; i<top_n_; i++) {
        auto& pqelm = PQ.top();
        yv_data[i] = pqelm.first;
        yi_data[i] = static_cast<TIndex>(pqelm.second);
        PQ.pop();
    }

    return true;
}

REGISTER_CPU_OPERATOR(SelectBottomNSingle, SelectBottomNSingleOp<float, CPUContext>);

OPERATOR_SCHEMA(SelectBottomNSingle)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Get the top (potentially top_n) index for each of the feature maps.
)DOC")
    .Arg(
        "im",
        "(int) Index to the image.")
    .Arg(
        "top_n",
        "(int) Number of top values to consider for each image.")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Output(
        0,
        "yi",
        "index for the top_n values.")
    .Output(
        1,
        "yi",
        "value for the top_n values.");

SHOULD_NOT_DO_GRADIENT(SelectBottomNSingle);

} // namespace caffe2