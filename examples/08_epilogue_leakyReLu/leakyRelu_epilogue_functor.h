#pragma once

#include "cutlass/fragment_multiply_add.h"


namespace cutlass {
namespace gemm {
template <typename Scalar_, typename FragmentMultiplyAdd_ = FragmentMultiplyAdd<Scalar_, Scalar_> >
struct ReLuEpilogue
{
  // The scalar.
  typedef Scalar_ Scalar;
  // The accumulator Type
  typedef typename FragmentMultiplyAdd_::ScalarAccum ScalarAccum;
  // The adapater.
  typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;
    
  struct Params {
    /// The alpha/beta scaling params.
    Scalar alpha, beta, lReluFactor;

    // Constructor
    CUTLASS_HOST_DEVICE
    Params(Scalar _alpha = 0.0f, Scalar _beta = 0.0f, Scalar _lReluFactor = 0.0f)
        : alpha(_alpha), beta(_beta), lReluFactor(_lReluFactor) {}

    /// Initialize the parameters
    CUTLASS_HOST_DEVICE int initialize(Scalar _alpha, Scalar _beta, Scalar _lReluFactor) {
      alpha = _alpha;
      beta = _beta;
      lReluFactor = _lReluFactor;
      return 0;
    }

    /// Initialize the parameters.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {
      alpha = desc.alpha;
      beta = desc.beta;
      lReluFactor = desc.lReluFactor;
      return 0;
    }
  };

  CUTLASS_DEVICE ReLuEpilogue() {};

  CUTLASS_DEVICE ReLuEpilogue(Params const& params):
    alpha_(params.alpha),beta_(params.beta), lReluFactor_(params.lReluFactor)
  {}

  /// Method to determine whether the source accumulator matrix C is ever needed. This method
  /// may always safely return true, though better performance is possible if the source accumulator
  /// matrix is never loaded unnecessarily.
  CUTLASS_DEVICE
  bool source_required() const {
    return true;
  }

  // Evaluate the functor
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_& output)
  {
    FragmentMultiplyAdd mad;
    mad.multiply(alpha_, accum, output);

    for (int i=0; i<FragmentB_::kElements; i++)
    {
      output[i] = FragmentB_::Element(0) < output[i] ? output[i]:output[i]*lReluFactor_;
    }
  }

  // Evaluate the functor
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_ const&old, FragmentB_& output)
  {
    FragmentMultiplyAdd mad;
    FragmentB_ tmp;
 
    mad.multiply(beta_, old, tmp); // tmp = beta_ * old    
    mad.multiply_add(alpha_, accum, tmp, output); // output = alpha_*accum + tmp

    for (int i=0; i<FragmentB_::kElements; i++)
    {
      output[i] = FragmentB_::Element(0) < output[i] ? output[i]:output[i]*lReluFactor_;
    }
  }

private:
  Scalar alpha_, beta_, lReluFactor_;
};

}  // namespace gemm
}  // namespace cutlass
