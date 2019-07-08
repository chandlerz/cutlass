#pragma once

#include "biasLeakyRelu_fragment_multiply_add.h"
#include "biasLeakyRelu_tile_traits.h"

#include <cutlass/fragment.h>
#include <cutlass/gemm/gemm_global_tile.h>
#include <cutlass/gemm/gemm_traits.h>
#include <cutlass/shape.h>
#include <cutlass/fragment_multiply_add.h>
#include <cutlass/gemm/linear_scaling.h>

namespace cutlass {
namespace gemm {

template <typename Scalar_, typename GemmConfig_, typename FragmentMultiplyAdd_ = FragmentMultiplyAdd<Scalar_, Scalar_> >
struct BiasLeakyReluEpilogueFunctor 
{
  // The scalar.
  typedef Scalar_ Scalar;
  typedef Scalar_ InputScalar;
  // The accumulator Type
  typedef typename FragmentMultiplyAdd_::ScalarAccum ScalarAccum;
  // The adapater.
  typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;

  /// The number of iterations in the epilogue
  typedef Shape<1,
                GemmConfig_::MultiplyAdd::AccumulatorsPerThread::kH /
                GemmConfig_::kAccumulatorsPerLdsB,
                GemmConfig_::kAccumulatorsPerLdsB>
  Iterations;
  
  /// The iteration strides in the H/W dimension
  typedef Shape<0,
  			  GemmConfig_::kAccumulatorsPerLdsB *(
    			  GemmConfig_::Warps::kH *GemmConfig_::MultiplyAdd::ThreadsPerWarp::kH - 1),
  			  0>
  Delta;

  /// The traits class to build the iterator to load data from global memory for the Bias
  typedef GemmGlobalTileBiasTraits <
  	InputScalar const,
  	Shape<1, 
  		  GemmConfig_::OutputTile::kH / ShapeCount<Iterations>::kCount,
    		  GemmConfig_::OutputTile::kW>,
  	Shape<1, 
  		  ShapeCount<typename GemmConfig_::Warps>::kCount,
            GemmConfig_::kWarpSize>,
  	// How many elements do we jump over at each iteration?
  	Iterations::kW,
  	GemmConfig_::kScalarsPerLdgB>
  GlobalLoadTileBiasTraits;
  /// The iterator for BB in global memory.
  typedef GemmGlobalIteratorCd<GlobalLoadTileBiasTraits, int>
  GlobalLoadIteratorBias;
    
  struct Params {
    /// The alpha/beta scaling params.
    Scalar alpha, beta, lReluFactor;
	/// The params for the bias row vector iterator
	typename GlobalLoadIteratorBias::Params iterator_bias;
	/// The information from desc
	int m, n, k, ldd;

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
	  m = desc.problem_size.m();
	  n = desc.problem_size.n();
	  k = desc.problem_size.k();

      return 0;
    }

    /// Initialize the custom parameters. User code must call it!
    CUTLASS_HOST_DEVICE int initializeExtra(InputScalar const *row_vec) {

		// czhou: the leading dimension of the row_vec is 1
		// defined at gemm/gemm_global_tile.h at line 413
    	int error_code = iterator_bias.initialize(row_vec, n, 1, n, 0, 0);
    	return error_code;
	}
  }; // end struct Params

  CUTLASS_DEVICE BiasLeakyReluEpilogueFunctor() {}

  CUTLASS_DEVICE BiasLeakyReluEpilogueFunctor(Params const& params):
    alpha_(params.alpha),beta_(params.beta), lReluFactor_(params.lReluFactor)
  {}

  /// Method to determine whether the source accumulator matrix C is ever needed. This method
  /// may always safely return true, though better performance is possible if the source accumulator
  /// matrix is never loaded unnecessarily.
  
  CUTLASS_DEVICE
  bool source_required() const {
    return !cutlass::gemm::is_zero(beta_);
  }

  // Evaluate the functor
  template <typename FragmentA_, typename FragmentB_, typename FragmentRow_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, 
  							   FragmentB_& output, 
							   const int index[FragmentB_::kElements],
							   FragmentRow_ const &row)
  {
    FragmentMultiplyAdd mad;
    mad.multiply(alpha_, accum, output, index, row);

    for (int i=0; i<FragmentB_::kElements; i++)
    {
      output[i] = FragmentB_::Element(0) < output[i] ? output[i]:output[i]*lReluFactor_;
    }
  }

private:
  Scalar alpha_, beta_, lReluFactor_;
}; // BiasLeakyReluEpilogueFunctor

}  // namespace gemm
}  // namespace cutlass
