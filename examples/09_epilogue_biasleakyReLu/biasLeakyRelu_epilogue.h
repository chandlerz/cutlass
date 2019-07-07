/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cutlass/convert.h>
#include <cutlass/coord.h>
#include <cutlass/fragment.h>
#include <cutlass/gemm/gemm_epilogue.h>
#include <cutlass/gemm/gemm_global_tile.h>
#include <cutlass/gemm/gemm_traits.h>
#include <cutlass/iterator_access.h>
#include <cutlass/shape.h>

namespace cutlass {
namespace gemm {

namespace {
template <typename OutputIterator>
CUTLASS_HOST_DEVICE void extract_index_from_iterator(
  OutputIterator &iterator, typename OutputIterator::Pointer base_ptr,
  int index[OutputIterator::Fragment::kElements]) {
  int st = 0;
  typename OutputIterator::Pointer current_ptr = iterator.params.pointer;
  typename OutputIterator::Index current_pred_offset =
    iterator.params.predicate_offset;
  for (int d = 0; d < OutputIterator::Iterations::kD; ++d) {
    for (int h = 0; h < OutputIterator::Iterations::kH; ++h) {
      for (int w = 0; w < OutputIterator::Iterations::kW; ++w) {
        int const imm = ComputeOffsetFromStrides<
          typename OutputIterator::Base::ImmediateOffsetStrides>::get(0, 0, w,
                                                                      0);
        index[st++] = iterator.valid(d, h, w, 0)
                        ? (&iterator.params.pointer[imm] - base_ptr)
                        : -1;
        if (w < OutputIterator::Iterations::kW - 1) {
          iterator.inc_w();
        }
      }
      if (h < OutputIterator::Iterations::kH - 1) {
        iterator.inc_h();
      }
    }
    if (d < OutputIterator::Iterations::kD - 1) {
      iterator.inc_d();
    }
  }
  iterator.inc_advance();
  iterator.params.pointer = current_ptr;
  iterator.params.predicate_offset = current_pred_offset;
}

}  // end anonymous namespace

/**
 * @brief Base Epilogue for fused gemm
 * @tparam GemmEpilogueTraits_ the traits class to configure this epilogue
 */
 template <typename GemmEpilogueTraits_>
 struct GemmBiasLeakyReLuEpilogue {
  /// The traits class
  typedef GemmEpilogueTraits_ Traits;
  /// The params
  typedef typename Traits::Params Params;
  /// The shared storage
  typedef typename Traits::SharedStorage SharedStorage;
  
  /// The output tile
  typedef typename Traits::OutputTile OutputTile;
  /// The number of iterations
  typedef typename Traits::Iterations Iterations;
  /// The accumulators
  typedef typename Traits::Accumulators Accumulators;
  /// The scalar.
  typedef typename Traits::Scalar Scalar;
  /// The functor in charge of the math.
  typedef typename Traits::Functor Functor;
  
  /// We do not support 3D or 4D shapes.
  static_assert(Iterations::kD == 1 && Iterations::kC == 1, "Unsupported 3D/4D shapes");

  /// The iterator for C in global memory.
  typedef typename Traits::GlobalLoadIteratorC GlobalLoadIteratorC;
  /// The transformer for C.
  typedef typename Traits::GlobalTransformerC GlobalTransformerC;
  /// The transformer for D.
  typedef typename Traits::GlobalTransformerD GlobalTransformerD;
  /// The iterator for D in global memory.
  typedef typename Traits::GlobalStoreIteratorD GlobalStoreIteratorD;
  /// The iterator to store D in shared memory.
  typedef typename Traits::SharedStoreIteratorD SharedStoreIteratorD;
  /// The shared store transformer for D.
  typedef typename Traits::SharedStoreTransformerD SharedStoreTransformerD;
  /// The iterator to load D in shared memory.
  // czhou-comment:
  // cutlass 1.3 in gemm/gemm_epilogue.h at line #77
  typedef typename Traits::SharedLoadStreamD SharedLoadStreamD;
  // while minseok uses the following in distance/distance_epigloue.h at line 111
  // typedef typename Traits::SharedLoadIteratorD SharedLoadIteratorD;
  /// The shared load transformer for D.
  // typedef cutlass::Copy<typename SharedLoadIteratorD::Fragment>
  //   SharedLoadTransformerD;

  /// The index.
  typedef typename Traits::Index Index;

  /// The scalar for C.
  typedef typename GlobalLoadIteratorC::Scalar ScalarC;
  /// The scalar for D.
  typedef typename GlobalStoreIteratorD::Scalar ScalarD;

  /// The Bias fragment
  typedef typename Functor::GlobalLoadIteratorBias GlobalLoadIteratorBias;

  /// Ctor.
  /// czhou-comment: the interface is also changed in gemm/gemm_epilogue.h at line #94
  CUTLASS_DEVICE GemmBiasLeakyReLuEpilogue(Params const& params_,
                              			   SharedStorage& shared_storage_,
                              			   Coord<3> const& _problem_size)
      : params(params_), shared_storage(shared_storage_), problem_size(_problem_size), functor(params_.functor) {}

  /// Execute the epilogue.
  /// czhou-comment: I believe minseok changes the interface purposes to pase the fin_op 
  CUTLASS_DEVICE void epilogue(Accumulators& accumulators,
                               Coord<3> const& block = make_Coord(0, 0, 0),
                               int batch_id = 0) {
	// The bias fragment
	typename GlobalLoadIteratorBias::Fragment fragment_bias;

	// czhou-question: AAlike here
	// The bias a row vector, or 1 * n 
	// the order of problem_size.idx is [K, N, M]
	Coord<3> const bias_bounds = make_Coord(0, problem_size.idx[1], 1);
	// The bias block size.
	// czhou-question: what does the block means and how about the order in block
	Coord<3> const bias_block  = make_Coord(0, block[1], 0);

    // Preserve the base pointer of the output D matrix
    typename GlobalStoreIteratorD::Pointer global_base_ptr =
      this->params.iterator_d.pointer;

    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < Iterations::kH; ++h) {
      // Compute pointer and predicate offsets for C and D global iterators.
      int const pointer_offset =
          ((params.iterator_d.inc_h * (GlobalStoreIteratorD::Iterations::kH - 1) +
            params.iterator_d.inc_advance) *
               Iterations::kW +
           params.stride_h) *
          h;

      int const predicate_offset =
          ((params.iterator_d.predicate_inc_h * (GlobalStoreIteratorD::Iterations::kH - 1) +
            params.iterator_d.predicate_inc_advance) *
               Iterations::kW +
           Traits::Delta::kH) *
          h;

      // The transformer for D.
      GlobalTransformerD transformer_d;

      // The iterator to store into the D matrix.
      GlobalStoreIteratorD global_store_iterator(
          params.iterator_d, problem_size, block, pointer_offset, predicate_offset);

      SharedStoreTransformerD shared_store_transformer;
      typename SharedStoreTransformerD::OutputFragment shared_store_transformed_d;

      SharedStoreIteratorD shared_store_iterator(
          params.shared_store_iterator_d,
          reinterpret_cast<typename SharedStoreIteratorD::Scalar*>(shared_storage.data()));

      SharedLoadStreamD shared_load_stream(
          params.shared_load_stream_d,
          reinterpret_cast<typename SharedLoadStreamD::Scalar*>(shared_storage.data()));

	  // czhou-comment: copy the AAlike code in distance/distance_epigloue.h at line 254
      // Compute pointer and predicate offsets for bias global iterators.
      int const bias_pointer_offset =
      	((params.functor.iterator_bias.inc_h *
      	 (GlobalLoadIteratorBias::Iterations::kH - 1) +
      	  params.functor.iterator_bias.inc_advance) *
      	  Iterations::kW +
      	  Traits::Delta::kH) *
      	  h;
      int const bias_predicate_offset =
      	((params.functor.iterator_bias.predicate_inc_h *
      	 (GlobalLoadIteratorBias::Iterations::kH - 1) +
      	  params.functor.iterator_bias.predicate_inc_advance) *
      	  Iterations::kW +
      	  Traits::Delta::kH) *
      	  h;

      // The iterator to load the elements of the AA column vector.
      GlobalLoadIteratorBias global_load_iterator_bias(
        params.functor.iterator_bias, bias_bounds, bias_block,
        bias_pointer_offset, bias_predicate_offset);

      CUTLASS_PRAGMA_UNROLL
	  for (int w=0; w<Iterations::kW; ++w)
	  {
        iterator_load(global_load_iterator_bias, fragment_bias);

        // Make sure we can write to shared memory.
        shared_load_fence();

        // Copy the accumulators to shared memory.
        int const offset = (h * Iterations::kW + w) * SharedStoreIteratorD::Fragment::kElements;

        shared_store_transformer.transform(accumulators, offset, shared_store_transformed_d);

		// czhou-question: cutlass 1.3 uses the following line 
        shared_store_iterator.store_post_increment(shared_store_transformed_d);
		// while minseok uses the following line 
        //shared_iterator_store(shared_store_iterator,
        //                      shared_store_transformed_d);

        // Make sure the data is in shared memory.
        shared_store_fence();

        // Copy the accumulators back to registers from shared memory.
		// czhou-question: cutlass 1.3 uses the following two lines
        shared_load_stream.copy();
        shared_load_stream.commit();
		// while minseok uses following lines instead
        // typename SharedLoadIteratorD::Fragment fetched_d;
        // shared_iterator_load(shared_load_iterator, fetched_d);

        // Do the math.
        typename GlobalTransformerD::InputFragment fragment_d;

        // extract the global pointer index for each fragment element
        int index[GlobalStoreIteratorD::Fragment::kElements];
        extract_index_from_iterator(global_store_iterator, global_base_ptr,
                                    index);

        functor.evaluate(shared_load_stream.fragment(), fragment_d, index, fragment_bias)

        // Transform D fragment.
        typename GlobalTransformerD::OutputFragment global_transformed_d;
        transformer_d.transform(fragment_d, global_transformed_d);

        // Copy the results to global memory.
		// czhou-question: the interface also changes in cutlass 1.3
        global_store_iterator.store_post_increment(global_transformed_d);
        //iterator_store(global_store_iterator, transformed_d); // minseok's code
	  }
	}
 }

  /// The memory fence for shared loads.
  CUTLASS_DEVICE void shared_load_fence() { __syncthreads(); }

  /// The memory fence for shared stores.
  CUTLASS_DEVICE void shared_store_fence() { __syncthreads(); }

  /// The params.
  Params const& params;
  /// The shared storage.
  SharedStorage& shared_storage;
  /// The dimensions of the GEMM.
  Coord<3> problem_size;
  // The functor.
  Functor functor;

 };// GemmBiasLeakyReLuEpilogue
}  // namespace gemm
}  // namespace cutlass
