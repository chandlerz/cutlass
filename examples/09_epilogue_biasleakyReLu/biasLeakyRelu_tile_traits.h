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

#include <cutlass/coord.h>
#include <cutlass/gemm/gemm_global_tile.h>
#include <cutlass/gemm/gemm_operand.h>
#include <cutlass/matrix_traits.h>
#include <cutlass/shape.h>

namespace cutlass {
namespace gemm {	

/**
 * @brief Traits class to configure the iterator which accesses
 *  the bias row vector
 * @tparam Scalar_ the type of each element in the bias
 * @tparam Tile_ the tile shape
 * @tparam Threads_ the thread shape of each thread block
   @tparam kStrideH_ elements we jump over at each iteration
   @tparam kAccessSize_ the number of scalars accessed
 */
// czhou-comment: Do the first two template parameters make any difference for a vector
template <typename Scalar_, typename Tile_, typename Threads_, int kStrideH_, int kAccessSize_>	
struct GemmGlobalTileBiasTraits : public GemmGlobalTileTraits<GemmOperand::kB,			
                                                            MatrixLayout::kColumnMajor, 
                                                            Scalar_,
                                                            Tile_,
                                                            Threads_,
                                                            kAccessSize_> {

  /// The base class.
  typedef GemmGlobalTileTraits<GemmOperand::kC,
                               MatrixLayout::kColumnMajor,
                               Scalar_,
                               Tile_,
                               Threads_,
                               kAccessSize_>
      Base;

  /// The stride in the H dimension.
  static int const kStrideH = kStrideH_;

  /// Override the strides in each dimension between different loads/stores.
  // czhou-question
#ifdef AALike
  typedef cutlass::Shape<0, 0, 0, Base::Delta::kC> Delta;
#else // BBLike
  typedef Shape<0, 0, Base::Delta::kW, Base::Delta::kC> Delta;
#endif //AALike

  // czhou-question
#ifdef AALike
  /// Override the number of iterations needed to load/store the tile.
  typedef cutlass::Shape<1, Tile_::kH / Threads_::kH, 1,
                         Tile_::kC / kAccessSize_>
    Iterations;
#else // BBLike
  /// Override the number of iterations needed to load/store the tile.
  typedef cutlass::Shape<1, 1, Tile_::kW / Threads_::kW,
                         Tile_::kC / kAccessSize_>
    Iterations;
#endif //AALike

  typedef typename Base::Threads Threads;

  typedef typename Base::ThreadsDelta ThreadsDelta;

  typedef typename Base::ImmediateOffsetStrides ImmediateOffsetStrides;

  /// Computes the thread offset in (H, W) based on thread ID
  // czhou-question: AAlike here since the bias vector is 1 * n 
  struct ThreadOffset {
  CUTLASS_HOST_DEVICE 
  Coord<4> operator()() const {
  		int thread_offset_h = threadIdx.x / Threads::kW * kStrideH * Iterations::kH;	
  		int thread_offset_w = 0;
  
  		return make_Coord(0, thread_offset_h, thread_offset_w, 0);
  	}
  };
}; // struct GemmGlobalTileBiasTraits
}  // namespace gemm
}  // namespace cutlass
