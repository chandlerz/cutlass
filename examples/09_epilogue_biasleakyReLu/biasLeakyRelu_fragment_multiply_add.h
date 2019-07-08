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

#include <cutlass/fragment.h>
#include <cutlass/shape.h>

namespace cutlass {
namespace gemm {

template < typename ScalarAlphaBeta_, 
  typename ScalarAccum_, 
  bool fragMul2 = true /*number of element per fragment is multiple of 2*/
  >
struct FusedBiasLeakyReluFragmentMultiplyAdd {

  /// The shape of the instruction.
  typedef Shape<1, 1, 1, 1> InstructionShape;
  /// The type for alpha and beta
  typedef ScalarAlphaBeta_ ScalarAlphaBeta;
  /// The type for accumlator
  typedef ScalarAccum_ ScalarAccum;

  /// Ctor.
  CUTLASS_DEVICE FusedBiasLeakyReluFragmentMultiplyAdd() {}

  /// Multiply : d = a*b + bias
  template <typename FragmentB_, typename FragmentCd_, 
  			typename FragmentRow_>
  CUTLASS_DEVICE void multiply(ScalarAlphaBeta a, 
  							   FragmentB_ const& b, FragmentCd_& d,
							   const int index[FragmentCd_::kElements],
							   FragmentRow_ const &row) {
    int const kReduction = FragmentB_::kElements / FragmentCd_::kElements;
    int const width = FragmentCd_::kElements / FragmentRow_::kElements;
    for (int j = 0; j < FragmentCd_::kElements; ++j) {
      d[j] = b[j * kReduction + 0];
      for (int k = 1; k < kReduction; ++k) {
        d[j] += b[j * kReduction + k];
      }
      d[j] = a * ScalarAlphaBeta(d[j]);
   
   	  if (index[j] != -1) {
	  	/// czhou-question: why divide not the module? 
	  	//d[j] += row[j/width];
	  	d[j] += 1.0f;
	  }
    }
  }
}; // FusedBiasLeakyReluFragmentMultiplyAdd
}  // namespace gemm
}  // namespace cutlass
