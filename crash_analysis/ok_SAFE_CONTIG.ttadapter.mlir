#map = affine_map<(d0) -> (d0)>
module {
  func.func @triton_tem_fused_0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4_i32 = arith.constant 4 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %cst_2 = arith.constant 1.44269502 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %6 = tensor.empty() : tensor<1x64xi32>
    %7 = linalg.fill ins(%c64_i32 : i32) outs(%6 : tensor<1x64xi32>) -> tensor<1x64xi32>
    %8 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %10 = arith.divsi %arg15, %c4_i32 : i32
    %11 = arith.remsi %arg15, %c4_i32 : i32
    %12 = arith.muli %10, %c131072_i32 : i32
    %13 = arith.muli %11, %c32768_i32 : i32
    %14 = arith.addi %12, %13 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.index_cast %13 : i32 to index
    %17 = arith.muli %arg14, %c64_i32 : i32
    %18 = tensor.empty() : tensor<64xi32>
    %19 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%18 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %54 = linalg.index 0 : index
      %55 = arith.index_cast %54 : index to i32
      linalg.yield %55 : i32
    } -> tensor<64xi32>
    %20 = arith.divsi %arg14, %c2_i32 : i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.index_cast %17 : i32 to index
    %23 = arith.muli %22, %c64 : index
    %24 = arith.addi %23, %15 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
    %25 = bufferization.to_tensor %alloc restrict writable : memref<64x64xbf16>
    %26 = arith.index_cast %21 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%26], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %27 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %28 = arith.muli %27, %c128_i32 : i32
    %29 = arith.index_cast %20 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%29], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %30 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %31 = arith.muli %30, %c2_i32 : i32
    %32 = arith.minsi %31, %c8_i32 : i32
    %33 = linalg.fill ins(%28 : i32) outs(%18 : tensor<64xi32>) -> tensor<64xi32>
    %34 = arith.addi %33, %19 : tensor<64xi32>
    %expanded = tensor.expand_shape %34 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %35:6 = scf.for %arg17 = %c0_i32 to %32 step %c1_i32 iter_args(%arg18 = %5, %arg19 = %9, %arg20 = %1, %arg21 = %28, %arg22 = %28, %arg23 = %expanded) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>)  : i32 {
      %54 = arith.index_cast %arg22 : i32 to index
      %55 = arith.muli %54, %c64 : index
      %56 = arith.addi %55, %16 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %57 = arith.index_cast %arg21 : i32 to index
      %58 = arith.muli %57, %c64 : index
      %59 = arith.addi %58, %16 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_10, %alloc_12 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %60 = bufferization.to_tensor %alloc_12 restrict writable : memref<64x64xbf16>
      %61 = tensor.empty() : tensor<64x64xbf16>
      %transposed = linalg.transpose ins(%60 : tensor<64x64xbf16>) outs(%61 : tensor<64x64xbf16>) permutation = [1, 0] 
      %62 = linalg.matmul {input_precison = "ieee"} ins(%25, %transposed : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %63 = arith.mulf %62, %4 : tensor<64x64xf32>
      %64 = arith.mulf %63, %3 : tensor<64x64xf32>
      %reduced = linalg.reduce ins(%64 : tensor<64x64xf32>) outs(%1 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %79 = arith.maxnumf %in, %init : f32
          linalg.yield %79 : f32
        }
      %65 = arith.maxnumf %arg20, %reduced : tensor<64xf32>
      %66 = arith.subf %arg20, %65 : tensor<64xf32>
      %67 = math.exp2 %66 : tensor<64xf32>
      %broadcasted_13 = linalg.broadcast ins(%65 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %68 = arith.subf %64, %broadcasted_13 : tensor<64x64xf32>
      %69 = math.exp2 %68 : tensor<64x64xf32>
      %70 = arith.mulf %arg19, %67 : tensor<64xf32>
      %reduced_14 = linalg.reduce ins(%69 : tensor<64x64xf32>) outs(%9 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %79 = arith.addf %in, %init : f32
          linalg.yield %79 : f32
        }
      %71 = arith.addf %70, %reduced_14 : tensor<64xf32>
      %broadcasted_15 = linalg.broadcast ins(%67 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %72 = arith.mulf %arg18, %broadcasted_15 : tensor<64x64xf32>
      %alloc_16 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_11, %alloc_16 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %73 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x64xbf16>
      %74 = arith.truncf %69 : tensor<64x64xf32> to tensor<64x64xbf16>
      %75 = linalg.matmul {input_precison = "ieee"} ins(%74, %73 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%72 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %76 = arith.addi %arg21, %c64_i32 : i32
      %77 = arith.addi %arg22, %c64_i32 : i32
      %78 = arith.addi %arg23, %7 : tensor<1x64xi32>
      scf.yield %75, %71, %65, %76, %77, %78 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>
    }
    %reinterpret_cast_6 = memref.reinterpret_cast %arg9 to offset: [%26], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %36 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %37 = arith.muli %36, %c128_i32 : i32
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%29], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %38 = memref.load %reinterpret_cast_7[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %39 = arith.muli %38, %c2_i32 : i32
    %40 = arith.minsi %39, %c8_i32 : i32
    %41 = linalg.fill ins(%37 : i32) outs(%18 : tensor<64xi32>) -> tensor<64xi32>
    %42 = arith.addi %41, %19 : tensor<64xi32>
    %expanded_8 = tensor.expand_shape %42 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %43:6 = scf.for %arg17 = %c0_i32 to %40 step %c1_i32 iter_args(%arg18 = %35#0, %arg19 = %35#1, %arg20 = %35#2, %arg21 = %37, %arg22 = %37, %arg23 = %expanded_8) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>)  : i32 {
      %54 = arith.index_cast %arg22 : i32 to index
      %55 = arith.muli %54, %c64 : index
      %56 = arith.addi %55, %16 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %57 = arith.index_cast %arg21 : i32 to index
      %58 = arith.muli %57, %c64 : index
      %59 = arith.addi %58, %16 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_10, %alloc_12 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %60 = bufferization.to_tensor %alloc_12 restrict writable : memref<64x64xbf16>
      %61 = tensor.empty() : tensor<64x64xbf16>
      %transposed = linalg.transpose ins(%60 : tensor<64x64xbf16>) outs(%61 : tensor<64x64xbf16>) permutation = [1, 0] 
      %62 = linalg.matmul {input_precison = "ieee"} ins(%25, %transposed : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %63 = arith.mulf %62, %4 : tensor<64x64xf32>
      %64 = arith.mulf %63, %3 : tensor<64x64xf32>
      %reduced = linalg.reduce ins(%64 : tensor<64x64xf32>) outs(%1 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %79 = arith.maxnumf %in, %init : f32
          linalg.yield %79 : f32
        }
      %65 = arith.maxnumf %arg20, %reduced : tensor<64xf32>
      %66 = arith.subf %arg20, %65 : tensor<64xf32>
      %67 = math.exp2 %66 : tensor<64xf32>
      %broadcasted_13 = linalg.broadcast ins(%65 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %68 = arith.subf %64, %broadcasted_13 : tensor<64x64xf32>
      %69 = math.exp2 %68 : tensor<64x64xf32>
      %70 = arith.mulf %arg19, %67 : tensor<64xf32>
      %reduced_14 = linalg.reduce ins(%69 : tensor<64x64xf32>) outs(%9 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %79 = arith.addf %in, %init : f32
          linalg.yield %79 : f32
        }
      %71 = arith.addf %70, %reduced_14 : tensor<64xf32>
      %broadcasted_15 = linalg.broadcast ins(%67 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %72 = arith.mulf %arg18, %broadcasted_15 : tensor<64x64xf32>
      %alloc_16 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_11, %alloc_16 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %73 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x64xbf16>
      %74 = arith.truncf %69 : tensor<64x64xf32> to tensor<64x64xbf16>
      %75 = linalg.matmul {input_precison = "ieee"} ins(%74, %73 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%72 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %76 = arith.addi %arg21, %c64_i32 : i32
      %77 = arith.addi %arg22, %c64_i32 : i32
      %78 = arith.addi %arg23, %7 : tensor<1x64xi32>
      scf.yield %75, %71, %65, %76, %77, %78 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>
    }
    %44 = arith.cmpf oeq, %43#1, %9 : tensor<64xf32>
    %45 = arith.select %44, %8, %43#1 : tensor<64xi1>, tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%45 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
    %46 = arith.divf %43#0, %broadcasted : tensor<64x64xf32>
    %47 = arith.addi %23, %16 : index
    %reinterpret_cast_9 = memref.reinterpret_cast %arg10 to offset: [%47], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %48 = arith.truncf %46 : tensor<64x64xf32> to tensor<64x64xbf16>
    %49 = arith.addi %22, %c64 : index
    %50 = arith.maxsi %22, %c512 : index
    %51 = arith.minsi %49, %50 : index
    %52 = arith.subi %51, %22 : index
    %53 = arith.minsi %52, %c64 : index
    %extracted_slice = tensor.extract_slice %48[0, 0] [%53, 64] [1, 1] : tensor<64x64xbf16> to tensor<?x64xbf16>
    %subview = memref.subview %reinterpret_cast_9[0, 0] [%53, 64] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<?x64xbf16, strided<[64, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x64xbf16>, memref<?x64xbf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}

