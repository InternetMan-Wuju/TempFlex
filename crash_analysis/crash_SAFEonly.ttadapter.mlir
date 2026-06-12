#map = affine_map<(d0) -> (d0)>
module {
  func.func @triton_tem_fused_0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4_i32 = arith.constant 4 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 1.000000e+00 : f32
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
    %6 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %8 = arith.divsi %arg15, %c4_i32 : i32
    %9 = arith.remsi %arg15, %c4_i32 : i32
    %10 = arith.muli %8, %c131072_i32 : i32
    %11 = arith.muli %9, %c32768_i32 : i32
    %12 = arith.addi %10, %11 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.index_cast %11 : i32 to index
    %15 = arith.muli %arg14, %c64_i32 : i32
    %16 = tensor.empty() : tensor<64xi32>
    %17 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%16 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %52 = linalg.index 0 : index
      %53 = arith.index_cast %52 : index to i32
      linalg.yield %53 : i32
    } -> tensor<64xi32>
    %18 = arith.divsi %arg14, %c2_i32 : i32
    %19 = arith.muli %18, %c4_i32 : i32
    %20 = arith.index_cast %15 : i32 to index
    %21 = arith.muli %20, %c64 : index
    %22 = arith.addi %21, %13 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
    %23 = bufferization.to_tensor %alloc restrict writable : memref<64x64xbf16>
    %24 = arith.index_cast %19 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%24], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %25 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %26 = arith.muli %25, %c128_i32 : i32
    %27 = arith.index_cast %18 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %28 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %29 = arith.muli %28, %c2_i32 : i32
    %30 = arith.minsi %29, %c8_i32 : i32
    %31 = linalg.fill ins(%26 : i32) outs(%16 : tensor<64xi32>) -> tensor<64xi32>
    %32 = arith.addi %31, %17 : tensor<64xi32>
    %expanded = tensor.expand_shape %32 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %33:6 = scf.for %arg17 = %c0_i32 to %30 step %c1_i32 iter_args(%arg18 = %5, %arg19 = %7, %arg20 = %1, %arg21 = %26, %arg22 = %26, %arg23 = %expanded) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>)  : i32 {
      %52 = arith.index_cast %arg22 : i32 to index
      %53 = arith.muli %52, %c64 : index
      %54 = arith.addi %53, %14 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %55 = arith.index_cast %arg21 : i32 to index
      %56 = arith.muli %55, %c64 : index
      %57 = arith.addi %56, %14 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%57], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_10, %alloc_12 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %58 = bufferization.to_tensor %alloc_12 restrict writable : memref<64x64xbf16>
      %59 = tensor.empty() : tensor<64x64xbf16>
      %transposed = linalg.transpose ins(%58 : tensor<64x64xbf16>) outs(%59 : tensor<64x64xbf16>) permutation = [1, 0] 
      %60 = linalg.matmul {input_precison = "ieee"} ins(%23, %transposed : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %61 = arith.mulf %60, %4 : tensor<64x64xf32>
      %62 = arith.mulf %61, %3 : tensor<64x64xf32>
      %reduced = linalg.reduce ins(%62 : tensor<64x64xf32>) outs(%1 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %102 = arith.maxnumf %in, %init : f32
          linalg.yield %102 : f32
        }
      %63 = arith.maxnumf %arg20, %reduced : tensor<64xf32>
      %64 = arith.subf %arg20, %63 : tensor<64xf32>
      %65 = math.exp2 %64 : tensor<64xf32>
      %broadcasted_13 = linalg.broadcast ins(%63 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %66 = arith.subf %62, %broadcasted_13 : tensor<64x64xf32>
      %67 = math.exp2 %66 : tensor<64x64xf32>
      %68 = arith.mulf %arg19, %65 : tensor<64xf32>
      %reduced_14 = linalg.reduce ins(%67 : tensor<64x64xf32>) outs(%7 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %102 = arith.addf %in, %init : f32
          linalg.yield %102 : f32
        }
      %69 = arith.addf %68, %reduced_14 : tensor<64xf32>
      %broadcasted_15 = linalg.broadcast ins(%65 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %70 = arith.mulf %arg18, %broadcasted_15 : tensor<64x64xf32>
      %alloc_16 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_11, %alloc_16 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %71 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x64xbf16>
      %72 = arith.truncf %67 : tensor<64x64xf32> to tensor<64x64xbf16>
      %73 = linalg.matmul {input_precison = "ieee"} ins(%72, %71 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%70 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %74 = arith.divsi %arg17, %c2_i32 : i32
      %75 = arith.index_cast %74 : i32 to index
      %76 = arith.addi %24, %75 : index
      %reinterpret_cast_17 = memref.reinterpret_cast %arg7 to offset: [%76], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %77 = memref.load %reinterpret_cast_17[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %78 = arith.addi %74, %c1_i32 : i32
      %79 = arith.cmpi slt, %78, %28 : i32
      %80 = arith.addi %76, %c1 : index
      %reinterpret_cast_18 = memref.reinterpret_cast %arg7 to offset: [%80], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %81 = memref.load %reinterpret_cast_18[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %82 = tensor.empty() : tensor<1xi1>
      %inserted = tensor.insert %79 into %82[%c0] : tensor<1xi1>
      %83 = tensor.empty() : tensor<1xi32>
      %inserted_19 = tensor.insert %81 into %83[%c0] : tensor<1xi32>
      %84 = linalg.fill ins(%c0_i32 : i32) outs(%83 : tensor<1xi32>) -> tensor<1xi32>
      %85 = arith.select %inserted, %inserted_19, %84 : tensor<1xi1>, tensor<1xi32>
      %extracted = tensor.extract %85[%c0] : tensor<1xi32>
      %86 = arith.addi %arg17, %c1_i32 : i32
      %87 = arith.remsi %86, %c2_i32 : i32
      %88 = arith.cmpi eq, %87, %c0_i32 : i32
      %89 = arith.subi %extracted, %77 : i32
      %90 = arith.muli %89, %c128_i32 : i32
      %91 = arith.subi %90, %c64_i32 : i32
      %92 = arith.extui %88 : i1 to i32
      %93 = arith.muli %91, %92 : i32
      %94 = arith.subi %c1_i32, %92 : i32
      %95 = arith.muli %94, %c64_i32 : i32
      %96 = arith.addi %93, %95 : i32
      %97 = arith.addi %arg21, %96 : i32
      %98 = arith.addi %arg22, %96 : i32
      %99 = tensor.empty() : tensor<1x64xi32>
      %100 = linalg.fill ins(%96 : i32) outs(%99 : tensor<1x64xi32>) -> tensor<1x64xi32>
      %101 = arith.addi %arg23, %100 : tensor<1x64xi32>
      scf.yield %73, %69, %63, %97, %98, %101 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>
    }
    %reinterpret_cast_6 = memref.reinterpret_cast %arg9 to offset: [%24], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %34 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %35 = arith.muli %34, %c128_i32 : i32
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %36 = memref.load %reinterpret_cast_7[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %37 = arith.muli %36, %c2_i32 : i32
    %38 = arith.minsi %37, %c8_i32 : i32
    %39 = linalg.fill ins(%35 : i32) outs(%16 : tensor<64xi32>) -> tensor<64xi32>
    %40 = arith.addi %39, %17 : tensor<64xi32>
    %expanded_8 = tensor.expand_shape %40 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %41:6 = scf.for %arg17 = %c0_i32 to %38 step %c1_i32 iter_args(%arg18 = %33#0, %arg19 = %33#1, %arg20 = %33#2, %arg21 = %35, %arg22 = %35, %arg23 = %expanded_8) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>)  : i32 {
      %52 = arith.index_cast %arg22 : i32 to index
      %53 = arith.muli %52, %c64 : index
      %54 = arith.addi %53, %14 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %55 = arith.index_cast %arg21 : i32 to index
      %56 = arith.muli %55, %c64 : index
      %57 = arith.addi %56, %14 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%57], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_10, %alloc_12 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %58 = bufferization.to_tensor %alloc_12 restrict writable : memref<64x64xbf16>
      %59 = tensor.empty() : tensor<64x64xbf16>
      %transposed = linalg.transpose ins(%58 : tensor<64x64xbf16>) outs(%59 : tensor<64x64xbf16>) permutation = [1, 0] 
      %60 = linalg.matmul {input_precison = "ieee"} ins(%23, %transposed : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %61 = arith.mulf %60, %4 : tensor<64x64xf32>
      %62 = arith.mulf %61, %3 : tensor<64x64xf32>
      %reduced = linalg.reduce ins(%62 : tensor<64x64xf32>) outs(%1 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %102 = arith.maxnumf %in, %init : f32
          linalg.yield %102 : f32
        }
      %63 = arith.maxnumf %arg20, %reduced : tensor<64xf32>
      %64 = arith.subf %arg20, %63 : tensor<64xf32>
      %65 = math.exp2 %64 : tensor<64xf32>
      %broadcasted_13 = linalg.broadcast ins(%63 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %66 = arith.subf %62, %broadcasted_13 : tensor<64x64xf32>
      %67 = math.exp2 %66 : tensor<64x64xf32>
      %68 = arith.mulf %arg19, %65 : tensor<64xf32>
      %reduced_14 = linalg.reduce ins(%67 : tensor<64x64xf32>) outs(%7 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %102 = arith.addf %in, %init : f32
          linalg.yield %102 : f32
        }
      %69 = arith.addf %68, %reduced_14 : tensor<64xf32>
      %broadcasted_15 = linalg.broadcast ins(%65 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %70 = arith.mulf %arg18, %broadcasted_15 : tensor<64x64xf32>
      %alloc_16 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_11, %alloc_16 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %71 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x64xbf16>
      %72 = arith.truncf %67 : tensor<64x64xf32> to tensor<64x64xbf16>
      %73 = linalg.matmul {input_precison = "ieee"} ins(%72, %71 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%70 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %74 = arith.divsi %arg17, %c2_i32 : i32
      %75 = arith.index_cast %74 : i32 to index
      %76 = arith.addi %24, %75 : index
      %reinterpret_cast_17 = memref.reinterpret_cast %arg9 to offset: [%76], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %77 = memref.load %reinterpret_cast_17[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %78 = arith.addi %74, %c1_i32 : i32
      %79 = arith.cmpi slt, %78, %36 : i32
      %80 = arith.addi %76, %c1 : index
      %reinterpret_cast_18 = memref.reinterpret_cast %arg9 to offset: [%80], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %81 = memref.load %reinterpret_cast_18[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %82 = tensor.empty() : tensor<1xi1>
      %inserted = tensor.insert %79 into %82[%c0] : tensor<1xi1>
      %83 = tensor.empty() : tensor<1xi32>
      %inserted_19 = tensor.insert %81 into %83[%c0] : tensor<1xi32>
      %84 = linalg.fill ins(%c0_i32 : i32) outs(%83 : tensor<1xi32>) -> tensor<1xi32>
      %85 = arith.select %inserted, %inserted_19, %84 : tensor<1xi1>, tensor<1xi32>
      %extracted = tensor.extract %85[%c0] : tensor<1xi32>
      %86 = arith.addi %arg17, %c1_i32 : i32
      %87 = arith.remsi %86, %c2_i32 : i32
      %88 = arith.cmpi eq, %87, %c0_i32 : i32
      %89 = arith.subi %extracted, %77 : i32
      %90 = arith.muli %89, %c128_i32 : i32
      %91 = arith.subi %90, %c64_i32 : i32
      %92 = arith.extui %88 : i1 to i32
      %93 = arith.muli %91, %92 : i32
      %94 = arith.subi %c1_i32, %92 : i32
      %95 = arith.muli %94, %c64_i32 : i32
      %96 = arith.addi %93, %95 : i32
      %97 = arith.addi %arg21, %96 : i32
      %98 = arith.addi %arg22, %96 : i32
      %99 = tensor.empty() : tensor<1x64xi32>
      %100 = linalg.fill ins(%96 : i32) outs(%99 : tensor<1x64xi32>) -> tensor<1x64xi32>
      %101 = arith.addi %arg23, %100 : tensor<1x64xi32>
      scf.yield %73, %69, %63, %97, %98, %101 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>
    }
    %42 = arith.cmpf oeq, %41#1, %7 : tensor<64xf32>
    %43 = arith.select %42, %6, %41#1 : tensor<64xi1>, tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%43 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
    %44 = arith.divf %41#0, %broadcasted : tensor<64x64xf32>
    %45 = arith.addi %21, %14 : index
    %reinterpret_cast_9 = memref.reinterpret_cast %arg10 to offset: [%45], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %46 = arith.truncf %44 : tensor<64x64xf32> to tensor<64x64xbf16>
    %47 = arith.addi %20, %c64 : index
    %48 = arith.maxsi %20, %c512 : index
    %49 = arith.minsi %47, %48 : index
    %50 = arith.subi %49, %20 : index
    %51 = arith.minsi %50, %c64 : index
    %extracted_slice = tensor.extract_slice %46[0, 0] [%51, 64] [1, 1] : tensor<64x64xbf16> to tensor<?x64xbf16>
    %subview = memref.subview %reinterpret_cast_9[0, 0] [%51, 64] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<?x64xbf16, strided<[64, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x64xbf16>, memref<?x64xbf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}

