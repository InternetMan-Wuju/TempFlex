#map = affine_map<(d0) -> (d0)>
module {
  func.func @triton_tem_fused_0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", parallel_mode = "simd"} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4_i32 = arith.constant 4 : i32
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
    %cst_3 = arith.constant 0xFF800000 : f32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tensor.empty() : tensor<64x64xi32>
    %1 = linalg.fill ins(%c64_i32 : i32) outs(%0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %2 = tensor.empty() : tensor<64xf32>
    %3 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = linalg.fill ins(%cst_2 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %6 = linalg.fill ins(%cst_3 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %11 = arith.divsi %arg15, %c4_i32 : i32
    %12 = arith.remsi %arg15, %c4_i32 : i32
    %13 = arith.muli %11, %c131072_i32 : i32
    %14 = arith.muli %12, %c32768_i32 : i32
    %15 = arith.addi %13, %14 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.index_cast %14 : i32 to index
    %18 = arith.muli %arg14, %c64_i32 : i32
    %19 = tensor.empty() : tensor<64xi32>
    %20 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%19 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %57 = linalg.index 0 : index
      %58 = arith.index_cast %57 : index to i32
      linalg.yield %58 : i32
    } -> tensor<64xi32>
    %21 = linalg.fill ins(%18 : i32) outs(%19 : tensor<64xi32>) -> tensor<64xi32>
    %22 = arith.addi %21, %20 : tensor<64xi32>
    %23 = arith.divsi %arg14, %c2_i32 : i32
    %24 = arith.muli %23, %c4_i32 : i32
    %25 = arith.index_cast %18 : i32 to index
    %26 = arith.muli %25, %c64 : index
    %27 = arith.addi %26, %16 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
    %28 = bufferization.to_tensor %alloc restrict writable : memref<64x64xbf16>
    %29 = arith.index_cast %24 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%29], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %30 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %31 = arith.muli %30, %c128_i32 : i32
    %32 = arith.index_cast %23 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%32], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %33 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %34 = arith.muli %33, %c2_i32 : i32
    %35 = arith.minsi %34, %c8_i32 : i32
    %36 = linalg.fill ins(%31 : i32) outs(%19 : tensor<64xi32>) -> tensor<64xi32>
    %37 = arith.addi %36, %20 : tensor<64xi32>
    %expanded = tensor.expand_shape %37 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %broadcasted = linalg.broadcast ins(%22 : tensor<64xi32>) outs(%0 : tensor<64x64xi32>) dimensions = [1] 
    %38:6 = scf.for %arg17 = %c0_i32 to %35 step %c1_i32 iter_args(%arg18 = %8, %arg19 = %10, %arg20 = %3, %arg21 = %31, %arg22 = %31, %arg23 = %expanded) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>)  : i32 {
      %57 = arith.index_cast %arg22 : i32 to index
      %58 = arith.muli %57, %c64 : index
      %59 = arith.addi %58, %17 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %60 = arith.index_cast %arg21 : i32 to index
      %61 = arith.muli %60, %c64 : index
      %62 = arith.addi %61, %17 : index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%62], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_11, %alloc_13 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %63 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x64xbf16>
      %64 = tensor.empty() : tensor<64x64xbf16>
      %transposed = linalg.transpose ins(%63 : tensor<64x64xbf16>) outs(%64 : tensor<64x64xbf16>) permutation = [1, 0] 
      %65 = linalg.matmul {input_precison = "ieee"} ins(%28, %transposed : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %66 = arith.mulf %65, %7 : tensor<64x64xf32>
      %collapsed = tensor.collapse_shape %arg23 [[0, 1]] : tensor<1x64xi32> into tensor<64xi32>
      %broadcasted_14 = linalg.broadcast ins(%collapsed : tensor<64xi32>) outs(%0 : tensor<64x64xi32>) dimensions = [0] 
      %67 = arith.subi %broadcasted, %broadcasted_14 : tensor<64x64xi32>
      %68 = arith.cmpi slt, %67, %1 : tensor<64x64xi32>
      %69 = arith.cmpi sge, %broadcasted, %broadcasted_14 : tensor<64x64xi32>
      %70 = arith.andi %68, %69 : tensor<64x64xi1>
      %71 = arith.select %70, %66, %6 : tensor<64x64xi1>, tensor<64x64xf32>
      %72 = arith.mulf %71, %5 : tensor<64x64xf32>
      %reduced = linalg.reduce ins(%72 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %114 = arith.maxnumf %in, %init : f32
          linalg.yield %114 : f32
        }
      %73 = arith.maxnumf %arg20, %reduced : tensor<64xf32>
      %74 = arith.cmpf oeq, %73, %3 : tensor<64xf32>
      %75 = arith.select %74, %10, %73 : tensor<64xi1>, tensor<64xf32>
      %76 = arith.subf %arg20, %75 : tensor<64xf32>
      %77 = math.exp2 %76 : tensor<64xf32>
      %broadcasted_15 = linalg.broadcast ins(%75 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
      %78 = arith.subf %72, %broadcasted_15 : tensor<64x64xf32>
      %79 = math.exp2 %78 : tensor<64x64xf32>
      %80 = arith.mulf %arg19, %77 : tensor<64xf32>
      %reduced_16 = linalg.reduce ins(%79 : tensor<64x64xf32>) outs(%10 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %114 = arith.addf %in, %init : f32
          linalg.yield %114 : f32
        }
      %81 = arith.addf %80, %reduced_16 : tensor<64xf32>
      %broadcasted_17 = linalg.broadcast ins(%77 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
      %82 = arith.mulf %arg18, %broadcasted_17 : tensor<64x64xf32>
      %alloc_18 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_12, %alloc_18 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %83 = bufferization.to_tensor %alloc_18 restrict writable : memref<64x64xbf16>
      %84 = arith.truncf %79 : tensor<64x64xf32> to tensor<64x64xbf16>
      %85 = linalg.matmul {input_precison = "ieee"} ins(%84, %83 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%82 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %86 = arith.divsi %arg17, %c2_i32 : i32
      %87 = arith.index_cast %86 : i32 to index
      %88 = arith.addi %29, %87 : index
      %reinterpret_cast_19 = memref.reinterpret_cast %arg7 to offset: [%88], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %89 = memref.load %reinterpret_cast_19[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %90 = arith.addi %86, %c1_i32 : i32
      %91 = arith.cmpi slt, %90, %33 : i32
      %92 = arith.addi %88, %c1 : index
      %reinterpret_cast_20 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %93 = memref.load %reinterpret_cast_20[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %94 = tensor.empty() : tensor<1xi1>
      %inserted = tensor.insert %91 into %94[%c0] : tensor<1xi1>
      %95 = tensor.empty() : tensor<1xi32>
      %inserted_21 = tensor.insert %93 into %95[%c0] : tensor<1xi32>
      %96 = linalg.fill ins(%c0_i32 : i32) outs(%95 : tensor<1xi32>) -> tensor<1xi32>
      %97 = arith.select %inserted, %inserted_21, %96 : tensor<1xi1>, tensor<1xi32>
      %extracted = tensor.extract %97[%c0] : tensor<1xi32>
      %98 = arith.addi %arg17, %c1_i32 : i32
      %99 = arith.remsi %98, %c2_i32 : i32
      %100 = arith.cmpi eq, %99, %c0_i32 : i32
      %101 = arith.subi %extracted, %89 : i32
      %102 = arith.muli %101, %c128_i32 : i32
      %103 = arith.subi %102, %c64_i32 : i32
      %104 = arith.extui %100 : i1 to i32
      %105 = arith.muli %103, %104 : i32
      %106 = arith.subi %c1_i32, %104 : i32
      %107 = arith.muli %106, %c64_i32 : i32
      %108 = arith.addi %105, %107 : i32
      %109 = arith.addi %arg21, %108 : i32
      %110 = arith.addi %arg22, %108 : i32
      %111 = tensor.empty() : tensor<1x64xi32>
      %112 = linalg.fill ins(%108 : i32) outs(%111 : tensor<1x64xi32>) -> tensor<1x64xi32>
      %113 = arith.addi %arg23, %112 : tensor<1x64xi32>
      scf.yield %85, %81, %73, %109, %110, %113 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>
    }
    %reinterpret_cast_6 = memref.reinterpret_cast %arg9 to offset: [%29], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %39 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %40 = arith.muli %39, %c128_i32 : i32
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%32], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %41 = memref.load %reinterpret_cast_7[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %42 = arith.muli %41, %c2_i32 : i32
    %43 = arith.minsi %42, %c8_i32 : i32
    %44 = linalg.fill ins(%40 : i32) outs(%19 : tensor<64xi32>) -> tensor<64xi32>
    %45 = arith.addi %44, %20 : tensor<64xi32>
    %expanded_8 = tensor.expand_shape %45 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %46:6 = scf.for %arg17 = %c0_i32 to %43 step %c1_i32 iter_args(%arg18 = %38#0, %arg19 = %38#1, %arg20 = %38#2, %arg21 = %40, %arg22 = %40, %arg23 = %expanded_8) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>)  : i32 {
      %57 = arith.index_cast %arg22 : i32 to index
      %58 = arith.muli %57, %c64 : index
      %59 = arith.addi %58, %17 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %60 = arith.index_cast %arg21 : i32 to index
      %61 = arith.muli %60, %c64 : index
      %62 = arith.addi %61, %17 : index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%62], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_11, %alloc_13 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %63 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x64xbf16>
      %64 = tensor.empty() : tensor<64x64xbf16>
      %transposed = linalg.transpose ins(%63 : tensor<64x64xbf16>) outs(%64 : tensor<64x64xbf16>) permutation = [1, 0] 
      %65 = linalg.matmul {input_precison = "ieee"} ins(%28, %transposed : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %66 = arith.mulf %65, %7 : tensor<64x64xf32>
      %67 = arith.mulf %66, %5 : tensor<64x64xf32>
      %reduced = linalg.reduce ins(%67 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %109 = arith.maxnumf %in, %init : f32
          linalg.yield %109 : f32
        }
      %68 = arith.maxnumf %arg20, %reduced : tensor<64xf32>
      %69 = arith.cmpf oeq, %68, %3 : tensor<64xf32>
      %70 = arith.select %69, %10, %68 : tensor<64xi1>, tensor<64xf32>
      %71 = arith.subf %arg20, %70 : tensor<64xf32>
      %72 = math.exp2 %71 : tensor<64xf32>
      %broadcasted_14 = linalg.broadcast ins(%70 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
      %73 = arith.subf %67, %broadcasted_14 : tensor<64x64xf32>
      %74 = math.exp2 %73 : tensor<64x64xf32>
      %75 = arith.mulf %arg19, %72 : tensor<64xf32>
      %reduced_15 = linalg.reduce ins(%74 : tensor<64x64xf32>) outs(%10 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %109 = arith.addf %in, %init : f32
          linalg.yield %109 : f32
        }
      %76 = arith.addf %75, %reduced_15 : tensor<64xf32>
      %broadcasted_16 = linalg.broadcast ins(%72 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
      %77 = arith.mulf %arg18, %broadcasted_16 : tensor<64x64xf32>
      %alloc_17 = memref.alloc() : memref<64x64xbf16>
      memref.copy %reinterpret_cast_12, %alloc_17 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
      %78 = bufferization.to_tensor %alloc_17 restrict writable : memref<64x64xbf16>
      %79 = arith.truncf %74 : tensor<64x64xf32> to tensor<64x64xbf16>
      %80 = linalg.matmul {input_precison = "ieee"} ins(%79, %78 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%77 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %81 = arith.divsi %arg17, %c2_i32 : i32
      %82 = arith.index_cast %81 : i32 to index
      %83 = arith.addi %29, %82 : index
      %reinterpret_cast_18 = memref.reinterpret_cast %arg9 to offset: [%83], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %84 = memref.load %reinterpret_cast_18[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %85 = arith.addi %81, %c1_i32 : i32
      %86 = arith.cmpi slt, %85, %41 : i32
      %87 = arith.addi %83, %c1 : index
      %reinterpret_cast_19 = memref.reinterpret_cast %arg9 to offset: [%87], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %88 = memref.load %reinterpret_cast_19[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %89 = tensor.empty() : tensor<1xi1>
      %inserted = tensor.insert %86 into %89[%c0] : tensor<1xi1>
      %90 = tensor.empty() : tensor<1xi32>
      %inserted_20 = tensor.insert %88 into %90[%c0] : tensor<1xi32>
      %91 = linalg.fill ins(%c0_i32 : i32) outs(%90 : tensor<1xi32>) -> tensor<1xi32>
      %92 = arith.select %inserted, %inserted_20, %91 : tensor<1xi1>, tensor<1xi32>
      %extracted = tensor.extract %92[%c0] : tensor<1xi32>
      %93 = arith.addi %arg17, %c1_i32 : i32
      %94 = arith.remsi %93, %c2_i32 : i32
      %95 = arith.cmpi eq, %94, %c0_i32 : i32
      %96 = arith.subi %extracted, %84 : i32
      %97 = arith.muli %96, %c128_i32 : i32
      %98 = arith.subi %97, %c64_i32 : i32
      %99 = arith.extui %95 : i1 to i32
      %100 = arith.muli %98, %99 : i32
      %101 = arith.subi %c1_i32, %99 : i32
      %102 = arith.muli %101, %c64_i32 : i32
      %103 = arith.addi %100, %102 : i32
      %104 = arith.addi %arg21, %103 : i32
      %105 = arith.addi %arg22, %103 : i32
      %106 = tensor.empty() : tensor<1x64xi32>
      %107 = linalg.fill ins(%103 : i32) outs(%106 : tensor<1x64xi32>) -> tensor<1x64xi32>
      %108 = arith.addi %arg23, %107 : tensor<1x64xi32>
      scf.yield %80, %76, %68, %104, %105, %108 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, tensor<1x64xi32>
    }
    %47 = arith.cmpf oeq, %46#1, %10 : tensor<64xf32>
    %48 = arith.select %47, %9, %46#1 : tensor<64xi1>, tensor<64xf32>
    %broadcasted_9 = linalg.broadcast ins(%48 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
    %49 = arith.divf %46#0, %broadcasted_9 : tensor<64x64xf32>
    %50 = arith.addi %26, %17 : index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg10 to offset: [%50], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %51 = arith.truncf %49 : tensor<64x64xf32> to tensor<64x64xbf16>
    %52 = arith.addi %25, %c64 : index
    %53 = arith.maxsi %25, %c512 : index
    %54 = arith.minsi %52, %53 : index
    %55 = arith.subi %54, %25 : index
    %56 = arith.minsi %55, %c64 : index
    %extracted_slice = tensor.extract_slice %51[0, 0] [%56, 64] [1, 1] : tensor<64x64xbf16> to tensor<?x64xbf16>
    %subview = memref.subview %reinterpret_cast_10[0, 0] [%56, 64] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<?x64xbf16, strided<[64, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x64xbf16>, memref<?x64xbf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}

