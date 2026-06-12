export const meta = {
  name: 'reorder-opt-loop',
  description: '实现 block reorder -> 正确性验证(Reviewer闭环) -> 性能分析(Reviewer闭环) -> 优化建议',
  phases: [
    { title: '实现', detail: 'Code Agent 实现 flex_attention_reorder.py' },
    { title: '正确性', detail: 'Test -> Reviewer -> Fix -> Loop' },
    { title: '性能分析', detail: 'Bench -> msprof -> Reviewer -> 优化建议' },
  ],
}

var CONFIG = {
  correctnessShapes: [
    '1,4,512,64 causal',
    '2,4,512,64 causal',
    '4,8,2048,128 causal',
    '1,4,512,64 sliding_window_64',
  ],
  perfBench: '--shape 4,8,2048,128 --sparse-config causal --enable-block-reorder --block-reorder-mode wave_overlap --target both --no-compare --warmup 2 --repeat 3',
  perMsprof: '--mode msprof --shape 4,8,2048,128 --sparse-config causal --enable-block-reorder --block-reorder-mode wave_overlap --msprof-aic-metrics PipeUtilization,ArithmeticUtilization --warmup 1 --repeat 1',
  maxRetries: 5,
}

var GP = { agentType: 'general-purpose' }

var FIX_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    stage: { type: 'string', enum: ['correctness'] },
    root_cause: { type: 'string' },
    fix_suggestions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          line_range: { type: 'string' },
          action: { type: 'string', enum: ['修改', '新增', '删除'] },
          detail: { type: 'string' },
        },
        required: ['file', 'action', 'detail'],
      },
    },
  },
  required: ['verdict', 'stage', 'root_cause', 'fix_suggestions'],
}

var PERF_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['OK', 'NEEDS_OPTIMIZATION'] },
    stage: { type: 'string', enum: ['performance'] },
    summary: { type: 'string' },
    bottlenecks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          type: { type: 'string', enum: ['compute-bound', 'memory-bound', 'helper-ops', 'ai-cpu-fallback', 'launch-overhead', 'reorder-overhead'] },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          detail: { type: 'string' },
          metric: { type: 'string' },
        },
        required: ['type', 'severity', 'detail'],
      },
    },
    optimization_directions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          priority: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          suggestion: { type: 'string' },
          expected_benefit: { type: 'string' },
        },
        required: ['priority', 'suggestion'],
      },
    },
  },
  required: ['verdict', 'stage', 'summary', 'bottlenecks', 'optimization_directions'],
}

// ====== 阶段 1: 实现 ======
phase('实现')
log('Code Agent 生成 flex_attention_reorder.py...')

await agent(
  '你是一个 Python 工程师。你的任务是实现 NPU Flex Attention 的 block reorder 模块。\n\n' +
  '1. 读设计文档：\n' +
  '   Read /wyh/code/TempFlex/docs/superpowers/specs/2026-06-08-flex-attention-block-reorder-design.md\n' +
  '   Read /wyh/code/TempFlex/docs/superpowers/plans/2026-06-08-flex-attention-block-reorder.md\n\n' +
  '2. 写文件 Newest/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py：\n' +
  '   - compute_block_hit_rate(kv_indices, kv_num_blocks) -> float\n' +
  '   - rebuild_block_mask(kv_num_blocks, kv_indices, n_blocks_q, n_blocks_kv, device) -> mask_float\n' +
  '   - wave_overlap_reorder(mask_float, wave_size=132, n_iter=10) -> perm\n' +
  '   - ReorderInfo dataclass\n' +
  '   - reorder_flex_forward(q, k, v, kv_num_blocks, kv_indices, ...) -> (ReorderInfo, baseline_hit, reordered_hit)\n' +
  '   - REORDER_REGISTRY\n\n' +
  '3. 检查 apply_newest.sh 是否包含 reorder.py 部署行：\n' +
  '   Read /wyh/code/TempFlex/Newest/apply_newest.sh\n' +
  '   如果没有，用 Edit 添加。参考 flex_attention.py 的 copy_one 写法，把文件名改为 flex_attention_reorder.py。\n\n' +
  '所有函数纯 PyTorch，Python 3.11 兼容。',
  { label: 'generate-reorder-py', ...GP, phase: '实现' }
)

log('部署...')
var deployResult = await agent(
  '执行以下命令：\n\n' +
  'bash /wyh/code/TempFlex/Newest/apply_newest.sh\n\n' +
  'python3 -c "import sys; sys.path.insert(0, \'Newest/site-packages\'); from torch_npu._inductor.kernel.flex_attention_reorder import *; print(\'IMPORT_OK\')"\n\n' +
  '最后一行必须是 IMPORT_OK 才算成功。输出全部结果。',
  { label: 'deploy-reorder', ...GP, phase: '实现' }
)

if (!deployResult || deployResult.indexOf('IMPORT_OK') < 0) {
  log('部署或导入失败')
  return { phase: 'error', message: deployResult || 'null' }
}

// ====== 阶段 2: 正确性验证循环 ======
phase('正确性')
log('正确性验证循环...')

var correctnessPassed = false
var attempt = 0

while (!correctnessPassed && attempt < CONFIG.maxRetries) {
  attempt++
  log('第 ' + attempt + ' 轮测试...')

  var outputs = []
  for (var i = 0; i < CONFIG.correctnessShapes.length; i++) {
    var spec = CONFIG.correctnessShapes[i]
    var parts = spec.split(' ')
    var shape = parts[0]
    var sparse = parts[1]
    var cmd = 'timeout 120 python3 /wyh/code/TempFlex/flex_attention_run_script.py' +
      ' --shape ' + shape +
      ' --sparse-config ' + sparse +
      ' --enable-block-reorder --block-reorder-mode wave_overlap' +
      ' --rtol 2e-2 --atol 2e-2 --warmup 1 --repeat 2'

    log('测试 ' + (i + 1) + ': ' + shape + ' ' + sparse)
    var result = await agent(
      '执行以下 bash 命令，返回 stdout+stderr（不要做分析）：\n\ncd /wyh/code/TempFlex && ' + cmd + ' 2>&1; echo "EXIT_CODE=$?"',
      { label: 'test-' + attempt + '-' + i, ...GP, phase: '正确性' }
    )
    outputs.push(result || '(no output)')
  }

  var allResults = outputs.join('\n--- NEXT TEST ---\n')

  var review = await agent(
    '你是一个严格的代码评审员。分析以下 flex_attention_reorder.py 的正确性测试结果。\n\n' +
    '判断标准：\n' +
    '- PASS: 所有测试 allclose=True, max_rel_diff < 3%, fail_ratio < 1%, 无报错\n' +
    '- FAIL: 任意测试不满足（包括 timeout/killed）\n\n' +
    '测试输出：\n' + allResults + '\n\n' +
    '如果 FAIL，分析根因并给出具体修复建议。\n\n' +
    '输出纯 JSON：\n' +
    '{"verdict":"PASS 或 FAIL","stage":"correctness","root_cause":"根因","fix_suggestions":[{"file":"Newest/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py","line_range":"函数名","action":"修改","detail":"具体改法"}]}',
    { label: 'review-' + attempt, phase: '正确性', schema: FIX_SCHEMA }
  )

  if (!review) return { phase: 'error', message: 'Reviewer returned null' }

  if (review.verdict === 'PASS') {
    correctnessPassed = true
    log('Reviewer: PASS')
    break
  }

  log('FAIL: ' + review.root_cause)
  if (attempt >= CONFIG.maxRetries) break
  if (budget.total && budget.remaining() < 50000) break

  await agent(
    '你是一个 Python 工程师。根据 Reviewer 的修复建议修改代码。\n\n' +
    'Code: /wyh/code/TempFlex/Newest/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py\n' +
    'Root cause: ' + review.root_cause + '\n' +
    'Suggestions: ' + JSON.stringify(review.fix_suggestions) + '\n\n' +
    '步骤：\n' +
    '1. Read 当前 flex_attention_reorder.py\n' +
    '2. 按 fix_suggestions 逐条 Edit 修改\n' +
    '3. Bash: bash /wyh/code/TempFlex/Newest/apply_newest.sh\n' +
    '4. Bash: python3 -c "from torch_npu._inductor.kernel.flex_attention_reorder import *; print(\'OK\')"',
    { label: 'fix-' + attempt, ...GP, phase: '正确性' }
  )
}

if (!correctnessPassed) {
  return { phase: 'correctness-failed', message: attempt + ' 轮修复后仍未通过' }
}

// ====== 阶段 3: 性能分析 ======
phase('性能分析')
log('benchmark...')

var benchOutput = await agent(
  '执行以下 benchmark 命令，返回原始输出：\n\n' +
  'cd /wyh/code/TempFlex && timeout 180 python3 flex_attention_run_script.py ' + CONFIG.perfBench + ' 2>&1; echo "EXIT_CODE=$?"',
  { label: 'benchmark', ...GP, phase: '性能分析' }
)

log('msprof 采集...')
var msprofOutput = await agent(
  '执行以下 msprof 采集命令，返回原始输出：\n\n' +
  'cd /wyh/code/TempFlex && timeout 300 python3 flex_attention_run_script.py ' + CONFIG.perMsprof + ' 2>&1; echo "EXIT_CODE=$?"',
  { label: 'msprof', ...GP, phase: '性能分析' }
)

var msprofDir = await agent(
  '执行以下命令：\n\nls -dt /wyh/code/TempFlex/msprof_out/*/ 2>/dev/null | head -1',
  { label: 'find-msprof-dir', ...GP, phase: '性能分析' }
)

var analysisOutput = 'no msprof data'
if (msprofDir && msprofDir.trim()) {
  analysisOutput = await agent(
    '执行以下命令：\n\ncd /wyh/code/TempFlex && python3 summarize_msprof.py ' + msprofDir.trim() + ' --scope auto 2>&1; echo "EXIT_CODE=$?"',
    { label: 'analyze-msprof', ...GP, phase: '性能分析' }
  )
}

var finalReview = await agent(
  '你是一个 NPU 性能评审员。分析以下 block reorder 的性能测试数据。\n\n' +
  '### Benchmark\n' + benchOutput + '\n\n' +
  '### msprof 解析报告\n' + analysisOutput + '\n\n' +
  '分析：\n' +
  '- reorder hit rate 提升 vs 实际加速比\n' +
  '- 瓶颈类型（compute-bound/memory-bound/helper-ops/launch-overhead/reorder-overhead）\n' +
  '- 与 manual 的差距\n\n' +
  '输出纯 JSON：\n' +
  '{"verdict":"OK 或 NEEDS_OPTIMIZATION","stage":"performance","summary":"总体判断","bottlenecks":[{"type":"...","severity":"P0|P1|P2","detail":"...","metric":"..."}],"optimization_directions":[{"priority":"P0|P1|P2","suggestion":"...","expected_benefit":"..."}]}',
  { label: 'final-review', phase: '性能分析', schema: PERF_SCHEMA }
)

return {
  phase: 'done',
  correctness: { verdict: 'PASS', attempts: attempt },
  performance: finalReview || { verdict: 'UNKNOWN' },
}