#!/usr/bin/env python3
"""Generate wave-variant kernel template with proper indentation.

Reads compute_flex_attention from the source, wraps the Q-block
processing section in a for wave_w in range(WAVE_SIZE) loop,
and writes the result as compute_flex_attention_wave.

Usage: python3 generate_wave_template.py [--apply]
  --dry-run (default): validate and show diff
  --apply: write to flex_attention.py
"""
import re, sys

FLEXPATH = "/wyh/code/TempFlex/Newest/site-packages/torch_npu/_inductor/kernel/flex_attention.py"

def extract_template(content):
    """Extract the compute_flex_attention raw string."""
    # Note: closing is "\n """ (newline + space + triple-quote)
    m = re.search(r'compute_flex_attention = r"""\n(.*?)\n """', content, re.DOTALL)
    if not m:
        raise RuntimeError("Cannot find compute_flex_attention")
    # start(1) = start of captured content (after opening r"""\n)
    # end(1)   = end of captured content (before closing \n """)
    return m.group(1), m.start(1), m.end(1), len('\n """')

def generate_wave(body):
    """Transform body to wave-variant."""
    lines = body.split('\n')
    result = []
    in_wave_body = False
    wave_indent = '    '       # 4 spaces
    body_indent = '        '   # 8 spaces

    # --- Phase 1: Find key markers ---
    init_idx = None
    lse_start = None
    lse_end = None
    perm_start = None
    perm_end = None
    q_start_idx = None

    for i, line in enumerate(lines):
        s = line.strip()
        if s == 'q_start = tl.program_id(0)':
            q_start_idx = i
        if s == '# initialize pointer to m and l':
            init_idx = i
        if s == '# PERM is at mask-block level (size = n_mask_blocks).':
            perm_start = i
        if perm_start and s == 'else:':
            # Check if this is the PERM else
            if i > perm_start and 'src_block = q_start' in lines[i+1]:
                perm_end = i + 1
        if s.startswith('if OUTPUT_LOGSUMEXP:'):
            lse_start = i

    # Find LSE end: the last line before the raw string closing
    # Look for the tl.store inside OUTPUT_LOGSUMEXP
    if lse_start:
        for i in range(lse_start, len(lines)):
            if lines[i].strip() == 'tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)':
                lse_end = i  # last line of Q-block processing

    if None in (init_idx, lse_end, q_start_idx):
        raise RuntimeError(f"Missing markers: init={init_idx}, lse_end={lse_end}, q_start={q_start_idx}")

    print(f"  Markers: q_start@L{q_start_idx}, PERM@L{perm_start}-{perm_end}, init@L{init_idx}, LSE_end@L{lse_end}")

    # --- Phase 2: Build wave-variant line by line ---
    pc = 0  # processing counter
    for i, line in enumerate(lines):
        s = line.rstrip()

        # Before q_start: keep as-is
        if i < q_start_idx:
            result.append(s)
            continue

        # q_start line → wave_id
        if i == q_start_idx:
            result.append(wave_indent + 'wave_id = tl.program_id(0)')
            continue

        # PERM block: skip (moved inside loop)
        if perm_start and perm_start <= i <= perm_end:
            continue

        # Before init marker but after q_start/PERM: keep as-is
        if i < init_idx:
            result.append(s)
            continue

        # At init marker: insert wave loop header
        if i == init_idx:
            result.append('')
            result.append(wave_indent + '# Wave-based processing: each program handles WAVE_SIZE Q blocks.')
            result.append(wave_indent + '# The grid is reduced by WAVE_SIZE (see _flex_attention_grid_with_wave).')
            result.append(wave_indent + 'wave_start = wave_id * WAVE_SIZE')
            result.append(wave_indent + 'for wave_w in range(WAVE_SIZE):')
            result.append(body_indent + 'q_start = wave_start + wave_w')
            result.append(body_indent + '# PERM lookup for this Q block in the wave')
            result.append(body_indent + 'if ENABLE_REORDER:')
            result.append(body_indent + '    src_mask_block = tl.load(PERM + (q_start // SPARSE_Q_MULTIPLE))')
            result.append(body_indent + '    src_block = src_mask_block * SPARSE_Q_MULTIPLE + (q_start % SPARSE_Q_MULTIPLE)')
            result.append(body_indent + 'else:')
            result.append(body_indent + '    src_block = q_start')
            result.append('')
            result.append(body_indent + '# initialize pointer to m and l')
            continue

        # Q-block processing body: indent by +4 spaces
        if init_idx < i <= lse_end:
            if s == '':  # preserve empty lines
                result.append('')
            else:
                result.append(body_indent + s.lstrip())
            continue

        # After the Q-block body (past LSE end): nothing more to add
        # The wave loop end is appended after the loop

    # Append wave loop end after the last Q-block line
    result.append(wave_indent + '# end of wave loop iteration')
    return '\n'.join(result)


def validate(body):
    """Basic validation of generated template."""
    checks = [
        'for wave_w in range(WAVE_SIZE):',
        'wave_id = tl.program_id(0)',
        'wave_start = wave_id * WAVE_SIZE',
        '# end of wave loop iteration',
        '# initialize pointer to m and l',
    ]
    for c in checks:
        if c not in body:
            raise RuntimeError(f"Validation failed: missing '{c}'")

    # Check: original PERM block should NOT be outside the loop
    if '    if ENABLE_REORDER:\n        src_mask_block = tl.load(PERM' in body.replace('        if ENABLE_REORDER:', ''):
        # If found at 4-space (outside loop), it's wrong
        pass

    # Count indent levels: should only be 0, 4, 8, 12 (no 1,2,3,5,6,7,9,10,11)
    for i, line in enumerate(body.split('\n')):
        if line.strip() == '':
            continue
        indent = len(line) - len(line.lstrip())
        if indent not in (0, 4, 8, 12):
            print(f"  WARNING: unusual indent {indent} at line {i}: {line[:60]}")

    print("  Validation passed")
    return True


def main():
    dry_run = '--apply' not in sys.argv

    with open(FLEXPATH, 'r') as f:
        content = f.read()

    print("=== Extracting original template ===")
    body, start, end, closing_len = extract_template(content)
    print(f"  Template: {len(body)} chars, {body.count(chr(10))} lines")

    print("=== Generating wave variant ===")
    wave_body = generate_wave(body)
    print(f"  Wave template: {len(wave_body)} chars, {wave_body.count(chr(10))} lines")

    print("=== Validating ===")
    validate(wave_body)

    if dry_run:
        print("\n=== Dry run: showing key sections ===")
        wave_lines = wave_body.split('\n')
        for i, line in enumerate(wave_lines):
            if any(kw in line for kw in ['wave_id', 'wave_start', 'for wave_w',
                                          '# Wave-based', '# end of wave',
                                          '# initialize pointer',
                                          'if ENABLE_REORDER:']):
                # Show context
                start_i = max(0, i-1)
                end_i = min(len(wave_lines), i+3)
                for j in range(start_i, end_i):
                    marker = '>>>' if j == i else '   '
                    print(f"  {marker} L{j}: {wave_lines[j]}")
        print("\n  (use --apply to write changes)")
    else:
        print("=== Writing to file ===")
        # Build the new variable
        # Strip trailing whitespace from wave_body to avoid indented closing quotes
        wave_body_clean = wave_body.rstrip()
        wave_var = (
            "\n\n# Auto-generated wave-variant template.\n"
            "# See Flex_attn_wave_reorder_plan.md for details.\n"
            "compute_flex_attention_wave = r\"\"\"\n"
            + wave_body_clean +
            "\n\"\"\"\n"
        )
        insert_pos = end + closing_len  # right after '\n """'
        new_content = content[:insert_pos] + wave_var + content[insert_pos:]
        with open(FLEXPATH, 'w') as f:
            f.write(new_content)
        print(f"  Written. New file size: {len(new_content)} chars")

        # Verify Python syntax
        import py_compile
        try:
            py_compile.compile(FLEXPATH, doraise=True)
            print("  ✅ Python syntax OK")
        except py_compile.PyCompileError as e:
            print(f"  ❌ Syntax error: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
