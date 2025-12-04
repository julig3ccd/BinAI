# Category-Based Masking Approach for CLAP Assembly

## Pipeline Overview

```
Binary → IDA Pro → process_asm.py → Rebased JSON → Normalize → Mask → Tokenize → Model
```

## Step-by-Step Approach

### Step 1: Load Rebased Assembly
Load the JSON file containing rebased assembly where jump targets are already converted to `INSTRN` format.

```python
# Structure: {"1": "mov edx, 6", "2": "xor eax, eax", ...}
```

### Step 2: Define Category Patterns
Create regex patterns for each token category you want to mask (registers, immediates, hex values, memory addresses, etc.).

```python
patterns = {
    'register': r'\b(rax|rbx|rcx|rdx|...)\b',
    'immediate': r'\b\d+\b',
    'hex_value': r'0x[0-9a-fA-F]+',
    'memory': r'\[([^\]]+)\]',
}
```

### Step 3: Normalize Whitespace
Apply `re.sub(r'\s+', ' ', instruction).strip()` to collapse multiple spaces and ensure consistent formatting across all instructions.

### Step 4: Apply Category-Specific Masking
For each instruction, use `re.sub(pattern, '<mask>', instruction)` to replace matched tokens with the `<mask>` special token for selected categories.

```python
# Example: "mov edx, 6" → "mov <mask>, 6" (register masked)
```

### Step 5: Tokenize with CLAP Tokenizer
Pass the masked function dictionary to the pretrained tokenizer, which converts `<mask>` to token ID 5 and tokenizes everything else normally.

```python
asm_input = tokenizer([masked_function], padding=True, return_tensors="pt")
```

### Step 6: Feed to Model
Send the tokenized input to the CLAP model, where masked positions have the mask token and all other positions are processed normally.

## Key Points

- **Whitespace**: Token IDs remain consistent regardless of whitespace variations, but normalize for clean, reproducible code
- **Order matters**: Apply category masks sequentially; overlapping patterns may interfere
- **Atomic `<mask>`**: The mask token (ID=5) is never split by WordPiece tokenization
- **Category flexibility**: Choose which categories to mask based on your analysis goal (e.g., mask registers to study control flow)

## Example Output

```
Original: mov     edx, 6
Masked:   mov <mask>, 6
Tokens:   ['mov', ' <mask>', '[UNK]', '6']
Token IDs: [1159, 5, 4, 1050]
```
