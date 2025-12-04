# CLAP Assembly Processing and Tokenization Analysis

## Overview

This document provides an in-depth analysis of how the CLAP (Contrastive Language-Assembly Pre-training) framework processes and tokenizes assembly code for machine learning applications. The analysis covers the preprocessing pipeline, tokenization strategy, and special token handling.

---

## 1. Assembly Preprocessing (`process_asm.py`)

### 1.1 Script Dependencies

The `process_asm.py` script requires:
- **IDA Pro modules** (commercial/free binary analysis tool):
  - `idautils` - Utilities for iterating over functions and instructions
  - `idaapi` - IDA Pro API functions
  - `idc` - IDA command functions
- **Python standard library**:
  - `re` - Regular expression operations
  - `json` - JSON serialization

### 1.2 The Rebasing Process

#### Purpose
The `rebase()` function converts address-based assembly to instruction-index-based assembly, making it position-independent.

#### Input Format
```python
{
    0x401000: "push rbp",
    0x401001: "mov rbp, rsp",
    0x401004: "jmp loc_40100a",
    0x40100a: "ret"
}
```

#### Output Format
```python
{
    "1": "push rbp",
    "2": "mov rbp, rsp", 
    "3": "jmp INSTR4",
    "4": "ret"
}
```

#### What Gets Rebased
The function **only processes jump instructions** (starting with `j`):
- `jmp loc_401050` → `jmp INSTR25`
- `jz $+0x10` → `jz INSTR30`
- `jne loc_401234` → `jne INSTR42`

#### What Stays Unchanged
- **Memory addresses in operands**: `mov rax, [0x404000]` (unchanged)
- **Immediate values**: `mov edx, 6` (unchanged)
- **Call instructions**: `call sub_401234` (unchanged - not starting with 'j')
- **Data references**: `lea rax, [0x404000]` (unchanged)

#### Patterns Matched
```python
loc_pattern = re.compile(r' (loc|locret)_(\w+)')  # Named labels
self_pattern = re.compile(r'\$\+(\w+)')           # Relative offsets
```

### 1.3 Processing Workflow

```
Binary File
    ↓
IDA Pro Analysis
    ↓
For each function:
    1. Get all instruction addresses (idautils.FuncItems)
    2. Get disassembly text (idc.GetDisasm)
    3. Build address→instruction dict
    4. Call rebase() to normalize jump targets
    5. Append to results
    ↓
Output: JSON array of functions
```

### 1.4 Output Structure

```json
[
    {"1": "instruction1", "2": "instruction2", "3": "instruction3"},  // Function 1
    {"1": "instruction1", "2": "instruction2"},                        // Function 2
    {"1": "instruction1", "2": "instruction2", "3": "instruction3"}   // Function 3
]
```

**Key characteristics**:
- Each array element = one function
- Keys are sequential string indices starting at "1"
- Each function's numbering resets independently

### 1.5 Accessing Processed Data

```python
import json

# Load processed assembly
with open('binary.json', 'r') as f:
    result = json.load(f)

# Access second instruction of first function
instruction = result[0]["2"]  # Note: string key "2"
```

---

## 2. Tokenization Strategy

### 2.1 Tokenizer Configuration

- **Type**: Custom `AsmTokenizer` (WordPiece-based, similar to BERT)
- **Vocabulary size**: 33,555 tokens
- **Case handling**: Lowercase (`do_lower_case: True`)
- **Subword tokenization**: Yes (uses `##` prefix for continuations)
- **Model**: `hustcw/clap-asm` on Hugging Face

### 2.2 Special Tokens

#### Standard Tokens
- `<s>` - Beginning of sequence (BOS/CLS)
- `</s>` - End of sequence (EOS/SEP)
- `<pad>` - Padding token
- `<mask>` - Mask token (for masked language modeling)
- `[UNK]` - Unknown token

#### Instruction Reference Tokens
**1,024 special tokens** for rebased jump targets:
```
INSTR1, INSTR2, INSTR3, ..., INSTR1024
```

These are **atomic vocabulary entries**, ensuring jump targets are single tokens:
```
'jmp INSTR25' → ['jmp', 'INSTR25']  ✅ Two tokens
'jmp loc_401050' → ['jmp', 'loc', '_', '4010', '##50']  ❌ Five tokens
```

This demonstrates why the `rebase()` function is critical.

---

## 3. Token Categories and Examples

### 3.1 Opcodes/Instructions

Common instructions are single vocabulary tokens:
```
'mov'  → ['mov']
'push' → ['push']
'jmp'  → ['jmp']
'call' → ['call']
'add'  → ['add']
```

### 3.2 Registers

All common x86-64 registers are dedicated tokens:
```
Vocabulary contains:
rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp,
eax, ebx, ecx, edx
```

Example:
```
'mov rax, rbx' → ['mov', 'rax', '[UNK]', 'rbx']
```
(Note: comma becomes `[UNK]`)

### 3.3 Operators and Symbols

Single-character operators are individual tokens:
```
Vocabulary contains: +, -, *, [, ], (, )
NOT in vocabulary: ,
```

Example:
```
'[rdi+rax*4]' → ['[', 'rdi', '+', 'rax', '*', '4', ']']
```

### 3.4 Hex Values

Split using WordPiece algorithm:
```
'0x404000'    → ['0x', '##404', '##000']
'0x12345678'  → ['0x', '##123456', '##78']
'0x20'        → ['0x', '##20']
'0x10'        → ['0x', '##10']
```

The prefix `0x` is a dedicated vocabulary token, hex digits use subword units.

### 3.5 Decimal Immediates

```
'6'   → ['6']      # Single digit in vocab
'42'  → ['42']     # Small number in vocab
'100' → ['100']    # In vocabulary
```

Common small numbers and powers of 2 are likely in the vocabulary.

### 3.6 Memory Addresses

Decomposed into structural components:
```
'[rdi+rax*4]'        → ['[', 'rdi', '+', 'rax', '*', '4', ']']
'[rbp-8]'            → ['[', 'rbp', '-', '8', ']']
'[0x404000]'         → ['[', '0x', '##404', '##000', ']']
'[rip+0x2004]'       → ['[', 'ri', '##p', '+', '0x', '##200', '##4', ']']
'[rbx+rcx*8+16]'     → ['[', 'rbx', '+', 'rcx', '*', '8', '+', '16', ']']
```

### 3.7 Function Names and Labels

Split by underscores and WordPiece:
```
'sub_401234'  → ['sub', '_', '4012', '##34']
'loc_401050'  → ['loc', '_', '4010', '##50']
'printf'      → ['printf']  # Known symbol
```

---

## 4. Complete Tokenization Examples

### Example 1: Simple Move
```
Instruction: "mov edx, 6"
Tokens: ['mov', 'edx', '[UNK]', '6']
```

### Example 2: Memory Access
```
Instruction: "mov ecx, [rdi+rax*4]"
Tokens: ['mov', 'ecx', '[UNK]', '[', 'rdi', '+', 'rax', '*', '4', ']']
```

### Example 3: Hex Immediate
```
Instruction: "add rax, 0x10"
Tokens: ['add', 'rax', '[UNK]', '0x', '##10']
```

### Example 4: Memory Address
```
Instruction: "mov rax, [0x404000]"
Tokens: ['mov', 'rax', '[UNK]', '[', '0x', '##404', '##000', ']']
```

### Example 5: Rebased Jump
```
Instruction: "jmp INSTR25"
Tokens: ['jmp', 'INSTR25']  ✅ Two tokens only
```

### Example 6: Function Call
```
Instruction: "call sub_401234"
Tokens: ['call', 'sub', '_', '4012', '##34']
```

### Example 7: Complex Addressing
```
Instruction: "lea rax, [rbx+rcx*8+16]"
Tokens: ['lea', 'rax', '[UNK]', '[', 'rbx', '+', 'rcx', '*', '8', '+', '16', ']']
```

---

## 5. Key Insights

### 5.1 Design Philosophy

1. **Semantic Structure Preservation**: Dedicated tokens for instructions, registers, and operators maintain semantic meaning
2. **Flexible Subword Units**: Rare symbols, addresses, and hex values use learned subword units
3. **Position Independence**: Rebasing converts control flow to instruction indices
4. **Context-Based Learning**: The model learns to interpret numeric values from context rather than preprocessing

### 5.2 Why Rebasing Matters

Without rebasing:
```
'jmp loc_401050' → ['jmp', 'loc', '_', '4010', '##50']  // 5 tokens
```

With rebasing:
```
'jmp INSTR25' → ['jmp', 'INSTR25']  // 2 tokens, INSTR25 is atomic
```

Benefits:
- **Reduced token count** for jump instructions
- **Position independence** - same code at different addresses has same representation
- **Semantic clarity** - jump targets are clear instruction references
- **Better model performance** - atomic tokens easier to learn

### 5.3 What Doesn't Get Normalized

Important: The following remain as raw text and are tokenized as-is:
- Hex values in operands (`0x404000`)
- Memory addresses (`[rip+0x2004]`)
- Immediate values (`100`, `0x20`)
- Data pointers
- String references

The model learns to interpret these from context during training.

### 5.4 The Unknown Token (`[UNK]`)

Commas (`,`) are not in the vocabulary, so they become `[UNK]`. This is intentional:
- The model learns to ignore/handle unknown tokens
- Instruction structure is clear without commas
- Reduces vocabulary size

---

## 6. Vocabulary Statistics

- **Total vocabulary entries**: 33,555
- **Instruction reference tokens**: 1,024 (INSTR1-INSTR1024)
- **Hex-related tokens**: 12 (including `0x`, `##0x`, etc.)
- **Bracket/parenthesis tokens**: 5 (`[`, `]`, `(`, `)`, `[UNK]`)
- **Address-related tokens**: 118+ (ptr, addr, offset variants)
- **Common registers**: All x86-64 general-purpose registers

---

## 7. Practical Usage

### 7.1 Loading and Using the Tokenizer

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "hustcw/clap-asm", 
    trust_remote_code=True
)

# Load preprocessed assembly
import json
with open("bubblesort.json") as f:
    asm = json.load(f)

# Tokenize
tokens = tokenizer([asm], padding=True, return_tensors="pt")
```

### 7.2 Data Flow Summary

```
Binary → IDA Pro → process_asm.py → JSON → Tokenizer → Model
         ↓                           ↓
    Analysis              Per-function rebased     
                          instruction indices
```

---

## 8. Requirements and Installation

### 8.1 For Assembly Processing

**Option 1: IDA Pro** (Commercial)
- Purchase from hex-rays.com (~$1,000+)
- Includes Python API

**Option 2: IDA Free**
- Free version with limitations
- Download from hex-rays.com/ida-free
- 64-bit only, no decompiler

**Option 3: Alternatives**
- Ghidra (free, open-source)
- radare2/rizin
- angr
- Capstone

### 8.2 For Using CLAP Model

```bash
pip install transformers torch
```

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
model = AutoModel.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
```

---

## 9. Conclusion

The CLAP assembly processing pipeline demonstrates a sophisticated approach to preparing binary code for machine learning:

1. **Rebasing** normalizes control flow while preserving data references
2. **WordPiece tokenization** balances vocabulary size with flexibility
3. **Special INSTR tokens** ensure efficient jump target representation
4. **Context-based learning** handles numeric values without hardcoded rules

This design enables the model to learn transferable representations of binary code that generalize across different compilation settings, addresses, and binary structures.

---

## References

- CLAP Paper: [arXiv:2402.16928](https://arxiv.org/abs/2402.16928)
- Hugging Face Models: [clap-asm](https://huggingface.co/hustcw/clap-asm) | [clap-text](https://huggingface.co/hustcw/clap-text)
- Repository: [Hustcw/CLAP](https://github.com/Hustcw/CLAP)
