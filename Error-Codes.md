# Error codes

Every TEX diagnostic carries a stable `ENNNN` (error) or `WNNNN` (warning) code and links here. Codes are grouped by compiler/runtime phase (the first digit). The exact message is written for your specific program at the point of failure; this page explains the *class* and how to approach it. **This page is generated** (`tools/gen_error_codes.py`) — do not edit by hand.

## Internal (`E0xxx`)

Internal errors — a compiler phase raised without a structured diagnostic, so TEX synthesized a fallback (E0000) rather than surface a bare message. Rare and not your fault; please file an issue with the program that triggered it.

### E0000

Internal. See the message shown with the code for the specific cause and fix; the class is described above.

## Lexer (`E1xxx`)

Tokenization errors — a character or literal the scanner can't read (bad number, unterminated string/comment, stray symbol).

### E1000

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1001

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1002

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1003

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1004

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1005

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1006

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1007

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

### E1008

Lexer. See the message shown with the code for the specific cause and fix; the class is described above.

## Parser (`E2xxx`)

Grammar errors — a statement or expression that doesn't parse (missing `;`/`)`/`}`, a misplaced token, an unexpected keyword).

### E2000

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2001

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2002

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2003

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2004

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2005

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2006

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2010

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2011

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

### E2020

Parser. See the message shown with the code for the specific cause and fix; the class is described above.

## Type checker (`E3xxx`)

Type errors — an operation on the wrong type (vec/scalar/matrix/string/array mismatch, bad swizzle, wrong argument type or arity, assigning to a built-in, indexing a non-array).

### E3000

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3001

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3002

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3003

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3010

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3011

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3012

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3013

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3014

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3100

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3101

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3102

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3103

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3200

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3201

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3202

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3203

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3204

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3300

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3301

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3302

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3303

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3400

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3401

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3402

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3500

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3600

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3601

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3700

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3800

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

### E3900

Type checker. See the message shown with the code for the specific cause and fix; the class is described above.

## Optimizer (`E4xxx`)

Errors surfaced while optimizing the checked AST (rare; usually indicates an internal invariant — please file an issue with the program).

### E4000

Optimizer. See the message shown with the code for the specific cause and fix; the class is described above.

## Compile / cache (`E5xxx`)

Errors compiling or loading a cached program (codegen fell back, a cache artifact was rejected).

### E5001

Compile / cache. See the message shown with the code for the specific cause and fix; the class is described above.

### E5002

Compile / cache. See the message shown with the code for the specific cause and fix; the class is described above.

### E5003

Compile / cache. See the message shown with the code for the specific cause and fix; the class is described above.

## Runtime / node (`E6xxx`)

Errors while executing or wiring the node (an input `@X` isn't connected, a runtime value went non-finite, a fused-chain or lazy-input problem, an OOM the node re-raised for ComfyUI to handle).

### E6000

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6001

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6002

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6003

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6004

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6005

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6006

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6010

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6020

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6021

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6030

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6040

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6050

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6051

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

### E6060

Runtime / node. See the message shown with the code for the specific cause and fix; the class is described above.

## Tools (`E9xxx`)

Tool errors — building or preflighting a `.textool` bundle failed (a fused-tool graphspec is malformed, or its stages don't compile together). Check the tool's stages and manifest; the message names the stage that broke.

### E9001

Tools. See the message shown with the code for the specific cause and fix; the class is described above.

## Warnings (`W7xxx`)

Non-fatal advisories (LANG-2). The program still compiles and runs; these flag likely mistakes — an unused variable or wired input, or a name that shadows a built-in or an outer-scope variable.

### W7001

Warnings. See the message shown with the code for the specific cause and fix; the class is described above.

### W7002

Warnings. See the message shown with the code for the specific cause and fix; the class is described above.

### W7003

Warnings. See the message shown with the code for the specific cause and fix; the class is described above.

### W7004

Warnings. See the message shown with the code for the specific cause and fix; the class is described above.

### W7005

Warnings. See the message shown with the code for the specific cause and fix; the class is described above.
