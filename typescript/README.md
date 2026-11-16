# TypeScript

The TypeScript implementation of ALF — Arrays in Local Frames.

## Layout

- `alf/` is the publishable `@avala-ai/alf` package. Its runtime is pure
  TypeScript over `Uint8Array` with pinned, browser-safe gzip support.
- `conformance/` is private repository tooling. `read.ts` implements the shared
  canonical-reader contract, and `roundtrip.ts` exercises cross-language encode
  compatibility.

Application-specific label merging and sidecars belong in the consuming
application. This implementation supports the five channels defined by ALF.

## Checks

```bash
cd typescript/alf
bun install --frozen-lockfile
bun run build
bun run lint
bun test

cd ../..
python3 tests/conformance/generate.py --verify
python3 tests/conformance/run.py --runner "bun run typescript/conformance/read.ts"
python3 tests/conformance/encode_roundtrip.py \
  --encoder "bun run typescript/conformance/roundtrip.ts"
```

CI runs these reader and independent Python-decoded writer gates on both Bun
1.2.14 and 1.3.11, then imports the built ESM package in Node. Ordinary CI
verifies the checked-in corpus; regeneration is an explicit contract-update
operation.
