import { readFile } from 'node:fs/promises';

const bytes = await readFile(new URL('../target/wasm32-unknown-unknown/release/examples/portable_runtime.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(bytes);
instance.exports.verify_counts();
instance.exports.verify_geometry_bounds();
console.log('Rust Wasm: exact wide counts, payload-length refusals, and geometry error bounds passed');
