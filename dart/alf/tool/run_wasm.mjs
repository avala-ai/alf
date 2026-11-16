import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

// Execute the loader emitted beside `dart compile wasm ... -o <file>.wasm`.
// A compile alone cannot catch differences in Dart's numeric or gzip runtime.
const [wasmPath] = process.argv.slice(2);
if (!wasmPath || !wasmPath.endsWith('.wasm') || process.argv.length !== 3) {
  throw new Error('Usage: node tool/run_wasm.mjs <file.wasm>');
}
const wasmUrl = pathToFileURL(resolve(wasmPath));
const loaderUrl = new URL(wasmUrl.href.replace(/\.wasm$/, '.mjs'));
const { compile } = await import(loaderUrl.href);
const application = await (await compile(await readFile(wasmUrl))).instantiate();
application.invokeMain();
