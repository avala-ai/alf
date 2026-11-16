const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const { mkdtempSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const { test } = require("node:test");

function box(type, payload) {
  const bytes = Buffer.alloc(8 + payload.length);
  bytes.writeUInt32BE(bytes.length);
  bytes.write(type, 4, "ascii");
  payload.copy(bytes, 8);
  return bytes;
}

function icns(zeroSize) {
  const entry = box("icp4", Buffer.alloc(0));
  if (zeroSize) entry.writeUInt32BE(0);
  // ICNS puts its type before its size (the opposite of ISO image boxes).
  return Buffer.concat([
    Buffer.from("icns"),
    Buffer.from([0, 0, 0, 16]),
    entry.subarray(4, 8),
    entry.subarray(0, 4),
  ]);
}

function jxl(zeroSize) {
  // A 16x16 small-image codestream, following a four-byte sequence index.
  const partial = box("jxlp", Buffer.from([0, 0, 0, 0, 255, 10, 67, 0]));
  if (zeroSize) partial.writeUInt32BE(0);
  return Buffer.concat([
    box("JXL ", Buffer.from([13, 10, 135, 10])),
    box("ftyp", Buffer.from("jxl ")),
    partial,
  ]);
}

function heif(zeroSize) {
  const dimensions = Buffer.alloc(12);
  dimensions.writeUInt32BE(16, 4);
  dimensions.writeUInt32BE(16, 8);
  const spatial = box("ispe", dimensions);
  if (zeroSize) spatial.writeUInt32BE(0);
  return Buffer.concat([
    box("ftyp", Buffer.from("heic")),
    box("meta", Buffer.concat([Buffer.alloc(4), box("iprp", box("ipco", spatial))])),
  ]);
}

// Each package entry point bundles its own parser, including fromFile, which
// Docusaurus uses. A fix to the top-level CommonJS export alone is insufficient.
const entries = ["cjs", "esm", "file-cjs", "file-esm", "type-cjs", "type-esm"];
const fixtures = { icns, jxl, heif };
const child = `
  import { createRequire } from 'node:module';
  const require = createRequire(process.cwd() + '/package.json');
  const [entry, type, encoded, filename] = process.argv.slice(1);
  const fromFile = entry.startsWith('file-');
  const moduleName = entry.startsWith('type-')
    ? 'image-size/types/' + type
    : fromFile ? 'image-size/fromFile' : 'image-size';
  const parser = entry.endsWith('cjs') ? require(moduleName) : await import(moduleName);
  try {
    const bytes = Buffer.from(encoded, 'base64');
    const size = fromFile ? await parser.imageSizeFromFile(filename)
      : entry.startsWith('type-') ? parser[type.toUpperCase()].calculate(bytes)
      : parser.imageSize(bytes);
    console.log(JSON.stringify({ size }));
  } catch (error) {
    console.log(JSON.stringify({ error: error.message, name: error.name }));
  }
`;

for (const entry of entries) {
  for (const [type, fixture] of Object.entries(fixtures)) {
    test(`${entry}: ${type} rejects a zero-length entry and reads a valid image`, () => {
      const directory = mkdtempSync(join(tmpdir(), "alf-image-size-"));
      try {
        for (const malformed of [true, false]) {
          const bytes = fixture(malformed);
          const filename = join(directory, `image.${type}`);
          writeFileSync(filename, bytes);
          // A timer in the parser process cannot interrupt a blocked event loop.
          // Kill the isolated child if the regression returns, keeping CI bounded.
          const result = spawnSync(
            process.execPath,
            [
              "--input-type=module",
              "--eval",
              child,
              entry,
              type,
              bytes.toString("base64"),
              filename,
            ],
            { cwd: join(__dirname, ".."), timeout: 2000, maxBuffer: 8192 },
          );
          assert.ifError(result.error);
          assert.equal(result.status, 0, result.stderr.toString());
          const response = JSON.parse(result.stdout.toString());
          if (malformed) {
            assert.equal(response.name, "TypeError");
            assert.match(response.error, /^Invalid (ICNS|JXL|HEIF)/);
          } else {
            assert.equal(response.error, undefined);
            assert.equal(response.size.width, 16);
            assert.equal(response.size.height, 16);
          }
        }
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });
  }
}
