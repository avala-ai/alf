// Execute against the built ESM entry point, as an installed Node consumer does.
import assert from 'node:assert/strict';
import { readAlf, writeAlf, writeAli, writeAll, writeAlp, writeAlx } from '../dist/index.js';

const points = [12.34567, -23.45678, 34.56789, NaN, NaN, NaN, 12.45678, -23.56789, 34.67891];
const rotation = { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 };

for (const gzip of [false, true]) {
  for (const frameConvention of ['world-aligned', 'device-aligned']) {
    const file = await readAlf(
      await writeAlp(points, {
        codec: 'int16',
        geomScale: 0.00001,
        geomOffset: [12.4, -23.5, 34.6],
        frameConvention,
        ...(frameConvention === 'device-aligned' ? { frameRotation: rotation } : {}),
        metadata: { device_position: { x: 1000.125, y: -2000.25, z: 30.5 }, description: 'synthetic ✓' },
        gzip,
      }),
      { channel: 'alp' },
    );
    assert.equal(file.version, 3);
    assert.equal(file.codec, 'int16');
    assert.equal(file.totalPoints, 3);
    assert.equal(file.frameConvention, frameConvention);
    assert.equal(file.metadata.description, 'synthetic ✓');
    assert.deepEqual(file.metadata.device_position, { x: 1000.125, y: -2000.25, z: 30.5 });
    assert.equal(file.points.length, points.length);
    assert.ok(Number.isFinite(file.maxGeomErrorM) && file.maxGeomErrorM > 0 && file.maxGeomErrorM < 0.00001);
    for (let i = 0; i < points.length; i++) {
      if (Number.isNaN(points[i])) assert.ok(Number.isNaN(file.points[i]));
      else assert.ok(Math.abs(file.points[i] - points[i]) <= file.maxGeomErrorM);
    }
    if (frameConvention === 'device-aligned') {
      assert.ok(Math.abs(file.frameRotation.w - rotation.w) < 1e-15);
      assert.ok(Math.abs(file.frameRotation.z - rotation.z) < 1e-15);
      assert.equal(file.frameRotation.x, 0);
      assert.equal(file.frameRotation.y, 0);
    } else {
      assert.equal(file.frameRotation, null);
    }
    // Rewriting a decoded v3 file must keep its convention and invalid slots.
    const rewritten = await readAlf(await writeAlf(file, { gzip: !gzip }), { channel: 'alp' });
    assert.equal(rewritten.version, 3);
    assert.equal(rewritten.frameConvention, file.frameConvention);
    assert.deepEqual(rewritten.frameRotation, file.frameRotation);
    assert.deepEqual(rewritten.points, file.points);

    // Sidecars still describe original indices, including the invalid return.
    const intensity = await readAlf(await writeAli([11, 22, 33], { gzip }), { channel: 'ali' });
    const labels = await readAlf(await writeAll([7, 8, 9], { gzip }), { channel: 'all' });
    const mapping = await readAlf(
      await writeAlx([2, 5, 9], { alpTotalPoints: 3, sourceTotalPoints: 10, gzip }),
      { channel: 'alx' },
    );
    assert.deepEqual(Array.from(intensity.intensity), [11, 22, 33]);
    assert.deepEqual(Array.from(labels.categories), [7, 8, 9]);
    assert.deepEqual(Array.from(mapping.keptIndices), [2, 5, 9]);
    assert.equal(mapping.sourceTotalPoints, 10);
  }
}

const legacy = await readAlf(await writeAlp([1.5, -2.25, 3.75]), { channel: 'alp' });
assert.equal(legacy.version, 1);
assert.equal(legacy.codec, 'fp16');
assert.equal(legacy.frameConvention, null);
assert.deepEqual(Array.from(legacy.points), [1.5, -2.25, 3.75]);
await assert.rejects(writeAlp(points, { codec: 'int16' }), { code: 'writer-input-invalid' });

// A live pose may expose getters. Validate and emit the same sampled rotation;
// the conjugate is also a valid unit quaternion, so a reader cannot detect drift.
const poseReads = { w: 0, x: 0, y: 0, z: 0 };
const livePose = {};
for (const axis of ['w', 'x', 'y', 'z']) {
  Object.defineProperty(livePose, axis, {
    get: () => ++poseReads[axis] === 1 || axis === 'w' ? 0.525 : -0.525,
  });
}
const posed = await readAlf(await writeAlp(points, {
  codec: 'int16', frameConvention: 'device-aligned', frameRotation: livePose,
}), { channel: 'alp' });
assert.deepEqual(posed.frameRotation, { w: 0.5, x: 0.5, y: 0.5, z: 0.5 });
assert.deepEqual(poseReads, { w: 1, x: 1, y: 1, z: 1 });
console.log('Built ESM runtime: v3 precision, frame declarations, rewrites, and channel correlation passed');
