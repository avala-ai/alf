import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import React from "react";

import styles from "./index.module.css";

// Entry i of every channel describes point i. Everything below is a consequence of
// that one sentence, which is why the channel table comes before the property grid.
const channels = [
  { suffix: ".alp", name: "Points", holds: "XYZ geometry, frame-local", payload: "6 bytes/point" },
  { suffix: ".ali", name: "Intensity", holds: "Normalized return strength", payload: "1 byte/point" },
  { suffix: ".alc", name: "Colour", holds: "Palette indexes", payload: "1 byte/point" },
  {
    suffix: ".all",
    name: "Labels",
    holds: "Semantic category, optionally + instance id",
    payload: "1 or 3 bytes/point",
  },
  {
    suffix: ".alx",
    name: "Indexes",
    holds: "Decimation sidecar into the source cloud",
    payload: "4 bytes/kept point",
  },
];

// Each point restates something the specification states. Nothing here claims a
// capability `spec/format.md` or the support matrix does not.
const properties = [
  {
    title: "One invariant, defended",
    body: "Entry i of every channel describes point i. A length mismatch does not corrupt one file — it silently misattributes every label and intensity reading after the gap, and the file stays perfectly readable while it does.",
  },
  {
    title: "A frame-local origin",
    body: "Coordinates are local to the frame, and the header carries the placement. Giving up global coordinates is what buys the precision back; ALF is not a geospatial archive, and LAS/LAZ/COPC own that job.",
  },
  {
    title: "Two codecs, identical 6 bytes",
    body: "Version 1 is float16, whose error scales with magnitude. Version 3 is int16 fixed point around the frame's own bounding-box midpoint, so error is uniform — 20× tighter on a centred ±50 m frame, 82× on an off-centre one.",
  },
  {
    title: "Stated, measured error bounds",
    body: "_geom_max_err_m is what the writer measured, not what the step size implies. Compare an accuracy threshold against that field; the suite runs the measurement rather than asserting it.",
  },
  {
    title: "Invalid returns are content",
    body: "A NaN component means a ray that did not come back. Readers preserve its index so labels, intensity, and color stay aligned with the geometry.",
  },
  {
    title: "It fails closed",
    body: "Where guessing would return plausible-but-wrong data, ALF refuses. Every refusal has a stable identifier that is the same string in every language, because 'both readers raised an error' is not agreement.",
  },
  {
    title: "Gzip by magic, not by name",
    body: "Readers detect compression from the 1f 8b bytes rather than a filename or a Content-Encoding header. Concatenated members are refused: the second envelope would otherwise become payload in the first.",
  },
  {
    title: "One format, four implementations",
    body: "Python, Dart, TypeScript and Rust are held to one shared corpus of 117 cases — 45 valid variants and 72 that each break exactly one rule and name the refusal it must produce.",
  },
];

const languages = [
  { name: "Python", to: "/guides/getting-started/read-python" },
  { name: "TypeScript", to: "/guides/getting-started/read-web" },
  { name: "Dart", to: "/guides/getting-started/read-dart" },
  { name: "Rust", to: "/guides/getting-started/read-rust" },
];

export default function Home() {
  return (
    <Layout
      title="ALF"
      description="Arrays in Local Frames: an open container format for annotation-grade point-cloud sequences. Index-correlated channels, a frame-local origin, two 6-byte geometry codecs with stated error bounds, and four SDKs held to one conformance suite."
    >
      <header className={styles.hero}>
        <div className="container">
          <h1 className={styles.heroTitle}>
            {/* Inline rather than an <img>, so the mark scales with the heading and
                the page still needs no request beyond the document itself. */}
            <svg
              className={styles.heroMark}
              viewBox="0 0 64 64"
              aria-hidden="true"
              focusable="false"
            >
              <g fill="none" stroke="#a422e1" strokeWidth="5" strokeLinecap="round">
                <path d="M42 32H56" />
                <path d="M39.07 24.93 48.97 15.03" />
                <path d="M32 22V8" />
                <path d="M24.93 24.93 15.03 15.03" />
                <path d="M22 32H8" />
                <path d="M24.93 39.07 15.03 48.97" />
                <path d="M32 42V56" />
                <path d="M39.07 39.07 48.97 48.97" />
              </g>
              <circle cx="32" cy="32" r="6" fill="#a422e1" />
            </svg>
            ALF
          </h1>
          <p className={styles.heroSubtitle}>
            An open container format for point-cloud sequences in a <strong>local</strong>{" "}
            coordinate frame, streamed one frame at a time. Each file is one typed array plus a JSON
            header — an <code>.alp</code> is an <em>array of local points</em>.
          </p>
          <p className={styles.heroDetail}>
            Annotation-grade clouds delivered to a browser: small files, a GPU-friendly layout, a
            frame-local origin, and measured error bounds in the header. Five index-correlated
            channels, two geometry codecs at identical 6 bytes per point, and SDKs in four languages
            held to one shared conformance suite. Apache-2.0.
          </p>
          <div className={styles.buttons}>
            <Link className="button button--primary button--lg" to="/guides/">
              Read the guides
            </Link>
            <Link className="button button--secondary button--lg" to="/spec/format">
              Read the specification
            </Link>
          </div>
        </div>
      </header>

      <main>
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Five channels, one invariant</h2>
            <div className={styles.tableWrap}>
              <table className={styles.channels}>
                <thead>
                  <tr>
                    <th>Suffix</th>
                    <th>Channel</th>
                    <th>Holds</th>
                    <th>Payload</th>
                  </tr>
                </thead>
                <tbody>
                  {channels.map((channel) => (
                    <tr key={channel.suffix}>
                      <td>
                        <code>{channel.suffix}</code>
                      </td>
                      <td>{channel.name}</td>
                      <td>{channel.holds}</td>
                      <td>{channel.payload}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className={styles.note}>
              Channels are <strong>index-correlated</strong>: entry <em>i</em> of every channel
              describes point <em>i</em>. That single invariant is what most of the validation in
              this format exists to defend — see <Link to="/guides/concepts">Concepts</Link>.
            </p>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Why the format looks the way it does</h2>
            <div className={styles.grid}>
              {properties.map((property) => (
                <div className={styles.item} key={property.title}>
                  <h3>{property.title}</h3>
                  <p>{property.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>SDKs</h2>
            <div className={styles.languages}>
              {languages.map((language) => (
                <Link className={styles.language} key={language.name} to={language.to}>
                  {language.name}
                </Link>
              ))}
            </div>
            <p>
              Support is recorded per feature and per boundary, not per language: the{" "}
              <Link to="/spec/support">support matrix</Link> records only what the{" "}
              <Link to="/reference/conformance">conformance suite</Link> proves, and code existing
              without that evidence is marked unverified rather than counted. The Rust crate is a
              reader, so it earns the reader gate and is <code>n/a</code> for the writer one.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
