// @ts-check
/* eslint-env node */

const { themes } = require("prism-react-renderer");

const repositoryUrl = "https://github.com/avala-ai/alf";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "ALF",
  tagline: "Arrays in Local Frames — an open container for point-cloud sequences",
  favicon: "img/favicon.svg",

  // The Pages workflow supplies the configured origin and base path so canonical
  // links and the sitemap point to the actual host. Local builds use alf.dev.
  url: process.env.SITE_URL || "https://alf.dev",
  baseUrl: process.env.SITE_BASE_PATH || "/",

  // The docs cross-reference each other with relative paths, and a link that no longer
  // resolves is a documentation bug rather than a cosmetic one — fail the build on it.
  onBrokenLinks: "throw",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "throw",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: "docs",
          routeBasePath: "/",
          sidebarPath: "./sidebars.js",
          editUrl: `${repositoryUrl}/tree/main/website/`,
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
        // No analytics: the site collects nothing and loads nothing from a third party.
        gtag: undefined,
        googleTagManager: undefined,
      }),
    ],
  ],

  plugins: [
    // The specification is NOT copied into the site. `spec/` at the repository root is
    // the normative text — thirty-odd source comments across four languages cite it by
    // that path, and the README links to it — so the site mounts those files where they
    // are rather than keeping a second copy that drifts. Editing `spec/format.md`
    // publishes it; there is no sync step, because there is nothing to sync.
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "spec",
        path: "../spec",
        routeBasePath: "spec",
        sidebarPath: "./sidebarsSpec.js",
        editUrl: `${repositoryUrl}/tree/main/`,
      },
    ],
    [
      "@docusaurus/plugin-client-redirects",
      {
        // The spec's documents are its files; `/spec` itself is not one of them, and a
        // reader who types the shorter address should land on the normative text rather
        // than a 404.
        redirects: [
          { from: "/spec", to: "/spec/format" },
          { from: "/docs", to: "/guides/" },
        ],
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "ALF",
        logo: {
          alt: "ALF",
          src: "img/logo.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "guidesSidebar",
            position: "left",
            label: "Guides",
          },
          {
            type: "docSidebar",
            docsPluginId: "spec",
            sidebarId: "specSidebar",
            position: "left",
            label: "Specification",
          },
          {
            type: "docSidebar",
            sidebarId: "referenceSidebar",
            position: "left",
            label: "Reference",
          },
          {
            href: "https://avala.ai",
            label: "An Avala project",
            position: "right",
          },
          {
            href: repositoryUrl,
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Documentation",
            items: [
              { label: "Guides", to: "/guides/" },
              { label: "Specification", to: "/spec/format" },
              { label: "Support matrix", to: "/reference/" },
              { label: "Conformance suite", to: "/reference/conformance" },
              { label: "Refusals", to: "/reference/refusals" },
            ],
          },
          {
            title: "Source",
            items: [
              { label: "Repository", href: repositoryUrl },
              { label: "Releases", href: `${repositoryUrl}/releases` },
              {
                label: "Contributing",
                href: `${repositoryUrl}/blob/main/CONTRIBUTING.md`,
              },
              {
                label: "License",
                href: `${repositoryUrl}/blob/main/LICENSE`,
              },
            ],
          },
        ],
        copyright:
          'An <a href="https://avala.ai" style="color: inherit">Avala</a> project. ' +
          "The ALF specification and SDKs are licensed under Apache-2.0.",
      },
      prism: {
        theme: themes.github,
        darkTheme: themes.dracula,
        additionalLanguages: ["rust", "dart", "python", "bash", "json", "toml"],
      },
      colorMode: {
        respectPrefersColorScheme: true,
      },
    }),
};

module.exports = config;
