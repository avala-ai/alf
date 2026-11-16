// @ts-check
/* eslint-env node */

// The sidebar for the docs instance mounted on the repository's `spec/` directory.
// It lives here, and not in the documents, so the normative text stays plain markdown
// with no site-only front matter in front of it.

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  specSidebar: [
    { type: "doc", id: "format", label: "Wire format" },
    { type: "doc", id: "support", label: "Implementation support" },
    { type: "doc", id: "adoption", label: "Adoption" },
  ],
};

module.exports = sidebars;
