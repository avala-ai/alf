# Website

The documentation site, published to <https://alf.dev/> by
[`.github/workflows/website.yml`](../.github/workflows/website.yml) on every push to `main` that
touches this directory.

## Running it

```sh
cd website
npm ci
npm start          # http://localhost:3000/
npm run build      # static output in website/build/, which is not committed
```

Node 20 or newer. The site is self-contained — it is not a workspace of anything, so there is no
root install step, and it does not interact with the Bun setup under `typescript/`.

The build fails on a broken internal link rather than publishing one, so a renamed or moved page
must be fixed everywhere it is referenced.

The build runs its regression tests before compiling the site; they keep a malformed-image hang in
the `image-size` parser Docusaurus depends on from coming back with a dependency update.

## Where the content lives

Two docs sources, and the split is deliberate.

**`website/docs/`** is site content: the guides and the reference material. It is authored here and
read here.

**`spec/`, at the repository root, is not copied.** It is mounted by a second docs plugin instance
(`id: "spec"`, `path: "../spec"`) and served at `/spec/`. That directory is the normative text —
around thirty source comments across four languages cite it by that path, the README links to it,
and `AGENTS.md` and `CONTRIBUTING.md` point at it. Keeping a second copy under `website/` would give
the format two specifications that drift. Editing `spec/format.md` publishes it; there is no sync
step, because there is nothing to sync.

The consequence for links: within `website/docs/` use relative `.md` links, which resolve both on
the site and on GitHub. Linking **to the spec** from a guide has to use a site-absolute path
(`/spec/format`), because the two directories are not relatives of each other on disk. The spec's
own internal links stay relative and keep working in both places.

## Conventions

- **Navigation lives in `sidebars.js` and `sidebarsSpec.js`**, listed explicitly. Ordering and
  sidebar labels are set there so the documents themselves stay plain markdown with no front matter
  — which is what lets the normative spec be served here without being modified for the site.
- **The support matrix has one copy.** It is `spec/support.md`, served at `/spec/support`. The
  Reference section deliberately does not restate it: a support table that exists in two places is a
  support table that is wrong in one of them.
- **Nothing is loaded from a third party** — no analytics, no fonts, no scripts, no images from
  anywhere but `static/`. The site works offline and behind a proxy.
- **Light and dark are both supported** and follow the reader's system preference by default.
- **The palette is Avala purple** (`#a422e1`), on the same neutrals the
  [4dgs](https://4dgs.dev) site uses, so the two read as one family. Neither end of the scale is the
  raw brand token, and both departures are contrast — see the comment in `src/css/custom.css`.

## Deploying to alf.dev

The workflow builds on every pull request and deploys only from `main`. Deployment uses GitHub
Pages' Actions flow (`upload-pages-artifact` → `deploy-pages`), so the custom domain is configured in
the repository's **Settings → Pages** rather than by a `CNAME` file in `static/`, and DNS for
`alf.dev` must point at GitHub Pages. This mirrors how `4dgs.dev` is set up.
