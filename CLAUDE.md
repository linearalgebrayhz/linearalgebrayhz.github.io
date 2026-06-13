# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jekyll 3.9 static site for an academic personal homepage, deployed via GitHub Pages. No Node.js — all tooling is Ruby-based.

## Commands

```bash
# Install dependencies
bundle install

# Serve locally with live reload
bundle exec jekyll serve

# Build without serving
bundle exec jekyll build
```

The built site is output to `_site/` (git-ignored). GitHub Pages deploys automatically on push to `main`.

## Architecture

### Content — edit these to update the site

| Path | Purpose |
|------|---------|
| `_data/profile.yml` | Primary personal info: name, bio, education, experience, social links |
| `_data/display.yml` | Feature flags — toggle sections on/off, set news item count, etc. |
| `_data/navigation.yml` | Top navigation bar links |
| `_data/authors.yml` | Author definitions used in publication cards |
| `_publications/` | Publication entries as Markdown files with YAML frontmatter, organized by year subdirectory |
| `_news/` | News/update entries as Markdown files |
| `_posts/` | Blog posts (standard Jekyll naming: `YYYY-MM-DD-title.md`) |
| `_showcase/` | Project showcase entries |

### Templates & Components

- `_layouts/` — page-level templates (`default.html`, `post.html`, `prompt.html`)
- `_includes/widgets/` — reusable UI components (profile card, publication card, news card, carousel, etc.) included from layouts and pages
- `index.html`, `publications.html`, `blog.html`, `courses.html`, `showcase.html` — top-level page files; each selects a layout and assembles widgets

### Assets

- `assets/css/global.css` — site-wide styles
- `assets/js/common.js` — shared JS
- `assets/js/bubble_visual_hash.js` — visual effect on the homepage
- `assets/images/` — photos, publication covers, badges

### Key Dependencies (all via CDN, no local install needed)

- Bootstrap 4.6 — layout and UI
- Font Awesome 6.5 + Academicons 1.9 — icons
- KaTeX 0.16 — math rendering in posts
- Masonry 4 + ImagesLoaded 5 — grid layout for showcase/publications

### Jekyll Configuration

`_config.yml` defines the three custom collections (`publications`, `news`, `showcase`) and enables `jekyll-email-protect` to obfuscate email addresses in rendered HTML.
