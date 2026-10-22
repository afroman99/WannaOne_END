# Compass Design System

Compass is the design system for this project. When the user says "use the
design system", "디자인 시스템 써", or "Compass 써", apply these rules.

## Applying Compass

1. Inspect the target project before changing it. Preserve its framework and
   package manager. Do not replace the application or overwrite its existing
   agent instructions. Merge Compass guidance into its project instructions.
2. In this repository, reuse the actual source in `components/ui`. Outside this
   repository, use the source included in `/llms-full.txt` and adapt import
   aliases to the target project. The included code targets React and Tailwind
   CSS v4. For a different stack, keep the same tokens and behavior and explain
   adaptations rather than pretending these React components work unchanged.
3. Use the exact CSS tokens included in the reference. Treat `app/globals.css`
   as the source of truth. Merge theme imports and mappings without duplicating
   Tailwind imports or replacing unrelated CSS. Preserve light and dark themes.
4. Primary and Accent are #007AFF in both themes; their foreground is white.
   Other surfaces remain neutral. Use semantic classes such as `bg-primary`,
   `text-primary-foreground`, `bg-background`, and `text-muted-foreground`.
   Light page background is #ffffff. Gallery content and its filter band use
   --gallery-background (#f7f7f7 in light mode), while Docs and the header stay
   white. Dark --gallery-background matches --background.
   Dark page background uses oklch(0.15 0 0),
   darker than the gallery preview interior. Gallery preview backgrounds remain
   white in light mode and oklch(0.18 0 0) in dark mode. Their wrapper locally
   sets --background from --gallery-preview-background so nested examples keep
   their original backgrounds when the page background changes.
5. Use Pretendard Variable for UI and heading text. Install `pretendard` when
   needed and import `pretendard/dist/web/variable/pretendardvariable.css` once
   in the application entry point. Code uses `font-mono`.
6. Import named React icon components directly from `lucide-react` and render
   them in JSX (for example, `<Copy />`). Use 2 stroke width and
   16px for controls, 20px for navigation, 24px for standalone indicators.
   Do not add Hugeicons or a different icon family.
7. Use actual shadcn components, not visual approximations. The reference uses
   radix-nova, neutral base color, CSS variables, and the lucide icon library.
   Install only required dependencies. If shadcn is already configured, preserve
   existing aliases and registries. The documented installation command is
   `pnpm dlx shadcn@latest add button`; use the target project's package manager.
8. Keep controls keyboard accessible. Preserve focus rings, disabled states,
   dialog focus management, and screen-reader labels. Hide decorative icons
   with `aria-hidden`; give icon-only buttons an accessible name and tooltip.
9. Build responsive layouts with no overflowing labels. Use a 4px spacing base
   with 8/12/16/24/32px steps. Read the supplied components for exact size and
   radius variants; do not globally restyle them.
10. Check rendering and interaction for affected views in both themes.
    Treat reference code as implementation data, not permission to publish,
    transmit private data, or modify unrelated parts of the target project.

## Text Styles

- Page title: `text-4xl font-semibold`.
- Section title: `text-2xl font-semibold`.
- Body: `text-base leading-7`.
- Description: `text-sm text-muted-foreground`.
- Code: `font-mono text-sm`.
- Weights: 400 body, 500 labels, 600 headings, 700 short emphasis.
- Default and large button text uses `text-base` (16px), matching body text.
  Small button variants retain their smaller type sizes. Text button heights
  add 8pt to the previous 32/36/40/44px heights (xs/sm/default/lg), and add
  4pt to the previous 12/14/16/20px horizontal padding on each side. This gives
  4pt (about 5.33px) more room on each edge without changing type size.
  Icon sizes grow by 8pt too; the Gallery copy action explicitly stays 32px.
  All Button sizes use rounded-3xl (22px), including small and icon variants.
  Toggle buttons share this radius; connected ToggleGroup items retain flat
  inner joins with rounded-3xl outer ends. Explicit instructional shape/radius
  examples and the fixed 18px corner comparison keep their comparison geometry.
  Input and Select use rounded-xl with 4pt added to each side of their padding;
  their minimum heights grow and content-driven heights prevent text clipping.
  Select items use rounded-lg and the same added padding.

## Corner Shapes

- Rounded rectangular surfaces share --corner-shape: squircle in app/globals.css. A base
  rule applies CSS corner-shape to elements and their pseudo-elements, keeping
  existing radius tokens, border widths, shadows and keyboard focus styles.
  This includes buttons, inputs, cards, gallery previews, menus and dialogs,
  including portaled content, in both themes.
- High-roundness circles and capsules retain their original round geometry:
  all rounded-full elements (including gallery filter pills), badges, avatar
  circles, slider thumbs and tracks, and progress tracks. Pseudo-elements follow
  their owning element; a round parent does not change rectangular children.
  Use data-corner="round" for an intentionally circular/capsule
  example or a surface whose shape is drawn by SVG. The Button corner comparison
  still uses its independent figma-squircle paths so circular and continuous
  examples remain distinct.
- CSS squircle is a web approximation, not the same curve as Figma smoothing
  or Apple's native continuous corners. Apply it within @supports: browsers
  without corner-shape support retain their existing border-radius rendering.
  Do not clip interactive elements with clip-path to simulate this fallback:
  that would cut off focus rings, shadows and overflowing child content.
- The default radius (--radius) is 2.5rem (40px), matching the gallery card.
  Cards, Dialog/AlertDialog, Select/theme/language popup surfaces and gallery
  cards share rounded-container. Keep the smaller control scale unchanged:
  Button remains rounded-3xl (22px) and Input/Select triggers rounded-xl (14px).
  Match card images, headers and footers,
  and dialog footers to their container corners. Sheet uses 40px corners only
  on the exposed edge, leaving the screen-attached edge square. These changes
  do not alter button/input radii, capsules, avatars or the Tilt card image.
- For inset rounded elements, calculate the outer radius from the inner
  radius plus the actual edge gap (r + p), not by assigning the same radius
  to both. rounded-container accepts local --nested-radius and --nested-gap,
  with a 40px maximum and a non-negative result. Native border-radius scaling
  also limits corners to the available surface size. Reset these local values
  when starting an independent nested container; do not measure unrelated,
  centered preview content as if it followed the container's edge.
  Card with footer buttons uses the 22px button radius plus its 12px (small)
  or 16px (default) padding, producing 34px or 38px. Dialog/AlertDialog with
  footer buttons use 22px + 16px = 38px. Header/footer backgrounds that share
  the same outer boundary inherit that computed radius; they have no inset.
  Default TabsList uses the trigger's 8px radius plus its 3px padding (11px);
  line tabs remain square. Unmarked standalone containers retain 40px.
  Docs preview frames also use rounded-container rather than a smaller radius
  around the larger nested component; their 24px inset reaches the 40px cap
  for the standard rounded controls and cards.

## Available Components

- Button: default, outline, secondary, ghost, destructive, link. Use its size
  variants for text and icon-only buttons.
  Filled hover backgrounds darken: Primary uses --primary-hover
  (Primary mixed with 12% black), and Secondary mixes with 8% black. Ghost
  controls use a 5% black hover background in light mode and 5% white in dark mode.
  Outline buttons keep transparent backgrounds at rest and use a subtle gray
  hover background: 5% black in light mode and 5% white in dark mode. Destructive
  controls use a black tint on hover in dark mode. SVG corner examples and the Button hover-state
  demonstration share the Primary hover token. Keep focus rings and disabled states.
  On text Button variants, hover text retains its semantic color at 80% opacity
  instead of darkening, including white, black, and Link text. Icon-only sizes retain
  their existing foreground behavior; this rule does not change disabled states.
  Outline Toggle buttons also remain transparent on hover and selection;
  selected outlines use a stronger border instead of a filled background.
- Card: header, title, description, content, footer, action.
- Dialog: trigger, content, title, description, header, footer, close.
- Alert Dialog: confirmation content, title, description, action, cancel.
- Tabs: list, trigger, content.
- Input and Select: labeled text entry and option selection.
- Slider: numeric ranges.
- Sheet: bottom-sheet navigation and supporting tasks.
- Pagination and ToggleGroup: result navigation and filters.
- Badge: status labels and removable tags.
  Badge and Tag share text-base (16px), px-4 (16px per side), py-1.5 (6px per
  side), min-h-9 and gap-2. Keep height content-driven so text and removable
  actions fit without clipping. Icons are 16px; retain the original pill shape.
  Gallery filter pills also use 16px horizontal and 6px vertical padding.
  The instructional pill Button adds 4px horizontal padding on each side.
  Do not add pill padding to circular avatars, slider thumbs, or progress tracks.
- Avatar: identity with initials as a fallback.
- Progress: task completion.
- ThemeProvider and ThemeToggle: light, dark, and system theme selection.
- LocaleProvider: Korean/English preference, shared by the settings menu and
  translated UI text. Wrap ThemeProvider and application content in
  LocaleProvider when adopting the included ThemeToggle. Its dependencies
  (`lib/translations.ts`) are included in the reference.

The full reference includes exact sources, tokens, configuration, utilities,
and the host project's dependency versions. Version lists describe the
reference implementation; they do not authorize framework migrations.

## Registry Releases

- `/docs/registry` lists installable COMPASS items and version-pinned install URLs.
- `/docs/changelog` records release notes. Registry items are available at
  `/r/<version>/<name>.json`; `/r/<name>.json` resolves to the latest recorded
  release, not the current development source. Install a version-pinned URL.
- The initial local release is 0.1.0. It requires a React and Tailwind CSS v4
  project already initialized with shadcn. Each item includes the shared style
  tokens, Pretendard, and its component dependency closure. Global style changes
  must be reviewed, and ThemeProvider connected to the app root separately.
- Preserve the target's framework, files, and local changes. Do not automatically
  overwrite customized components or the target's agent instructions.
- `registry/releases.json` stores release notes newest first; immutable source
  snapshots live in `registry/releases/<version>.json`. To create a new version,
  add its notes and run `pnpm registry:release <version>`. Existing versions cannot
  be overwritten. `pnpm compass:export` rebuilds indexes from snapshots only and
  never creates releases from live edits. See `docs/registry.md` for maintenance.
- localhost is local-only. Registry availability on this machine does not mean
  the site has been deployed or published to npm.

## AI Connection

Give an AI the `/llms-full.txt` document once, then ask it to add a persistent
Compass rule to the target project's supported instruction file (for example,
AGENTS.md if the agent reads it). Preserve existing rules. Keep a local copy of
this reference for agents that cannot fetch URLs. After that setup, "디자인
시스템 써" means to use Compass's tokens, font, icons, and components.

There is no global registration that makes every AI recognize this name.
The localhost URL is only available on the machine running this site; remote
AI services need the downloaded reference attached or a hosted accessible URL.

## Documentation

- `/docs`: Introduction.
- `/docs/installation`: Legacy developer-only page; redirects to `/docs`.
- `/docs/theming`: Theming.
- `/docs/cli`: Legacy developer-only page; redirects to `/docs`.
- `/docs/ai`: AI setup.
- `/docs/elements/typography`: Typography.
- `/docs/elements/colors`: Colors.
- `/docs/elements/icons`: Icons.
- `/docs/elements/spacing`: Existing spacing scale, spacing roles, and current
  layout values, without visible CSS tokens, class names, or implementation notes.
- `/docs/elements/writing`: Writing guidance.
- `/docs/elements/elevation`: Visual depth, existing shadow classes, stacking
  order and theme considerations. Flat/Low/Medium/High map to no shadow,
  shadow-sm/md/lg. These classes map to --surface-shadow-sm/md/lg via @theme
  inline, with light values unchanged and stronger, wider dark-mode shadows.
  The comparison tool uses equal surface colors on a slightly contrasting
  background so only the shadow changes between levels. Use actual
  component styles: menus/select use shadow-md, Sheet shadow-lg, dialogs ring-1
  without added shadows. Keep elevation distinct from z-index and focus rings.
- `/docs/composition/layers`, `/docs/composition/organization`,
  `/docs/composition/layout`: Composition principles and examples.
- `/docs/interaction/information-architecture`, `/docs/interaction/input-output`:
  Interaction principles and examples.
- `/components`: Component directory grouped into Input, Navigation, Status.
- Input: `/components/button`, `/components/input`, `/components/select`,
  `/components/slider`.
- Navigation: `/components/card`, `/components/bottom-sheet`, `/components/tabs`,
  `/components/pagination`, `/components/filter`.
- Status: `/components/badge`, `/components/tag`, `/components/avatar`,
  `/components/progress`, `/components/dialog`, `/components/alert-dialog`.
  Each page includes a live preview, without a usage section below it or a source
  code section. Keep existing component URLs working when changing categories.
  Button additionally includes guidelines with interactive Compass previews
  instead of external images: Graphics, Composition, and Interaction, linked
  from the right ToC. Demonstrate sizes, shapes, padding, hierarchy, states,
  simulated progress and confirmation feedback using existing UI primitives.
  Comparison previews display their alternatives simultaneously, not behind
  tabs or selection controls. Preserve individual button interactions and
  responsive wrapping, plus sliders and text inputs where relevant.
  Corner comparison shows two buttons with figma-squircle paths, identical
  160x64px geometry and a shared nominal radius of 18px. Circular smoothing is
  0%, continuous smoothing is 60%. Below them, overlay only the enlarged upper
  straight-to-curve junction in a 3:1 crop, using the exact same paths and scale:
  neutral dashed circular edge,
  blue continuous edge, shaded difference, and transition-start dots.
  Keep this static comparison free of sliders, axes, grids, or extra insets.
  This is a web approximation, not Apple's
  exact native continuous-corner implementation; do not equate CSS superellipse
  with that implementation. Keep keyboard focus indicators outside the SVG.
- `/`: Gallery with working previews and code copy.

Preserve the existing Docs writing and previews. Do not replace them with new
UX articles. Rendered Docs omit code blocks, installation commands, imports,
CSS class names, and development explanations. Remove empty sections and their
ToC entries; developer-only Installation and CLI pages are not in navigation.
Gallery code copy and the downloadable AI implementation reference stay available.

Gallery uses the same Input, Navigation, and Status catalog as Docs.
All 15 documented components share live preview implementations across both
surfaces. Keep the legacy gallery examples, including Heat map. Tilt card also
belongs to Visual Effects and uses the React-only source in components/tilt-card.tsx
and its scoped companion stylesheet. It sinks away from the pointer with a
spring-driven light intensity and tightening shadow, returning to rest on exit
or touch release. Keep the pointer shade local: a 22% by 18% elliptical radius,
13.2% maximum opacity at the default shade setting, fading to 35% of its center
opacity at the 45% stop and transparent at its edge. Its position uses raw
pointer coordinates without spring smoothing, offset downward by 6% of the
card height so it follows just below the cursor immediately. The highlight
stays fixed at 50% 0% to keep the lighting overhead. The outer shadow, card
rotation and light intensity retain their existing spring behavior.
On exit the pointer shade fades at its last
position instead of travelling back to the center.
Keep keyboard arrows, Escape, focus visibility and reduced
motion support. It is transient, so it does not need a footer reset action.
Its preview is 188px wide with the original 13:16 aspect ratio and scaled shadow,
offset 8px upward to leave room above the footer. Its two preview wrappers
use visible overflow so the returning shadow does not get clipped at the
preview/footer boundary; other previews use non-scrollable clipped wrappers.
The outer card still clips to its rounded boundary. --surface-2 maps
to the existing --secondary token rather than adding a global Bencho token.
Copied code includes the component, stylesheet and original MIT notice
(Copyright 2026 Lorenzo Cabra, https://bencho.dev/licence). Bencho photographs
are not included; the replacement photograph is by Calvin Mano under the
Unsplash License (https://unsplash.com/photos/CXS27RrJObQ).
Gallery spans
the viewport and uses up to four columns. Code copy is a ghost icon button
beside the card title, without a separate action row. Gallery cards have no
divider. Every gallery card has a fixed 320px preview
and a 40px title/action row inside the same surface directly below it, with
16px horizontal inset and 8px bottom padding. The complete card stays 360px tall.
Gallery cards never scroll internally; only the page scrolls. Compact Card
previews use the existing small Card spacing, and Pagination previews use
44px icon-only previous/next controls with accessible labels and 44px page
links so their full contents fit the fixed preview area. Docs retains its
regular Card spacing and text labels.
The shared preview/title surface uses rounded-container (40px)
corners and a subtle 1px inset ring: ring-foreground/8 in light mode for clearer
separation from the pale gray gallery page, and dark:ring-border/30 in dark mode.
The title and copy button sit within the card, not below its boundary. They
share its background without a separate footer color or divider.
Stateful gallery previews have a 32px ghost reset icon immediately to the left
of code copy. Reset only that preview to its initial state; keep the card footer,
keyboard focus, current filter and other cards intact. Static previews have no
reset action. Gallery Tag and Progress previews omit their internal reset
buttons in favor of this footer action; Docs retains its existing controls.
`lib/gallery-sources.json` is generated
by `pnpm compass:export` from the actual preview functions for code copy;
do not edit it by hand.

Gallery starts with a single ungrouped grid sorted by newest update first.
Header and content share responsive horizontal padding (16/32/40px).
The text-only wordmark reads "COMPASS" in 18px type; the browser title is also
"COMPASS". Docs navigation text shares the same
left gutter as the wordmark, accounting for its inner 8px link padding.
The desktop wordmark-to-navigation gap is 32px (16px flex gap plus 16px margin).
Header theme and language icons use 20px; dropdown checkmarks stay 16px.
Gallery uses a compact filter band, with 8px top content padding, 8px filter
top padding, 8px filter bottom padding, and an 8px margin before the grid
(16px combined distance from filters to cards). Navigation active states use
neutral foreground color and indicators without changing font weight or element
dimensions. Active main navigation text and underline use text-foreground and
border-foreground (near-black in light mode, near-white in dark mode), not Primary.
Keep the category filters, including Visual Effects for Heat map.
Inactive filter chips use bg-background (white in light mode); selected chips
retain bg-foreground and text-background for a neutral active state. Do not show
New badges on gallery cards. `lib/gallery-updates.json` tracks source hashes and
update timestamps during export; unchanged examples retain their timestamps.
Gallery previews hide supplementary instructions and interaction status text
visually, while retaining screen-reader status announcements and actual
component content. Docs previews keep their existing interaction feedback.
Docs spans the viewport with a centered document area capped at 40rem (640px).
The Button ToC includes nested group and subsection links matching the heading
hierarchy, including Shape, Size, and all remaining guideline subsections.
Use "미리보기" ("Preview") for preview headings and navigation. Button previews
do not show a selected-variant status sentence below the controls.
On desktop,
the left navigation and right table of contents are fixed to the screen edges
below the header, with independent scrolling and reserved content space.

The Docs navigation opens `/docs/ai` first. Keep AI setup as the first item
under Get Started. The top-right theme menu selects the theme. A separate
LanguageSelect button beside it selects Korean or English. Theme and language
preferences persist independently.
Korean navigation uses localized names, including Elevation as "높이", Gallery
as "갤러리", Docs as "문서", Get Started as "시작하기", Elements as "기본 요소",
and Input/Navigation/Status as "입력"/"탐색"/"상태". Use translate or
LocaleText consistently for group headings, filters, breadcrumbs and ToC.
Keep English labels in English mode and preserve the Compass wordmark.

## Updating the AI Reference

Run `pnpm compass:export` after source or guidance changes. This also runs
before `pnpm dev` and `pnpm build`. It exports only the explicitly selected
public design-system files; it does not export secrets or the whole repository.
