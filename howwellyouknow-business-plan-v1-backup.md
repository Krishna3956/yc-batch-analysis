# How Well You Know - Business Plan

## The Idea

A platform that turns product documentation into interactive, gamified quizzes. B2B SaaS companies embed these quizzes in their docs, onboarding flows, and marketing pages to drive product adoption, educate users, and capture leads.

Think: **testimonial.io but for product education.** Companies sign up, paste their docs URL, and get a branded, embeddable quiz that teaches their users.

**Delivery Model: Hybrid (Model C)**
- Phase 1: Manually build 10-15 showcase quizzes for popular tools (you curate, 95% quality)
- Phase 2: Self-serve AI auto-generation from docs URL in 60-90 seconds (70-80% quality + customer editing)
- Premium: Done-for-you hand-crafted quizzes at $500-2,000 per quiz

---

## 1. Competitive Landscape (Data-Backed)

### Direct Competitors (Gamified Quiz / Interactive Content Platforms)

| Company | What They Do | Pricing | Revenue / Funding | Gap |
|---------|-------------|---------|-------------------|-----|
| **Drimify** | General gamification platform (quizzes, personality tests, spin wheels, advent calendars) | Standard: $179/mo, Premium: higher | Bootstrapped, no public revenue | Generic. Not product/docs-specific. No AI generation. Clunky UI. |
| **Interacty** | Interactive content builder (quizzes, memory games, flip cards, timelines) | Free tier, Basic $29/mo, Pro $85/mo | No public funding data | Education-focused. Not designed for B2B product adoption. No docs-to-quiz pipeline. |
| **Quizgecko** | AI quiz generator from PDFs/URLs | ~$24/mo | Small indie tool | Education/student-focused. No embed, no branding, no analytics dashboard for B2B. |
| **Revisely** | AI quiz generator from documents | Freemium | Student tool | Same as above. Zero B2B positioning. |

### Adjacent Competitors (Interactive Demos / Product Adoption)

| Company | What They Do | Pricing | Revenue / Funding | Why They're Not You |
|---------|-------------|---------|-------------------|---------------------|
| **Navattic** (YC W21) | Interactive product demos from screenshots | $500+/mo | $5.6M raised (Seed) | Demos, not quizzes. Shows the product, doesn't test knowledge. No gamification. |
| **Storylane** (YC) | Interactive product demos | $40/user/mo | $125K raised, est. $10-25M rev, 43 employees | Same as Navattic. Demo tool, not education tool. |
| **Arcade** | Record-and-share interactive demos | $32/user/mo | $21.7M raised (Series A, Kleiner Perkins) | Demo/video tool. No quiz, no gamification, no scoring. |
| **Userpilot** | In-app onboarding & product adoption | $249+/mo | $18.5M raised | In-app tooltips/checklists. Heavy. Not gamified. Not embeddable outside the app. |
| **Appcues** | In-app user onboarding | $249+/mo | $36.6M raised | Same category as Userpilot. Onboarding flows, not quizzes. |

### Key Insight from Competitive Analysis

**Nobody is doing exactly what you're describing.** There's a clear gap:

- Quiz generators (Quizgecko, Revisely) are all **consumer/education** focused. They make quizzes for students, not for B2B SaaS companies.
- Gamification platforms (Drimify, Interacty) are **generic**. They're not AI-powered, not docs-specific, and their UX is dated.
- Interactive demo tools (Navattic, Storylane, Arcade) **show** the product but don't **test** knowledge. They're passive, not active.
- Product adoption tools (Userpilot, Appcues) are **in-app only**. Can't be embedded externally. Overkill for product education.

**Your positioning: AI-powered, docs-to-quiz, embeddable, gamified product education for B2B SaaS.** Nobody occupies this exact square.

---

## 2. Market Size

### Gamification Market (Broad)
- **2025:** $16-29 billion (varies by source)
- **2026:** $20-36 billion
- **2035:** $112-119 billion
- **CAGR:** 22-25%

### Interactive Demo Market (More Relevant)
- Navattic, Storylane, Arcade together have raised $27M+ and serve thousands of B2B companies
- Navattic starts at $500/mo. Arcade at $32/user/mo. Storylane est. $10-25M revenue.
- This validates that B2B companies WILL pay for tools that help prospects/users understand their product

### Your Addressable Market (SAM)
- Target: B2B SaaS companies with public documentation
- There are ~30,000+ B2B SaaS companies globally with documentation sites
- Even 0.1% penetration at $99/mo avg = $36K MRR
- At 1% penetration at $99/mo = $360K MRR

### $10K MRR Target: Is It Realistic?

$10K MRR = ~100 customers at $99/mo, or ~40 customers at $249/mo

**Honest answer: Yes, it's achievable but not "easily."** Here's why:

- Storylane reached est. $10-25M revenue with 43 people. They started scrappy.
- Testimonial.io (your model inspiration) hit $25K MRR within a year as a mostly solo founder.
- The interactive demo category proved B2B teams will pay $200-500/mo for tools that help users understand their product.
- But you'll need 6-12 months of execution, not 2 months.

---

## 3. Why Would They Buy vs Build?

This is the critical question. Let's break it down with second and third-order thinking.

### First Order: "I can build a quiz with Claude Code in a day"

True. Any individual DevRel person can build ONE quiz. But:

### Second Order: "Now maintain it, brand it, embed it, track analytics, capture leads, A/B test it, update it when docs change"

This is where DIY falls apart:
- Docs change every sprint. Who updates the quiz? The engineer who built it moved on.
- Where do leads go? Into a spreadsheet? They need CRM integration.
- The embed breaks on mobile. Who fixes it? No one. It rots.
- They want to A/B test question formats. No infrastructure for that.
- They want a leaderboard for their community. Custom build? That's 2 weeks of eng time.

### Third Order: "The quiz is now a lead gen channel. It needs to be treated like a product."

- Marketing wants branded scorecards users share on LinkedIn (free distribution)
- Sales wants lead data (email, score, which features they don't know) piped to HubSpot
- Product wants to know which features users consistently don't understand (product insight)
- Community wants a leaderboard and badges

No single DevRel person is building all of this. And no engineering team is prioritizing it over core product work.

### The Real Moat

It's not the quiz itself. It's:
1. **Speed:** Paste your docs URL, get a quiz in minutes (not days of eng work)
2. **Maintenance:** Quiz auto-updates when docs change
3. **Infrastructure:** Embed, analytics, lead capture, CRM integration, shareable scorecards, leaderboards - all out of the box
4. **Templates:** 5+ game formats (Truth or Myth, This or That, Speed Pick, etc.) - not just boring multiple choice
5. **Network effect:** "Powered by HowWellYouKnow" at the bottom of every quiz = free distribution

---

## 4. User Journey (Model C: Hybrid)

### For the SaaS Company (Buyer) - Self-Serve Flow

```
Step 1: Land on Homepage (NO signup required)
  - Sees hero: "Turn your docs into an interactive quiz"
  - Pastes their documentation URL
  - Clicks "Generate Quiz"

Step 2: Loading (60-90 seconds, NO signup required)
  - Fun progress animation:
    "Reading your docs..." (10s)
    "Crafting questions..." (30s)
    "Designing your quiz..." (20s)
  - Backend: scrapes docs with Cheerio/Puppeteer, sends to Claude API,
    receives quiz-config JSON, renders with quiz engine

Step 3: Preview (NO signup required)
  - Customer sees their quiz LIVE. Can play it immediately.
  - This is the magic moment. Value shown BEFORE any commitment.

Step 4: Conversion Gate
  - "Like what you see? Sign up to customize, embed, and track analytics."
  - THIS is where email is captured. After they've seen the value.
  - Google OAuth signup.

Step 5: Dashboard (after signup)
  - Edit any question in a simple inline editor
  - Swap brand colors/logo
  - Configure lead capture (optional: require email before showing score)
  - Configure embed settings
  - Get embed code (iframe) OR hosted URL (howwellyouknow.com/their-product)

Step 6: Publish & Analytics
  - Dashboard shows: plays, completion rate, avg score, lead captures
  - Per-question breakdown: which features users don't know
  - Lead export to CSV or CRM integration (HubSpot, Salesforce)

Step 7: Iterate
  - Gets notified when docs change significantly
  - One-click regenerate quiz with updated content
  - A/B test different question formats
```

### For the SaaS Company (Buyer) - Done-For-You Premium Flow

```
Step 1: Request via dashboard or contact page
  - "Want a hand-crafted, premium quiz? We'll build it for you."

Step 2: You manually build the quiz using Claude Code (12-24 hours)
  - 95% quality. Curated questions, custom touches.
  - Acceptable wait time because they're paying $500-2,000 for premium.

Step 3: Deliver via dashboard
  - Customer reviews, requests edits, publishes.
```

### For the End User (Quiz Player)

```
Step 1: Encounters the quiz
  - Sees it embedded in a product's docs, onboarding, or shared on social media

Step 2: Plays
  - 6 rounds, 15 challenges, ~3 minutes (same format as your Claude Code quiz)
  - Fun, interactive, not boring corporate training

Step 3: Gets score
  - Scorecard with spider-web chart showing strengths/weaknesses
  - Archetype title (fun, shareable)

Step 4: Lead capture (optional, depends on company's config)
  - "Enter your email to get your detailed results + tips to improve"

Step 5: Share
  - One-click share to LinkedIn/X with branded scorecard
  - "I scored 72/100 on [Product] Knowledge - I'm a [Archetype]!"
  - This drives organic traffic back to the quiz (growth loop)
```

---

## 5. Business Model & Pricing

### Freemium + Self-Serve SaaS (Testimonial.io Model)

| Tier | Price | What You Get |
|------|-------|-------------|
| **Free** | $0 | 1 quiz, 100 plays/month, "Powered by HowWellYouKnow" badge, basic analytics |
| **Pro** | $49/mo | 3 quizzes, 1,000 plays/month, remove branding, lead capture, embed code, basic analytics |
| **Business** | $149/mo | 10 quizzes, 10,000 plays/month, CRM integrations, advanced analytics, custom domain, A/B testing |
| **Enterprise** | $499/mo | Unlimited quizzes, unlimited plays, SSO, custom game formats, dedicated support, API access |

### Additional Revenue Streams

1. **Done-for-you quiz creation:** $500-2,000 per quiz (hand-crafted premium quality, 12-24hr delivery)
2. **Sponsored free quizzes:** Companies pay to "own" a popular free quiz on the platform (e.g., howwellyouknow.com/figma is sponsored by Figma)
3. **Certification badges:** Companies pay for "official" certification program powered by your platform

### Cost Structure

| Phase | Monthly Cost | What You're Paying For |
|-------|-------------|------------------------|
| Phase 1 (Month 1-2) | ~$1/mo | Domain only. Vercel Hobby (free), no DB needed, manual quiz creation. |
| Phase 2 (Month 3-4) | ~$25-45/mo | Vercel Pro ($20), Supabase free tier, Claude API ($5-20), Resend free. |
| Phase 3 (Month 5-8) | ~$85-115/mo | Supabase Pro ($25), Claude API ($20-50), Vercel Pro ($20), Resend Pro ($20). |
| At $10K MRR | ~$200-300/mo | Same as Phase 3 + Stripe fees (2.9% = ~$300/mo). **97% margins.** |

**AI API cost per quiz generation:** ~$0.10-0.50 (Claude Sonnet/Opus). Negligible even at scale.

### Revenue Projections (Conservative)

| Month | Free Users | Paid Users | MRR |
|-------|-----------|-----------|-----|
| 1-3 | 50 | 5 | $500 |
| 4-6 | 200 | 20 | $2,000 |
| 7-9 | 500 | 50 | $5,000 |
| 10-12 | 1,000 | 100 | $10,000 |

$10K MRR in 12 months is possible if execution is strong. Not "easy" but doable.

---

## 6. GTM (Go-To-Market) Strategy

### Phase 1: Build the Audience with Showcase Quizzes (Month 1-2)
Manually build 10-15 free quizzes for popular tools using Claude Code. These are your marketing engine and proof of quality.

**Target tools for free quizzes:**
- Claude Code, Cursor, Windsurf, GitHub Copilot (AI coding tools)
- ChatGPT, Perplexity, Gemini (AI assistants)
- Figma, Notion, Linear (design/productivity)
- Vercel, Supabase, Clerk (developer platforms)
- HubSpot, Intercom, Zendesk (B2B SaaS)

**Distribution channels:**
- **Product Hunt:** Launch each quiz individually AND launch the platform. You know how to do this.
- **Reddit:** r/ClaudeAI, r/cursor, r/figma, r/SaaS, r/webdev. Post: "I made a quiz to test how well you know [tool]. Most people score below 50."
- **Hacker News:** "Show HN: How well do you know [tool]?" Non-salesy, curiosity-driven.
- **LinkedIn:** Your proven playbook. Topic-first posts about skill gaps in popular tools, with quiz link in comments.
- **X/Twitter:** Tag the official accounts. "I made a quiz about @cursor_ai and most users score 40/100. Are you better?"
- **Tool-specific communities:** Discord servers, Slack communities, forums.

### Phase 2: Launch Self-Serve Auto-Generation (Month 3-4)
- Build the AI auto-generation pipeline: paste docs URL -> quiz in 60-90 seconds
- Homepage CTA: "Turn your docs into an interactive quiz" with URL input field
- Show quiz preview BEFORE asking for signup (value-first conversion)
- Add self-serve dashboard: edit questions, brand it, embed it, track analytics
- Outbound to companies: "Hey [Company], 2,000 people played our [Product] quiz this month. Want to own it? Customize it? Capture leads from it?"
- Pricing page + Stripe integration

### Phase 3: Scale & Monetize (Month 5-12)
- Done-for-you premium tier ($500-2,000/quiz) for companies wanting hand-crafted quality
- CRM integrations (HubSpot, Salesforce, Zapier)
- Lead capture configuration in dashboard
- Custom branding (logo, colors) via dashboard
- A/B testing for question formats
- Partner program: DevRel agencies resell your platform
- Case studies from early customers
- Content marketing: blog posts about "why interactive product education beats documentation"

---

## 7. The Embed: How It Actually Works

### Option A: Hosted Page (Default)
```
howwellyouknow.com/figma
howwellyouknow.com/cursor
howwellyouknow.com/your-company
```
Full-page experience. The company shares this URL. You host everything.

### Option B: Iframe Embed
```html
<iframe 
  src="https://howwellyouknow.com/embed/figma" 
  width="100%" 
  height="600" 
  frameborder="0"
></iframe>
```
Company pastes this into their docs site, blog, or onboarding page. Quiz loads inline.

### Option C: Custom Domain (Business/Enterprise)
```
quiz.figma.com  (CNAME to howwellyouknow.com)
learn.cursor.com
```
Company points their subdomain to your platform. Fully white-labeled.

### Data Flow
```
User plays quiz
    |
    v
Score + answers stored in your DB
    |
    v
Lead data (if email captured) -> Company's dashboard
    |                                   |
    v                                   v
Shareable scorecard generated     CSV export / CRM webhook
    |                             (HubSpot, Salesforce, Zapier)
    v
User shares on LinkedIn/X
    |
    v
New users discover quiz (growth loop)
```

---

## 8. What to Build First (Build Order)

### Week 1-2: Platform Foundation + Quiz Engine
- [ ] Set up howwellyouknow.com with Next.js (reuse Claude Code quiz codebase)
- [ ] Refactor quiz to read from JSON config (decouple content from design)
- [ ] Quiz engine renders any quiz from a config file (same design, different content)
- [ ] Theming system: swap brand colors/logo per quiz based on config
- [ ] Homepage: hero with "Turn your docs into a quiz" + cards for available quizzes
- [ ] Routing: /claude-code (existing), /cursor, /figma, etc.
- [ ] "Powered by HowWellYouKnow" footer on each quiz

### Week 3-4: 5 More Showcase Quizzes (Manually Built)
- [ ] Cursor quiz (15 questions from Cursor docs)
- [ ] ChatGPT quiz
- [ ] Figma quiz
- [ ] Notion quiz
- [ ] Vercel quiz
- [ ] All manually crafted using Claude Code for 95% quality
- [ ] Each quiz = a JSON config file consumed by the shared quiz engine

### Month 2: Launch & Distribution
- [ ] Launch each quiz on relevant subreddits, PH, HN, LinkedIn
- [ ] Track traffic, completion rates, scores
- [ ] Collect emails on results page ("Get weekly tips to improve your score")
- [ ] Measure demand: target 10,000+ total plays across all quizzes

### Month 3: AI Auto-Generation Pipeline
- [ ] Build docs scraper (Cheerio/Puppeteer)
- [ ] Build Claude API integration with quiz generation prompt
- [ ] Homepage flow: paste docs URL -> loading animation (60-90s) -> live quiz preview
- [ ] NO signup required to generate and preview a quiz (value-first)
- [ ] Conversion gate after preview: "Sign up to customize, embed, and track analytics"

### Month 4: Self-Serve Dashboard
- [ ] Sign up / login (Google OAuth via Clerk or NextAuth)
- [ ] Quiz editor: inline question editing, reorder, delete
- [ ] Brand customization: logo, colors upload
- [ ] Embed code generator (iframe)
- [ ] Basic analytics: plays, scores, completion rate
- [ ] Pricing page + Stripe integration

### Month 5-8: Monetize & Scale
- [ ] Lead capture configuration
- [ ] CRM integrations (start with Zapier webhook, then native HubSpot)
- [ ] Done-for-you premium service ($500-2,000/quiz)
- [ ] Custom domain support (CNAME)
- [ ] A/B testing
- [ ] Case studies from early customers

---

## 9. UI/UX Recommendations

### Homepage (howwellyouknow.com)
- **Hero section (above fold):**
  - Headline: "Turn your docs into an interactive quiz"
  - Large URL input field: "Paste your documentation URL"
  - CTA button: "Generate Quiz" (no signup required)
  - Subtext: "Free. Ready in 60 seconds. No account needed."
- **Below the fold:** Grid of showcase quiz cards
  - Each card: tool logo, tool name, "X people played", avg score, difficulty level
  - Search/filter by category (AI Tools, Design, Productivity, DevTools)
- **For companies section:** "Embed quizzes in your docs, capture leads, track analytics."
- The homepage serves dual purpose: playground for trying the tool + directory of existing quizzes

### Quiz Experience
- Keep your existing Claude Code quiz design. It's already beautiful.
- Consistent across all quizzes: same 6 rounds, same animations, same scoring
- Tool-specific branding: each quiz uses the tool's accent color
- Mobile-first. Every quiz must work perfectly on phone.

### Dashboard (for paying companies)
- Clean, modern SaaS dashboard (think Linear, not Salesforce)
- Left nav: Quizzes, Analytics, Leads, Settings
- Quiz editor: drag-and-drop question reorder, inline editing
- Analytics: graphs for plays over time, score distribution, per-question breakdown
- Lead table: name, email, score, date, export button

### Embed Widget
- Minimal, fast-loading iframe
- Responsive: adapts to container width
- Optional "pop-up" mode: button on company's site triggers full-screen quiz overlay

---

## 10. Quiz Delivery Architecture (Model C: Hybrid)

### How AI Auto-Generation Works (Self-Serve, 60-90 seconds)

```
1. Customer pastes docs URL on homepage (no signup)
2. Backend scrapes docs pages (Cheerio/Puppeteer, 10-30 seconds)
3. Scraped content sent to Claude API with system prompt:
   "Generate a quiz-config JSON with 15 questions across 6 rounds,
    formats: Truth or Myth, This or That, Speed Pick, etc."
4. Claude API returns structured JSON (15-30 seconds)
5. Quiz engine renders the quiz from JSON (instant)
6. Customer sees their quiz live, can play it immediately
7. Conversion gate: "Sign up to customize and embed"
```

### What AI Generates (JSON Config)
```json
{
  "tool": "Figma",
  "brandColor": "#A259FF",
  "dimensions": ["Prototyping", "Components", "Auto Layout", "Plugins", "Collaboration"],
  "rounds": [
    {
      "type": "truth_or_myth",
      "title": "Figma Fact Check",
      "questions": [
        {
          "statement": "Auto Layout can only flow in one direction per frame",
          "answer": "myth",
          "explanation": "Since Figma's 2023 update, Auto Layout supports wrap."
        }
      ]
    }
  ],
  "archetypes": [
    { "name": "The Pixel Perfectionist", "minScore": 85, "description": "..." }
  ]
}
```

### What YOUR Engine Handles (Built Once, Reused Forever)
- All design, animations, page transitions (Framer Motion)
- Radar chart, scorecard, sharing flow
- Responsive mobile-first layout
- Lead capture forms
- Embed/iframe rendering
- Analytics tracking

### Quality Expectations
- **AI auto-generated:** 70-80% quality. Some questions will be obvious or awkward. Customer edits in dashboard.
- **Manually curated (showcase + premium):** 95% quality. You personally review and polish.
- **System prompt is the real IP.** 2,000-3,000 tokens of carefully crafted instructions. Iterated 20-30 times.

### AI API Cost Per Quiz
- Claude 3.5 Sonnet: ~$0.10-0.15 per quiz generation
- Claude Opus (premium quality): ~$0.50-1.00 per quiz generation
- At 500 quizzes/month: ~$50-75/mo. Negligible.

---

## 11. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Companies build their own with AI | High | Your value is the platform (analytics, embed, lead capture, maintenance), not the quiz itself. One quiz is easy to build; a maintained system is not. |
| Low willingness to pay | Medium | Free tier with "Powered by" badge = organic distribution. Paid tier removes badge + adds analytics. Same model as Testimonial.io. |
| Quiz quality from auto-generation | Medium | Human-in-the-loop: AI generates 70-80% draft, customer edits in dashboard. Premium tier: you hand-craft for 95% quality. |
| Companies get annoyed you made a quiz about their product | Low | You're driving awareness and education for their product. Most DevRel teams would love this. Worst case: they ask you to take it down, and you offer them the paid version instead. |
| Small TAM | Medium | Start with dev tools (your strength), expand to all B2B SaaS. 30,000+ companies with docs = large enough. |
| AI-generated questions too easy/boring | Medium | System prompt engineering is key. Invest in making the prompt generate tricky, non-obvious questions. Iterate 20-30 times. |

---

## 12. Final Verdict

### Should you do it?

**Yes, but with guardrails.**

- Do NOT quit your job
- Do NOT spend months building the self-serve platform before validating demand
- DO build 10-15 free showcase quizzes manually, launch them, measure traffic (Phase 1)
- DO reach out to 5-10 DevRel teams with free quizzes and see if they want to pay for customization
- DO build the AI auto-generation pipeline + self-serve dashboard after traffic validates demand (Phase 2)
- DO offer done-for-you premium ($500-2K) as early revenue while building the self-serve product
- Key principle: show value BEFORE asking for signup (paste URL -> see quiz -> then sign up)

### The real test
If you can get 10,000+ total quiz plays across all tools within 2 months of launching, the demand is real. If you can't, the idea needs to pivot or die.

### Why this could work for YOU specifically
1. You've already built the quiz engine (Claude Code quiz). Reuse it.
2. You know how to launch on PH, Reddit, HN, LinkedIn. Distribution is your edge.
3. You work at Cisco (B2B SaaS). You understand the buyer.
4. This directly strengthens your YC Startup School application (platform thinking, shipped product, AI-native building).

### $10K MRR timeline
- Optimistic: 8-10 months
- Realistic: 12-15 months
- Pessimistic: 18+ months or doesn't work

The first $1K MRR is the hardest. Everything after that gets easier because you have proof, case studies, and organic traffic compounding.
