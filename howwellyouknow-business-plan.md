# How Well You Know — Business Plan v2

*Last updated: March 6, 2026*

---

## Executive Summary

**How Well You Know** turns product documentation into interactive micro-learning experiences for B2B SaaS companies. Not quizzes — learning games. Each experience teaches users through bite-sized lessons, then tests knowledge through gamified challenges, then delivers a shareable scorecard.

**Positioning:** A lightweight, embeddable "Trailhead" any B2B SaaS can set up in 5 minutes. Skilljar charges $10K+/year. We deliver 80% of that value for $49-149/mo, powered by AI.

**Why now:**
- 90% of companies see positive ROI from customer education (Forrester)
- Over 50% of SaaS customers churn if they don't understand the product
- involve.me (quiz/form builder) does $24M/year revenue
- Skilljar raised $33M, acquired by Gainsight in 2025
- Nobody does AI-powered docs-to-learning-game for B2B

---

## 1. What We ARE vs What We're NOT

### NOT:
- Quiz tool (Quizgecko, Revisely — for students)
- LMS (Skilljar, Thought Industries — $10K+/year enterprise)
- Interactive demo tool (Navattic, Storylane — shows product, doesn't teach)
- In-app onboarding (Userpilot, Appcues — tooltips inside the product)

### ARE:
**A micro-learning experience builder for B2B product education.**

Each experience has:
1. **Micro-lessons** — 30-second teaching moments before each challenge
2. **Interactive challenges** — gamified (Truth or Myth, This or That, Speed Pick, Scenario Match)
3. **Scorecard** — radar chart showing strengths/weaknesses
4. **Shareable result** — branded archetype card for LinkedIn/X (growth loop)
5. **Lead capture + analytics** — who played, what they scored, which features they don't know

**Key insight: we teach THEN test.** Every challenge has a micro-lesson before it. This is how Duolingo works: learn → practice → reinforce.

---

## 2. The Customer Education Market

### Why This Market Matters (Data)

| Stat | Source |
|------|--------|
| 90% of companies see positive ROI from customer education | Forrester / Intellum |
| 6.2% revenue increase from formalized education | Forrester |
| 7.4% retention increase | Forrester |
| 7.1% lifetime value increase | Forrester |
| 11.6% customer satisfaction increase | Forrester |
| 6.1% decrease in support costs | Forrester |
| 131% more likely to buy after educational content | Conductor |
| 86% of customers stay if onboarding is good | Wyzowl |
| 50%+ of SaaS customers churn if they don't understand the product | Baremetrics |
| Cutting churn by 5% can double growth rate | Baremetrics |
| Trailhead: 40% engagement increase, 20% feature adoption boost | SaaS Designer |

### Who Already Pays for Customer Education

| Company | What | Revenue / Funding | Price |
|---------|------|-------------------|-------|
| **Skilljar** | Customer education LMS | $33M raised, acquired by Gainsight (2025) | $10K+/year |
| **Thought Industries** | Customer training | ~$500M funding, ~$25.7M revenue | Enterprise |
| **WorkRamp** | Customer + employee education | Funded | Enterprise |
| **Trainn** | Customer training for SaaS | Funded | ~$99/mo+ |
| **7taps** | Microlearning platform | Funded | Enterprise |
| **involve.me** | Quiz/form builder for lead gen | **$24M revenue**, 3K customers | $29-79/mo |
| **Senja** | Testimonial widget (model comp) | **$300K revenue**, 2 employees | Freemium |

### The Gap We Fill

Skilljar/Thought Industries/WorkRamp = full LMS, $10K-100K/year, months to set up, need content team.

**How Well You Know** = micro-learning from your docs, $49-149/mo, set up in 5 minutes, AI does the content. Perfect for B2B SaaS companies 50-500 employees who can't afford enterprise LMS but need customer education.

---

## 3. How Companies Will Actually Use This

### Use Case 1: Customer Onboarding (HIGHEST VALUE)

**Problem:** New customers sign up, read some docs, don't understand the product, churn in 90 days.

**How they use us:** CS sends new users a link: "Complete your [Product] training in 5 minutes." Embedded in onboarding email sequences (Day 1, Day 7, Day 30). User plays → learns key features → feels confident → uses product more → doesn't churn.

**Who buys:** Head of Customer Success, VP CS
**Why they care:** Churn reduction is their #1 KPI.
**Where it embeds:** Onboarding emails, in-app onboarding page, customer portal

#### How the Learning Module Actually Works (Screen by Screen)

This is the most important part of the product. Imagine a new user just signed up for a project management tool called "TaskFlow." Their CS team sends them this link in their welcome email: `learn.taskflow.com/getting-started`

Here's exactly what the user sees, card by card. The format is similar to 7taps (swipeable cards on mobile, click-to-advance on desktop), but gamified — with points, timers, streaks, and a final scorecard.

**Screen 1: Welcome Card**
```
┌─────────────────────────────────┐
│                                 │
│  🎯 How Well Do You Know        │
│     TaskFlow?                   │
│                                 │
│  Learn the essentials in        │
│  5 minutes. No docs needed.     │
│                                 │
│  ◻ 4 rounds                     │
│  ◻ ~20 cards                    │
│  ◻ You'll learn real features   │
│                                 │
│  [Start Learning →]             │
│                                 │
└─────────────────────────────────┘
```
The user taps "Start Learning." No signup required (yet).

**Screen 2: Micro-Lesson Card (TEACH)**
```
┌─────────────────────────────────┐
│  Round 1: Core Concepts   1/4   │
│─────────────────────────────────│
│                                 │
│  💡 Did you know?               │
│                                 │
│  TaskFlow's "Smart Assign"      │
│  automatically assigns tasks    │
│  to the team member with the    │
│  most availability. You don't   │
│  need to check calendars.       │
│                                 │
│  It uses your team's workload   │
│  data in real-time.             │
│                                 │
│  [Got it →]                     │
│                                 │
└─────────────────────────────────┘
```
This is a TEACHING card. Short, one concept only. Written by AI from the product's documentation. The user reads it in 10-15 seconds and taps "Got it."

**Screen 3: Challenge Card (TEST what you just learned)**
```
┌─────────────────────────────────┐
│  ⚡ Truth or Myth?       +10pts │
│─────────────────────────────────│
│                                 │
│  "Smart Assign requires you     │
│   to manually check team        │
│   calendars before assigning."  │
│                                 │
│  ┌───────────┐ ┌───────────┐   │
│  │   TRUTH   │ │   MYTH ✓  │   │
│  └───────────┘ └───────────┘   │
│                                 │
│  ✅ Correct! Smart Assign       │
│  checks availability            │
│  automatically — no calendar    │
│  checking needed.               │
│                                 │
│  [Next →]              🔥 1     │
└─────────────────────────────────┘
```
This IMMEDIATELY tests what they just learned. It's not a random trivia question — it directly reinforces the micro-lesson they just read. The user gets instant feedback + points + a streak counter.

**Screen 4: Another Micro-Lesson Card (TEACH)**
```
┌─────────────────────────────────┐
│  Round 1: Core Concepts   2/4   │
│─────────────────────────────────│
│                                 │
│  💡 Pro tip                     │
│                                 │
│  You can create "Workflows"     │
│  in TaskFlow that trigger       │
│  automatic actions.             │
│                                 │
│  Example: When a task moves to  │
│  "In Review," TaskFlow auto-    │
│  notifies the reviewer and      │
│  sets a 48-hour deadline.       │
│                                 │
│  [Got it →]                     │
│                                 │
└─────────────────────────────────┘
```

**Screen 5: Challenge Card — Different Format (THIS OR THAT)**
```
┌─────────────────────────────────┐
│  🔀 This or That?       +10pts │
│─────────────────────────────────│
│                                 │
│  To auto-notify a reviewer      │
│  when a task moves to           │
│  "In Review," you use:          │
│                                 │
│  ┌─────────────────────────┐    │
│  │  A) Smart Assign        │    │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │  B) Workflows ✓         │    │
│  └─────────────────────────┘    │
│                                 │
│  ✅ Yes! Workflows handle       │
│  automatic actions. Smart       │
│  Assign is for task assignment. │
│                                 │
│  [Next →]              🔥 2     │
└─────────────────────────────────┘
```
Different game format keeps it fresh. Not just multiple choice.

**Screens 6-9: More teach → test pairs for Round 1**

**Screen 10: Round Transition Card**
```
┌─────────────────────────────────┐
│                                 │
│  🏆 Round 1 Complete!           │
│                                 │
│  Core Concepts: 3/4 correct     │
│  Points: 30/40                  │
│                                 │
│  You now know:                  │
│  ✅ Smart Assign                │
│  ✅ Workflows                   │
│  ✅ Team Workload View          │
│  ❌ Custom Fields (review this) │
│                                 │
│  [Continue to Round 2 →]        │
│                                 │
└─────────────────────────────────┘
```
Progress feedback. The user sees exactly what they learned and what they missed. The ❌ item links back to the docs section so they CAN go deeper if they want.

**Screen 11-16: Round 2 — "Power Features" (same teach → test pattern)**

**Screen 17-20: Round 3 — "Speed Round" (pure recall, timed)**
```
┌─────────────────────────────────┐
│  ⏱ Speed Round!  8 sec left     │
│  ████████░░░░░░░░               │
│─────────────────────────────────│
│                                 │
│  What feature auto-assigns      │
│  tasks based on availability?   │
│                                 │
│  ┌──────────┐ ┌──────────┐     │
│  │ Workflows│ │Smart     │     │
│  │          │ │Assign ✓  │     │
│  └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐     │
│  │ Templates│ │ Triggers │     │
│  └──────────┘ └──────────┘     │
│                                 │
│              +10pts    🔥 5     │
└─────────────────────────────────┘
```
The speed round tests RECALL of everything they learned in Rounds 1-2. It's fast, gamified, and creates urgency. This is where the "game" feeling kicks in — timer bar, quick taps, streak counter going up.

**Screen 21-23: Round 4 — "Real Scenarios" (apply knowledge)**
```
┌─────────────────────────────────┐
│  🎭 Scenario                    │
│─────────────────────────────────│
│                                 │
│  Your team lead asks:           │
│  "I want new tasks to auto-     │
│  assign to whoever has the      │
│  lightest workload AND notify   │
│  them via Slack."               │
│                                 │
│  What TWO features do you       │
│  combine?                       │
│                                 │
│  ☑ Smart Assign                 │
│  ☐ Templates                    │
│  ☑ Workflows (+ Slack trigger)  │
│  ☐ Custom Fields                │
│                                 │
│              +20pts    🔥 7     │
└─────────────────────────────────┘
```
This tests APPLICATION — can the user combine features to solve a real problem? This is the highest-value learning moment because it simulates what they'll actually do in the product.

**Screen 24: Scorecard (THE PAYOFF)**
```
┌─────────────────────────────────┐
│                                 │
│  🏆 You scored 82/100!          │
│                                 │
│  You are: "The Quick Learner"   │
│                                 │
│     Concepts ████████░░ 80%     │
│     Features ██████████ 100%    │
│     Speed    ██████░░░░ 60%     │
│     Scenarios████████░░ 90%     │
│                                 │
│  📊 You outperformed 68% of     │
│     TaskFlow users              │
│                                 │
│  Weak area: Speed recall.       │
│  Review: docs.taskflow.com/     │
│          shortcuts              │
│                                 │
│  [Share on LinkedIn 📤]         │
│  [Retake Challenge 🔄]         │
│  [Explore TaskFlow Docs →]      │
│                                 │
│  ── Powered by HowWellYouKnow ─│
└─────────────────────────────────┘
```

#### The Full Flow for CS Teams

Here's how a Customer Success Manager at TaskFlow actually uses this:

1. **Setup (5 min):** CS person goes to HowWellYouKnow dashboard → pastes docs.taskflow.com → AI generates a draft learning experience in 60 seconds → CS reviews, tweaks a few questions, hits publish.

2. **Distribute:** CS copies the link (`learn.taskflow.com/getting-started`) and adds it to their onboarding email sequence:
   - **Day 1 email:** "Welcome! Complete your TaskFlow training in 5 minutes → [link]"
   - **Day 7 email:** "Haven't started yet? Your teammates scored 78% → [link]"
   - **Day 30 email:** "Advanced features unlocked! Level 2 → [link]"

3. **Track:** CS sees a dashboard showing:
   - 73% of new users completed the experience (vs ~12% who read docs)
   - Average score: 71/100
   - Weakest feature: "Workflows" — only 40% got these questions right
   - Users who completed the experience have 2.3x higher product activation

4. **Act on data:** CS team realizes nobody understands Workflows → creates a dedicated "Workflows Deep Dive" experience → sends it to all users who scored <50% on that section.

#### Why This Beats the Alternatives

| What companies do today | Problem | How we're better |
|------------------------|---------|-----------------|
| Send users to docs site | Nobody reads docs (<12% completion) | 73%+ completion because it's interactive and only 5 min |
| Record video walkthroughs | 6-min avg watch time on 20-min video, no feedback | Every card is active — tap, choose, think. Immediate feedback |
| Live webinars | Scheduling hell, doesn't scale, no data | Async, self-paced, works at 2am, tracks per-feature knowledge |
| In-app tooltips (Appcues, Userpilot) | Only works inside the app, user must already be in product | Works BEFORE they open the product — in email, docs site, landing page |
| Full LMS (Skilljar, WorkRamp) | $10K+/year, months to set up, need content team | $49-149/mo, set up in 5 min, AI writes the content |
| Quizzes (Typeform, involve.me) | Tests only, doesn't teach, no product education focus | Teaches THEN tests, product-specific, knowledge gap analytics |

#### Key Design Principle: It's Not a Video. It's Not a Quiz. It's a Card-Based Learn-Then-Test Flow.

Think of it like Instagram Stories meets Duolingo:
- **Instagram Stories format:** Swipeable cards, mobile-first, tap to advance, short text, visual
- **Duolingo mechanics:** Points, streaks, immediate feedback, "aha moment" after each card, progress tracking
- **Product education content:** AI reads their docs, extracts key concepts, creates teach-then-test pairs

The card types (what AI auto-generates from docs):
1. **💡 Micro-lesson card** — teaches ONE concept in 2-3 sentences (the "teach" part)
2. **⚡ Truth or Myth** — reinforces what was just taught (the "test" part)
3. **🔀 This or That** — comparison between two features
4. **⏱ Speed Round** — rapid recall, timed
5. **🎭 Scenario** — apply knowledge to a real-world situation
6. **🏆 Scorecard** — personalized results with knowledge gaps

#### How This Compares to 7taps Specifically

7taps is the closest product in terms of format (card-based microlearning). Key differences:

| | 7taps | How Well You Know |
|---|-------|-------------------|
| **Content creation** | Manual: you write every card yourself | AI: paste docs URL → cards auto-generated |
| **Target audience** | Internal employee training (L&D teams) | External customer education (CS teams) |
| **Gamification** | Minimal (quiz cards exist but no points/streaks/leaderboards) | Core: points, streaks, speed rounds, archetypes, shareable scorecards |
| **Analytics** | Course completion, quiz scores | Per-feature knowledge gaps, radar charts, benchmark vs other users |
| **Pricing** | Enterprise only (custom pricing, no self-serve) | Self-serve from $49/mo |
| **Shareability** | Private (internal training) | Designed for sharing — branded scorecards on LinkedIn/X |
| **Distribution** | Link or LMS integration | Embed anywhere: email, docs site, landing page, iframe |
| **Brand/growth loop** | No viral mechanic | "Powered by HowWellYouKnow" + shareable scorecards = organic growth |

7taps is for L&D managers training their employees. We're for CS teams educating their customers. Same card-based format, completely different buyer, use case, and distribution.

### Use Case 2: Product Marketing / Lead Generation (HIGH VALUE)

**Problem:** Companies need engaging content to attract developers/users. Blog posts get ignored.

**How they use us:** Marketing creates "How Well Do You Know [Product]?" challenge. Shares on social media, newsletters. Users play → share scorecards → organic reach. Email-gated results capture leads.

**Who buys:** Head of Marketing, DevRel Lead (see note below)
**Why they care:** Lead gen + engagement. Interactive content converts at 40%+ vs 10-15% for landing pages.
**Where it embeds:** Landing pages, blog posts, social media, newsletters

### Use Case 3: Documentation Enhancement (MEDIUM VALUE)

**Problem:** Nobody reads docs. Completion rates are <12%.

**How they use us:** "Finished reading about Auto Layout? Test your knowledge." Embedded at end of docs sections. Analytics show which features users struggle with.

**Who buys:** Documentation Lead, DevRel Lead
**Where it embeds:** Documentation site (end of sections)

### Use Case 4: Community Engagement (MEDIUM VALUE)

**Problem:** Community managers need engagement content.

**How they use us:** "Friday Challenge: Who scores highest?" Leaderboards, shared scorecards.

**Who buys:** Community Manager, DevRel Lead
**Where it embeds:** Slack/Discord links, community forums

### What is DevRel?

**DevRel = Developer Relations.** It's a team (1-5 people) at B2B SaaS companies — especially developer-focused ones like Stripe, Twilio, Vercel, Supabase — that builds relationships with their developer community. They write tutorials, create content, speak at conferences, run Slack/Discord communities, and produce educational material.

DevRel people are natural buyers for us because they constantly need engaging, educational content about their product. BUT: most companies in our ICP (50-500 employees) won't have a dedicated DevRel team unless they sell to developers. **So DevRel is a secondary buyer persona, not primary. Head of Customer Success is our #1 target.**

### Are We Making Them Do Something New?

**No.** Every B2B SaaS already does customer onboarding, docs, community content, marketing. They do it through boring docs, long videos, webinars nobody attends. We replace that with a 5-minute interactive learning experience that's actually fun and measurable.

---

## 3B. Stress Test: Honest Second/Third/Fourth-Order Thinking

This section exists because we need to be brutally honest about the assumptions we're making. If any of these are wrong, the business doesn't work.

### Doubt #1: Will Users Drop Off on the "Teaching" Cards?

**The concern:** On Screen 2 (the micro-lesson card), we're essentially making users read a paragraph about a feature before they get to the fun interactive part. Won't they just bounce? Isn't the "lesson" part friction that kills engagement?

**First-order answer:** "No, because it's short — only 2-3 sentences, takes 10 seconds to read."

**Second-order answer (more honest):** It depends entirely on the CONTEXT in which the user arrives.

There are TWO very different user types, and the flow works differently for each:

**User Type A: The "Sent Here" User (Customer Onboarding)**
- They received a link from their CS team: "Complete your TaskFlow training"
- They have INTENT to learn. They signed up for the product. They need to understand it.
- For these users, a teaching card is not friction — it's the whole point. They WANT to learn.
- Completion rates for this type: likely 60-80% (in line with microlearning data — 80% avg completion rate for micro-courses vs 20% for long-form training).

**User Type B: The "Discovered It" User (Organic / Marketing)**
- They found "How Well Do You Know Claude Code?" on Reddit or Product Hunt
- They want to PLAY A GAME, not sit through a lesson
- For these users, a teaching card IS friction. They'll think "I didn't come here to study."
- Completion rates for this type: probably 40-60% if the teaching is light, much lower if it feels preachy.

**Third-order answer:** This means we may need TWO MODES:

1. **Learning Mode** (for customer onboarding) — teach → test → teach → test. Full micro-lessons. The CS team sends this to users who need to learn the product. This is the paid product.

2. **Challenge Mode** (for organic/marketing) — test → reveal → test → reveal. No upfront teaching. Questions come first, and the "lesson" is delivered as the EXPLANATION after they answer (right or wrong). This is the free viral game.

The Claude Code quiz you already built is actually Challenge Mode. People play, get questions, see the answer. There's no upfront lesson. And it works — 380 people played it.

**The key insight:** The teaching doesn't need to come BEFORE the question. It can come AFTER. The user answers a question → gets it wrong → the feedback card teaches them the concept. This is exactly what 7taps recommends: "Start with a question before teaching anything. You didn't test them. You sparked curiosity and delivered the key message."

**Revised flow for both modes:**

| | Learning Mode (paid B2B) | Challenge Mode (free viral) |
|---|---|---|
| **Card 1** | 💡 Micro-lesson: teaches concept | ⚡ Question: jumps straight into challenge |
| **Card 2** | ⚡ Question: tests what was just taught | 💡 Reveal: explains the right answer (teaches AFTER) |
| **Card 3** | 💡 Next micro-lesson | ⚡ Next question |
| **Who uses it** | CS teams send to customers | Marketing shares on social/communities |
| **Intent** | "I need to learn this product" | "This looks fun, let me try" |
| **Drop-off risk** | Low (intent-driven) | Medium (entertainment-driven) |

**This is actually a BETTER product.** Two modes from the same engine. The company can use Challenge Mode for marketing/lead gen and Learning Mode for onboarding. Same content, different sequence.

### Doubt #2: Are Docs Actually for Education, or Just Troubleshooting?

**The concern:** Maybe product documentation is only used when something breaks — "Error 403" → Google it → find the docs page → fix the bug → leave. If that's true, then "turn your docs into a learning experience" makes no sense because the docs aren't about learning, they're about fixing.

**First-order answer:** "Docs are for both learning and troubleshooting."

**Second-order answer (more honest):** You're partially right. There are actually THREE types of documentation, and they serve different purposes:

| Type | Purpose | Example | Used for Learning? |
|------|---------|---------|-------------------|
| **Help Center / Knowledge Base** | Fix specific problems | "Error 403 when connecting to API" | ❌ No — reactive, troubleshooting |
| **Product Docs / User Guides** | Explain how features work | "How to use Smart Assign" | ✅ Yes — proactive, learning |
| **Academy / Training** | Teach users systematically | "TaskFlow Fundamentals Course" | ✅ Yes — structured education |

Most B2B SaaS companies have a mix of the first two. Only larger companies (500+) have a formal Academy.

**The data says:**
- GitBook analytics show that documentation sites get a MIX of traffic: some is troubleshooting ("high traffic to troubleshooting pages points to where users struggle"), some is proactive learning ("tracking which internal links get clicked helps understand if docs guide users through the learning journey").
- But here's the truth: even "How to use Feature X" documentation is NOT great for learning. It's reference material. Reading "Smart Assign automatically assigns tasks based on availability" is like reading a dictionary definition. You understand the words but you haven't learned WHEN and WHY to use it.

**Third-order answer:** We shouldn't position ourselves as "turn your docs into learning." We should position as "turn your PRODUCT KNOWLEDGE into learning." The docs are just the INPUT source — the raw material AI reads to understand the product. The OUTPUT is a completely different format: interactive, gamified, contextual.

Think of it this way:
- Docs = reference manual (like a textbook)
- Our experience = interactive lesson (like Duolingo)
- Same knowledge, completely different delivery

**This changes the messaging.** Instead of "Turn your docs into a micro-learning experience" (which implies docs = learning), we say: "Turn your product into a 5-minute learning experience. We read your docs so your users don't have to."

**Fourth-order answer:** Actually, the INPUT doesn't even need to be docs. It could be:
- Product documentation URL
- Help center articles
- Blog posts about features
- Product changelog
- Marketing feature pages
- YouTube tutorial transcripts
- Even just a URL to the product itself (AI crawls the marketing site)

The docs are the most obvious input, but the real value is: "Give us any content about your product, and we'll turn it into an interactive learning experience." This broadens the appeal significantly.

### Doubt #3: Will Companies Actually PAY for This?

**The concern:** Even if the product is good, will a Head of Customer Success actually pull out a credit card for $149/mo for this?

**Let's think through this at multiple levels:**

**First-order:** "Yes, because customer education reduces churn, and churn costs them way more than $149/mo."

**Second-order (more honest):** Companies pay for things that solve a PAINFUL, URGENT problem they're ALREADY trying to solve. Let's check:

| Criteria | Assessment |
|----------|-----------|
| **Is customer onboarding a real problem?** | ✅ Yes. 75% of new users abandon in the first week when onboarding is bad. 23% churn during onboarding because they can't see value. |
| **Are companies already spending money on this?** | ✅ Yes. 90% of companies with customer education programs see positive ROI. Gainsight acquired TWO education platforms. Enterprise LMS is $10K-50K/yr. |
| **Is our ICP (50-500 employees) spending money on this?** | ⚠️ MAYBE. Most companies this size use: free tools (Loom videos, Google Docs, Notion), a basic help center (Zendesk/Intercom), and maybe webinars. They do NOT have a dedicated customer education budget line item. |
| **Is this a "painkiller" or a "vitamin"?** | ⚠️ This is the critical question. See below. |

**Third-order: Painkiller vs Vitamin**

A painkiller solves acute pain. A vitamin is "nice to have."

- **Painkiller scenario:** A company is losing customers because they don't understand the product. The CS team is spending 10+ hours/week doing manual onboarding calls. They're desperate for something scalable. → They'll pay $149/mo instantly.
- **Vitamin scenario:** A company is doing "fine" with docs and occasional webinars. Onboarding isn't great but isn't terrible. → They'll think "cool product" but won't pay.

**The honest truth:** For most companies in our ICP, this is a vitamin, not a painkiller. They'll only buy if:
1. They have a measurable churn/onboarding problem AND
2. They've already tried other solutions (docs, videos, webinars) that aren't working AND
3. They have someone (Head of CS, CS Ops) who owns onboarding as a KPI

**But here's the counter-argument:** involve.me does $24M revenue with 3K customers selling GENERIC quizzes and forms. Those are vitamins too — nobody dies without a marketing quiz. But marketers buy them because they marginally improve lead gen. Similarly, CS teams will buy our tool because it marginally improves onboarding — IF we make it dead simple to set up and prove ROI.

**Fourth-order: What Actually Needs to Be True for This Business to Work**

For this to reach $10K MRR, we need ~67 paying customers at $149/mo avg. Here's what must be true:

1. **The free games must generate enough traffic to reach potential buyers.** The Claude Code game got 380 plays in 2 days. We need ~10 free games to sustain ~5K-10K monthly visits. Some percentage of players work at B2B SaaS companies. Realistic? Yes, if we consistently publish games for popular tools.

2. **Some percentage of those visitors must convert to creating their own experience.** This is the weakest link. The conversion from "played a fun game" to "I need this for my company" is not obvious. The CTA needs to be extremely clear and the self-serve flow needs to be frictionless (paste URL → see result in 60 seconds → decide to pay).

3. **Cold outreach must work as a parallel channel.** We can't rely only on organic. Emailing 100 Heads of CS per week, with a pre-built game for their specific product, is the stronger channel. "I already built this for you — want to see it?" is a much better cold email than "check out our platform."

4. **The AI-generated content must be good enough.** If the auto-generated questions are bad or the micro-lessons are generic, nobody will pay. The quality bar is set by the Claude Code game you built manually. Can AI match 80% of that quality? Probably yes with Claude Sonnet + good prompting + human review step. But this needs validation.

5. **Companies must see measurable impact.** "73% completion rate vs 12% for docs" — if we can prove this with data from early customers, the product sells itself. Without data, it's a promise.

### What This Means for the Product

Based on this stress test, three strategic shifts:

**Shift 1: Two modes, not one.** Challenge Mode (question-first, for viral/marketing) + Learning Mode (lesson-first, for onboarding). Both from the same engine. This eliminates the drop-off concern for the viral games.

**Shift 2: Position around PRODUCT KNOWLEDGE, not docs.** "We read your docs so your users don't have to" is better than "turn your docs into learning." The input is docs/URLs, but the value prop is "your users learn your product in 5 minutes."

**Shift 3: Cold outreach is the primary revenue channel, not organic.** Don't wait for organic traffic to convert. Pre-build games for 50-100 target companies. Send them the link: "I built a 5-minute learning experience for [Product]. Your users' average score is X. Want the lead data?" This is the "I already built this for you" approach.

### Updated Conviction Level

| Aspect | Before Stress Test | After Stress Test |
|--------|-------------------|-------------------|
| Will end-users engage with it? | ✅ High confidence | ✅ High confidence (especially with Challenge Mode) |
| Will companies pay for it? | ✅ High confidence | ⚠️ Medium confidence — it's a vitamin for most, painkiller for some |
| Can we reach $10K MRR? | ✅ High confidence | ⚠️ Medium confidence — depends on cold outreach execution |
| Is the market real? | ✅ High confidence | ✅ High confidence — validated by acquisitions and revenue data |
| Are docs the right input? | ✅ High confidence | ⚠️ Medium — docs are input, not the value prop. Product knowledge is the value prop. |
| Is the competitive moat strong? | ✅ High confidence | ⚠️ Medium — moat is brand + content library + distribution, not technology |

### Doubt #4: Why Did 7taps Choose L&D (Employee Training) Instead of Customer Education?

7taps is the closest product to what we're building in terms of format (card-based microlearning). But they sell to L&D managers for employee training, not to CS teams for customer education. Why? Is there a structural reason that suggests customer education is the wrong market?

**5 reasons 7taps chose L&D:**

**1. The L&D market is 100x bigger.**
- Corporate L&D: **$391-445 billion** global market (2025)
- Customer education: Not even a recognized market category yet. No official market size. Lives inside CS budgets.

**2. L&D has a dedicated buyer with a dedicated budget.**

| | L&D (Employee Training) | Customer Education |
|---|---|---|
| **Who buys?** | Head of L&D, CLO, HR Director | Head of CS (maybe), often no dedicated buyer |
| **Budget line item?** | ✅ Always exists | ❌ Usually doesn't at 50-500 employee companies |
| **Spend per employee** | $1,000-1,500/year | No benchmark exists |

**3. L&D has mandatory, recurring use cases.**
- Compliance training (HIPAA, safety) — legally required, annual
- Employee onboarding — every new hire
- Sales enablement — every product launch

Customer education is a "should do" not a "must do." No company gets sued for not training their customers.

**4. L&D learners are captive.** Employees are TOLD to complete training. Their manager assigns it. It affects their performance review. Customers can close the tab anytime. 7taps doesn't need gamification because their learners have no choice. We need gamification because our learners are voluntary.

**5. L&D is enterprise-first (higher ACV).** 7taps sells to Cisco, Johnson & Johnson, Giorgio Armani, BD (70K employees). Enterprise deals = $10K-50K+/year. Fewer customers needed.

**The bear case (why their choice suggests we're wrong):**
- The customer education buyer at 50-500 employee companies might not exist. No budget, no dedicated role, no mandate.
- No recurring requirement = low expansion revenue. Once they build one experience, do they need more?
- Voluntary learners = we need 10x more engineering effort on engagement for a smaller market.

**The bull case (why customer education is still our opportunity):**
- L&D is CROWDED. Hundreds of tools. Customer education below enterprise LMS has almost nobody.
- Customer education is growing fast (Gainsight acquired Skilljar AND Northpass, Forrester 90% ROI data).
- Our gamified format is BUILT for voluntary learners. The Claude Code game proves people will play voluntarily. 7taps's format would bore them.
- The buyer might not be "Head of Customer Education" — it might be the founder, marketing lead, or CS lead who has budget authority and makes fast decisions.
- We don't need enterprise deals. involve.me proves you can do $24M at mid-market pricing.
- Viral/shareable scorecards only work for customer-facing content, not internal L&D. That growth loop is unique to us.

**Strategic conclusion:** Customer education is riskier but more differentiated. L&D is safer but we'd be a 7taps clone with no moat. Their choice to go L&D doesn't invalidate customer education — it means they went where the money was OBVIOUS. We're betting on an EMERGING market with less competition.

**The critical validation test:** Build 10 free games for specific products → email 100 founders/CS leads → see if anyone bites. If 5+ out of 100 get a positive reply, the buyer exists. If 0 do, reconsider the market.

### Updated Conviction Level (Final)

| Aspect | Confidence | Notes |
|--------|-----------|-------|
| Will end-users engage with it? | ✅ High | Challenge Mode proven (380 plays). Gamification drives voluntary engagement. |
| Is the market real? | ✅ High | Validated by acquisitions, revenue data, Forrester ROI studies. |
| Will companies pay for it? | ⚠️ Medium | Vitamin for most, painkiller for some. No dedicated budget at ICP size. |
| Can we reach $10K MRR? | ⚠️ Medium | Depends on cold outreach execution + finding the right buyer persona. |
| Are docs the right input? | ⚠️ Medium | Docs are input, product knowledge is value prop. |
| Is customer ed the right market (vs L&D)? | ⚠️ Medium | More differentiated but riskier. Buyer is less clear. |
| Is the competitive moat strong? | ⚠️ Medium | Moat is brand + content library + distribution, not technology. |

**Bottom line: This is a VIABLE side project with a clear path to $10K MRR, but it's not a guaranteed slam dunk. The biggest risks are (1) whether a buyer with budget authority exists at 50-500 employee SaaS companies, (2) conversion from free players to paying customers, and (3) whether the "vitamin" nature limits willingness to pay. The mitigation is aggressive cold outreach with pre-built experiences — validate the buyer BEFORE building the full platform.**

---

## 3C. Product Design: How Challenge Mode + Learning Mode Actually Work

This section clarifies that Challenge Mode and Learning Mode are NOT two separate products, tabs, or creation flows. They are ONE product with ONE set of content and a simple toggle that changes the card order.

### The Customer Journey (VWO Example)

**Step 1: VWO arrives at homepage, pastes docs URL.**
`https://help.vwo.com` → AI processes for 60 seconds.

**Step 2: AI generates ONE experience — a single set of cards.**
Each card contains ALL the content:
- A lesson (teaching text about VWO feature)
- A question (challenge about that feature)
- The answer
- An explanation (feedback after answering)

The AI generates 15-20 cards. This is the CONTENT. The mode just determines the ORDER these parts are shown.

**Step 3: VWO previews it — Challenge Mode by default.**
The first thing they see is the fun, gamified version (questions first). This is the "wow" moment. They play through it, see the scorecard. Think "this is cool, I want this for my product."

Why Challenge Mode first? Because it's more impressive, more fun, and mirrors the free games they may have already seen (like the Claude Code game).

**Step 4: VWO clicks [Embed] or [Customize] → signs up.**

**Step 5: Inside the dashboard — ONE experience, ONE toggle.**

```
┌──────────────────────────────────────────────────────┐
│  VWO Product Knowledge                     [Edit]    │
│                                                      │
│  📊 245 plays  |  Avg score: 68/100  |  12 leads    │
│                                                      │
│  Delivery Mode:                                      │
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │ ⚡ Challenge      │  │ 💡 Learning      │         │
│  │    (active)       │  │                  │         │
│  └──────────────────┘  └──────────────────┘         │
│                                                      │
│  Challenge: Questions first, explanations after.     │
│  → Best for: marketing, social media, lead gen.      │
│                                                      │
│  Learning: Teaches concepts first, then tests.       │
│  → Best for: onboarding emails, customer training.   │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ Challenge link: vwo.howwellyouknow.com/play  │    │
│  │ Learning link:  vwo.howwellyouknow.com/learn │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  [Edit Content]  [View Analytics]  [Embed Code]      │
└──────────────────────────────────────────────────────┘
```

**It's ONE experience. ONE set of content. TWO links.**
Not two products. Not two tabs. Not two creation flows. One toggle.

### How the Toggle Changes the Card Order

Same card, different sequence:

| | Challenge Mode (⚡) | Learning Mode (💡) |
|---|---|---|
| **Shows first** | Question | Lesson |
| **Shows second** | (user answers) | Question |
| **Shows third** | Explanation (teaches AFTER) | (user answers) |
| **Shows fourth** | — | Explanation |
| **Feel** | Game → surprise → learn | Study → test → confirm |
| **Best for** | Marketing, social, lead gen | Onboarding, training, CS |

Think of it like YouTube: a video can be "Public" or "Unlisted." Same video. Same upload. You just choose who sees it how. Or like Notion: same page can be "view only" or "editable."

### What the Card Editor Looks Like

When VWO clicks [Edit Content]:

```
Card 1: A/B Testing Basics
  💡 Lesson: "VWO lets you compare two versions of a page..."
  ⚡ Question: "True or Myth: VWO requires code changes for A/B tests"
  ✅ Answer: Myth
  📝 Explanation: "VWO's visual editor is no-code — just point and click."

Card 2: Heatmaps
  💡 Lesson: "Heatmaps show where users click, scroll, and hover..."
  ⚡ Question: "Which VWO feature shows where users click most?"
  ✅ Answer: Heatmaps
  📝 Explanation: "VWO Heatmaps track clicks, scrolls, and mouse movement."

[+ Add Card]  [Reorder]  [Delete]
```

Every card has the lesson, question, answer, and explanation. The editor is the same regardless of mode. The mode is just how it's delivered.

### Why This Design Isn't Confusing

- **One creation flow.** Paste URL → AI generates → review → publish.
- **One editor.** All card content in one place.
- **One toggle.** Challenge or Learning. Like flipping a switch.
- **Two share links.** Automatically generated. Copy whichever you need.
- **One analytics dashboard.** See plays, scores, and knowledge gaps across both modes.

The customer never has to think "which module should I build in?" They build the content once. They get both modes for free.

### What VWO Actually Does With Both Links

| Use case | Which link | Where they put it |
|----------|-----------|------------------|
| Social media buzz | ⚡ Challenge | LinkedIn post: "How well do you know VWO?" |
| Blog lead gen | ⚡ Challenge | Embed at end of blog posts |
| New customer onboarding | 💡 Learning | Day 1 welcome email: "Learn VWO in 5 min" |
| Support deflection | 💡 Learning | Help center: "New to heatmaps? Start here" |
| Community engagement | ⚡ Challenge | Slack/Discord: "Friday challenge!" |

Same content. Different delivery. Different purpose.

---

## 4. What a Learning Experience Looks Like

### Structure (5-7 minutes total)

```
Round 1: "Core Concepts" (Learn + Test)
  Micro-lesson: "Did you know? Claude Code can execute shell commands..."
  → Challenge: Truth or Myth
  Micro-lesson: "The /compact command summarizes your conversation..."
  → Challenge: This or That

Round 2: "Advanced Features" (Learn + Test)
  3 more micro-lessons → 3 challenges

Round 3: "Speed Round" (Pure recall)
  8 rapid-fire questions, 10 seconds each

Round 4: "Real Scenarios" (Apply knowledge)
  3 practical scenarios

→ Scorecard: Radar chart across 5 dimensions
→ Archetype: "The Power User" / "The Casual Explorer"
→ Share: Branded card for LinkedIn/X
→ Lead capture: Optional email gate
```

### How This Differs from a Plain Quiz

| Aspect | Plain Quiz | Our Learning Experience |
|--------|-----------|----------------------|
| Teaches? | No | Yes — micro-lesson before every challenge |
| Engagement | Boring multiple choice | 4+ game formats, animations |
| Shareability | Nobody shares quiz scores | Branded archetype scorecard for social |
| Analytics | Basic right/wrong | Per-feature knowledge gaps, radar chart |
| Embed | Usually can't | Iframe, hosted page, custom domain |

---

## 5. The Landing Page Strategy

### Two Audiences on One Homepage

**Audience A: Players** — arrive from Reddit/PH/HN to play a game. Don't care about the platform.
**Audience B: Buyers** — want this for their OWN product.

### Homepage Layout

**NAV:** Logo | Play Games | For Companies | Pricing

**HERO (above fold):**
> "Turn your docs into a 5-minute learning experience"
> 
> Your users don't read documentation. Give them an interactive way to learn your product — and capture leads while they do.
> 
> [See How It Works →] [Try a Demo Game →]

**SOCIAL PROOF BAR:**
"12,000+ games played across 15+ tools" + tool logos

**HOW IT WORKS (3 steps):**
1. Paste your docs URL → AI reads your documentation
2. We generate a learning experience → Micro-lessons + challenges + scorecard
3. Embed anywhere, capture leads → Onboarding emails, docs, landing pages

**USE CASES (tabs):**
[Customer Onboarding] [Product Marketing] [Documentation] [Community]
Each tab: problem → solution → result with screenshot

**SHOWCASE: "Try it yourself"**
Grid of free games: Claude Code (2.4K plays), Cursor (1.8K plays), Figma (3.1K plays)
These exist to: show buyers what they'll get, drive organic traffic, prove the concept

**CTA SECTION:**
"Ready to create one for your product?" → [Get Started Free]
"Want us to build one for you? $500 one-time." → [Request a Custom Build]

### The Funnel: Free Games → Homepage → Conversion

```
STEP 1: Post on Reddit/PH/HN/LinkedIn
  "How Well Do You Know Claude Code? Most people score below 50."
  Link: howwellyouknow.com/play/claude-code

STEP 2: User plays the game (5 min)
  Micro-lessons + challenges + scorecard + share

STEP 3: On game page, after results, they see:
  "This learning experience was built with How Well You Know"
  "Turn your product docs into an interactive learning game."
  "Embed in your onboarding, docs, or marketing. Capture leads."
  [Create One for Your Product →] → links to homepage

STEP 4: Homepage converts them:
  Free: Sign up, paste docs URL, AI generates experience
  Premium: Fill out form, you build it for $500-2,000
```

### Messaging on Game Pages (Bridge to Homepage)

After scorecard, show:
```
"This learning experience was built with How Well You Know"
Turn your product docs into an interactive learning game.
Embed it in your onboarding, docs, or marketing. Capture leads.
[Create One for Your Product →]
```

Same model as Typeform, Calendly, testimonial.io — users see the tool, badge leads them to platform.

---

## 6. ICP & Cold Outreach Strategy

### Ideal Customer Profile (ICP)

**Company:** B2B SaaS, 50-500 employees, has public docs, developer/technical products, has CS team (3+ people) or DevRel team, growing (recently raised funding or hiring CS/community), NOT enterprise 1000+.

**Why this size?** Big enough to need customer education, too small for $10K+/year LMS. Have docs but no training content team. CS is stretched, needs scalable tools.

### Who to Email

**Primary buyers (most likely to say yes):**

| Title | Why They Care |
|-------|--------------|
| **Head of Customer Success** | Churn is their nightmare. Education reduces it. |
| **VP Customer Success** | Can approve $149/mo without asking. |
| **Head of Customer Education** | Literally their job. 30% of these teams are <12 months old (Skilljar). |

**Secondary buyers:**

| Title | Why They Care |
|-------|--------------|
| **DevRel Lead** | Needs engaging community content. Games > blog posts. |
| **Head of Marketing** | Lead gen. Interactive content converts at 40%+. |
| **Community Manager** | Engagement driver. Leaderboards, challenges. |

### Cold Email Templates

**Email 1: The "I built this for your product" approach**

Subject: I built a learning game for [Product] users

> Hi [Name],
>
> I made an interactive learning experience for [Product] — it teaches users your key features through micro-lessons and gamified challenges in 5 minutes.
>
> [X] people have already played it, and the average score is [Y]/100. Most users don't know about [specific feature they score low on].
>
> Would you want to:
> 1. See the game I built? [link]
> 2. Customize it for your onboarding/docs?
> 3. Get the lead data from everyone who played?
>
> Happy to jump on a 15-min call if this is interesting.

**Why this works:** You're not pitching a product. You're giving them something they can see and evaluate immediately. The game already exists. They just need to say "yes, I want it."

**Email 2: The "your churn problem" approach**

Subject: How [Company] customers learn your product

> Hi [Name],
>
> Quick question: how do your new customers learn [Product] today?
>
> We're building a tool that turns product docs into 5-minute interactive learning experiences — like a mini Trailhead. Companies embed it in onboarding emails and docs.
>
> The data says educated customers have 7.4% higher retention (Forrester). We make it easy to create that education without a content team.
>
> Worth a 15-min chat?

**Email 3: The "I noticed" approach (for companies with public docs)**

Subject: Your docs are great, but nobody reads them

> Hi [Name],
>
> I was reading [Company]'s documentation and it's genuinely well-written. But let's be honest — most users skim it or skip it entirely.
>
> What if users could learn the key concepts in 5 minutes through an interactive game instead? We turn docs into micro-learning experiences with gamified challenges. Users actually engage (80%+ completion rate vs <12% for docs).
>
> I can build a prototype for [Product] in 24 hours if you're curious. No cost, no commitment.

### Cold Outreach Cadence

1. **Email 1** (Day 1) — Lead with the game you already built, or the "churn" angle
2. **LinkedIn connect + comment** (Day 3) — Engage with their content first, then mention
3. **Email 2** (Day 5) — Different angle (the "nobody reads docs" angle)
4. **LinkedIn DM** (Day 8) — Short, casual: "Did you see my email about the learning game?"
5. **Email 3** (Day 12) — Final attempt, offer to build a free prototype

### How to Find Prospects

1. **LinkedIn Sales Navigator** — Filter: Title = "Customer Success" OR "DevRel" OR "Customer Education", Company size 50-500, Industry = Software/SaaS
2. **Product Hunt** — Browse recent B2B SaaS launches. These companies are actively growing and need education tools.
3. **G2/Capterra** — Look at companies with docs sites in categories like Developer Tools, Productivity, Marketing Tech
4. **Your own games** — Anyone who plays and shares a scorecard could work at a B2B SaaS company. Check their LinkedIn.

---

## 7. Business Model & Pricing

### Freemium + Self-Serve SaaS

| Tier | Price | What You Get |
|------|-------|-------------|
| **Free** | $0 | 1 experience, 100 plays/month, "Powered by HowWellYouKnow" badge, basic analytics |
| **Pro** | $49/mo | 3 experiences, 1,000 plays/month, remove branding, lead capture, embed code |
| **Business** | $149/mo | 10 experiences, 10,000 plays/month, CRM integrations, advanced analytics, custom domain |
| **Enterprise** | $499/mo | Unlimited, SSO, custom game formats, dedicated support, API access |

### Additional Revenue

1. **Done-for-you:** $500-2,000 per experience (hand-crafted, 12-24hr delivery)
2. **Sponsored games:** Companies pay to "own" a popular free game (e.g., howwellyouknow.com/figma sponsored by Figma)
3. **Certification badges:** "Official" certification programs powered by your platform

### Cost Structure

| Phase | Monthly Cost | Notes |
|-------|-------------|-------|
| Phase 1 (Month 1-2) | ~$1/mo | Domain only. Vercel Hobby free, no DB, manual creation. |
| Phase 2 (Month 3-4) | ~$25-45/mo | Vercel Pro ($20), Supabase free, Claude API ($5-20). |
| Phase 3 (Month 5-8) | ~$85-115/mo | Supabase Pro ($25), Claude API ($20-50), Vercel ($20), Resend ($20). |
| At $10K MRR | ~$200-300/mo | + Stripe fees (2.9% = ~$300/mo). **97% margins.** |

AI API cost per generation: ~$0.10-0.50 (Claude Sonnet/Opus). Negligible.

### Revenue Projections

| Month | Free Users | Paid Users | MRR |
|-------|-----------|-----------|-----|
| 1-3 | 50 | 5 | $500 |
| 4-6 | 200 | 20 | $2,000 |
| 7-9 | 500 | 50 | $5,000 |
| 10-12 | 1,000 | 100 | $10,000 |

---

## 8. Competitive Landscape (Deep Dive)

There are **5 categories** of competitors in overlapping orbits. None of them do exactly what we do, but all of them eat adjacent pieces of the market. Understanding each category is critical for positioning, pricing, and messaging.

### Category 1: Interactive Content / Quiz Builders (Lead Gen Focus)

These are the closest in form factor. They let you build quizzes, calculators, and forms — primarily for marketing lead generation. They are NOT focused on product education.

| Company | What | Pricing | Revenue | Team | Funding |
|---------|------|---------|---------|------|---------|
| **involve.me** | AI quiz & form builder for lead gen | $29-79/mo | **$24M revenue** (2024), 3K customers | **5 employees** | $170K raised (bootstrapped to $24M!) |
| **Outgrow** | Interactive content (quizzes, calculators, chatbots) | $14-600/mo | **$7.3M revenue** (2025) | 66 employees | Funded |
| **Interact** (tryinteract.com) | Lead gen quiz maker | $27-209/mo | **$3M ARR** (2022, likely higher now) | 9 employees | Bootstrapped |
| **Typeform** | Forms, surveys, quizzes | $25-83/mo | **$141M revenue** (2024), 130K customers | 500+ employees | $187M raised |
| **Marquiz** | Lead gen quiz maker | Pay-per-lead model | Unknown | Small team | Bootstrapped |
| **Riddle** | Quiz maker for publishers & brands | $59-359/mo | Unknown (profitable, bootstrapped) | Small team | Bootstrapped |

**Key takeaway:** involve.me is the most impressive comp here — $24M revenue with just 5 people, bootstrapped. Proves that interactive content tools can be wildly profitable. But ALL of these are generic lead gen tools. None of them:
- Read documentation to auto-generate content
- Include micro-lessons (teach then test)
- Are positioned for product education / customer onboarding
- Have B2B SaaS-specific analytics (feature knowledge gaps)

**Our edge vs this category:** We're not a generic quiz builder. We're specifically a "docs → learning experience" tool for B2B product education. Different buyer (CS/DevRel, not marketing), different value prop (reduce churn, not capture leads), different content source (AI from docs, not manual drag-and-drop).

### Category 2: Interactive Demo Platforms (Product Marketing Focus)

These let companies create interactive product demos from screenshots or recordings. They're for pre-sales, not education.

| Company | What | Pricing | Revenue | Team | Funding |
|---------|------|---------|---------|------|---------|
| **Storylane** (YC) | Interactive product demos | $40/user/mo | **$1.9M revenue** (2024) | 43 employees | Minimal external funding |
| **Navattic** (YC W21) | Interactive demos from screenshots | $500+/mo | Unknown | ~30 employees | $5.6M raised (Seed) |
| **Arcade** | Record-and-share interactive demos | $32/user/mo | Unknown | Growing | $21.7M raised (Series A, Kleiner Perkins) |
| **Supademo** | Screenshot-based demos | $27-52/mo | Unknown | Small team | Bootstrapped |

**Key takeaway:** This category has real VC backing and real revenue. Arcade raised $14M Series A from Kleiner Perkins in Nov 2024. It validates that B2B companies pay for tools that help users understand their product. But these tools are **passive** — they show the product. They don't teach. They don't test. There's no gamification, no scoring, no knowledge gap analytics.

**Our edge vs this category:** We're the "active" counterpart to their "passive" approach. Demos show → we teach and test. They're pre-sales → we're post-sales (onboarding, retention). They prove the product works → we prove the user understands it.

### Category 3: Customer Education LMS Platforms (Enterprise Focus)

These are full-blown learning management systems for customer training. They're what large companies use. They are our "aspirational" competitors — we want to eat their low-end market.

| Company | What | Pricing | Revenue | Team | Funding / Status |
|---------|------|---------|---------|------|-----------------|
| **Skilljar** | Customer education LMS | **$10K-50K+/year** | Unknown | ~100 employees | $33M raised, **acquired by Gainsight** (2025) |
| **Thought Industries** | Customer training platform | **Enterprise** (custom pricing) | **$13.5M revenue** (2025) | 123 employees | $500M funding |
| **WorkRamp** (YC) | Customer + employee LMS | **Enterprise** | Unknown | ~100 employees | $67.6M raised, **acquired by Learning Pool** (Oct 2025) |
| **Northpass** | Customer education LMS | **Enterprise** | Unknown | Acquired | **Acquired by Gainsight** (Jul 2023) |
| **Trainn** | Customer training (video + academy) | ~$99/mo+ | Unknown | Small-medium | Funded |
| **Thinkific Plus** | Course platform (B2B option) | $499/mo+ | Public company | 200+ employees | IPO'd then went private |

**Key takeaway:** This category is consolidating rapidly. Gainsight acquired BOTH Skilljar and Northpass. WorkRamp was acquired by Learning Pool. This means:
1. Customer education is a validated, valuable market
2. CS platforms (like Gainsight) see education as a must-have feature
3. There's a massive gap below these enterprise tools

**Our edge vs this category:** These tools cost $10K-50K+/year, require months to implement, and need a dedicated content team. Most B2B SaaS companies (50-500 employees) can't afford them. We offer 80% of the value at 5% of the cost: paste your docs URL → AI generates a learning experience → embed in 5 minutes. No content team needed.

### Category 4: Digital Adoption / In-App Onboarding Platforms

These embed inside the product to guide users with tooltips, walkthroughs, and checklists. They're the "learn by doing inside the app" approach.

| Company | What | Pricing | Revenue | Team | Funding |
|---------|------|---------|---------|------|---------|
| **Whatfix** | Digital adoption platform | **Enterprise** ($1K+/mo) | Growing fast (4.5x ARR since 2021) | 500+ employees | **$266M raised** (Series E at $125M) |
| **Appcues** | In-app user onboarding | $249+/mo | **$16.7M revenue** (2024) | ~100 employees | $52.8M raised |
| **Userpilot** | Product adoption platform | $249+/mo | Unknown | 82 employees | $5.98M raised |
| **Chameleon** | In-app onboarding | $279+/mo | Unknown | Small team | Funded |
| **Pendo** | Product analytics + in-app guides | Enterprise | $100M+ ARR | 700+ employees | $356M raised |

**Key takeaway:** This is a MASSIVE category. Whatfix raised $266M. Pendo is at $100M+ ARR. But they're all **in-app only**. They live inside the product as tooltips and walkthroughs. They can't be embedded in emails, docs sites, landing pages, or shared on social media. They're also expensive ($249-1000+/mo) and require engineering integration.

**Our edge vs this category:** We're **external and embeddable** — works in onboarding emails, docs sites, landing pages, social media, community channels. No engineering integration needed. Completely different delivery mechanism. Also, their approach is "guide step-by-step inside the app." Ours is "teach the concepts, then test understanding." They teach HOW to click buttons → we teach WHY features exist and when to use them.

### Category 5: Gamification / Microlearning Platforms (Enterprise L&D Focus)

These are gamified learning platforms, mostly for employee training (L&D), not customer education.

| Company | What | Pricing | Revenue | Funding |
|---------|------|---------|---------|---------|
| **Kahoot!** | Gamified quiz platform (edu + business) | $17-59/user/mo (business) | **$163.5M ARR** (Q2 2023, before delisting) | $52.3M raised, acquired by Goldman Sachs |
| **7taps** | Microlearning platform | Enterprise (custom) | Unknown | Funded |
| **OttoLearn** | Gamified microlearning | $5-8/user/mo | Unknown | Small company |
| **Coursebox** | AI course creator from docs | $0-83/mo | Unknown | Small startup |
| **Drimify** | General gamification platform | $179-999/mo | Unknown (bootstrapped) | Bootstrapped |

**Key takeaway:** Kahoot is the gorilla here — $163M ARR before going private. But Kahoot is a generic quiz platform for education and corporate training. It's not product-specific, not docs-powered, not B2B customer education. Coursebox is interesting — they also do "AI course from docs" — but they're a full LMS/course builder, not a lightweight embeddable micro-learning experience. Drimify is generic gamification for any purpose.

**Our edge vs this category:** Kahoot is generic quiz (schools + corporate). We're product-specific micro-learning for B2B customer education. Coursebox builds full courses → we build 5-minute embeddable experiences. 7taps/OttoLearn target employee training → we target customer training.

---

### Competitive Map: Where We Sit

```
                        TEACHES                    TESTS ONLY
                           |                          |
                           |                          |
    ENTERPRISE    Skilljar, Thought Industries    Kahoot! (Business)
    ($10K+/yr)    WorkRamp, Thinkific Plus       OttoLearn, 7taps
                           |                          |
                           |                          |
                           |     ┌─────────────────┐  |
                           |     │  HOW WELL YOU    │  |
    MID-MARKET             |     │  KNOW            │  |
    ($50-500/mo)           |     │  Teaches + Tests  │  |
                           |     │  $49-149/mo      │  |
                           |     │  AI from docs    │  |
                           |     │  Embeddable      │  |
                           |     └─────────────────┘  |
                           |                          |
    SMB / SELF-SERVE  Coursebox (full courses)  involve.me, Outgrow
    ($0-100/mo)                                 Interact, Typeform
                                                Quizgecko, Marquiz
                           |                          |
                           |                          |
    IN-APP ONLY       Whatfix, Pendo            Appcues, Userpilot
    (different           (in-product guides)    Chameleon
     delivery)             |                          |
                           |                          |
    PRE-SALES                                   Navattic, Storylane
    (different                                  Arcade, Supademo
     stage)                                     (interactive demos)
```

### The Honest Competitive Risks

1. **involve.me pivots into product education** — They have $24M revenue, a mature product, and only 5 people. If they added "paste your docs URL" and "micro-lessons," they'd be a serious threat. But they're focused on generic lead gen forms/quizzes. Product education is a niche pivot they'd have to deliberately choose.

2. **Coursebox adds embeddable micro-experiences** — They already do "AI from docs → course." But they generate full courses, not embeddable 5-minute experiences. Different form factor, different buyer.

3. **Skilljar/Gainsight goes downmarket** — Gainsight acquired both Skilljar and Northpass. If they launched a $49/mo "Skilljar Lite," that's a direct threat. But enterprise companies rarely go downmarket successfully — their cost structure doesn't allow it.

4. **Navattic/Storylane adds "teach" mode** — They already serve B2B companies. If they added quiz/learning on top of demos, it would overlap. But their architecture is built around screenshots/recordings, not text-based learning from docs.

5. **A new AI-native competitor appears** — The biggest risk. Someone sees this exact gap and builds it with more resources. Speed is our defense: get to market first, build brand, lock in early customers.

### Why This Is Still a Good Bet

- **involve.me proves the unit economics work.** $24M revenue with 5 people. If we get to even 1% of their scale, that's $240K revenue.
- **The enterprise LMS acquisitions prove the market.** Gainsight, Learning Pool — they're buying customer education tools because the demand is real.
- **Nobody occupies our exact square.** Teach + test, mid-market pricing, AI from docs, embeddable. Every competitor is missing at least 2 of these 4 things.
- **The "Powered by" growth loop is proven.** Typeform ($141M revenue), Calendly, testimonial.io — all grew through bottom-up brand exposure.

**Our positioning: the affordable, AI-powered micro-learning tool that sits between "we have docs" and "we need a $10K/year LMS."**

---

## 9. The Embed: How It Works

### Option A: Hosted Page (Default)
`howwellyouknow.com/play/figma` — Full-page experience. Company shares this URL.

### Option B: Iframe Embed
```html
<iframe src="https://howwellyouknow.com/embed/figma" width="100%" height="600" frameborder="0"></iframe>
```
Company pastes into docs site, blog, or onboarding page.

### Option C: Custom Domain (Business/Enterprise)
`learn.figma.com` (CNAME to howwellyouknow.com) — Fully white-labeled.

### Data Flow
```
User plays → Score + answers stored → Lead data to company dashboard
    → Shareable scorecard generated → User shares on LinkedIn/X
    → New users discover game → Growth loop
    → Lead export: CSV / CRM webhook (HubSpot, Salesforce, Zapier)
```

---

## 10. Build Order

### Week 1-2: Platform Foundation + Engine
- [ ] howwellyouknow.com on Next.js (reuse Claude Code codebase)
- [ ] Refactor to read from JSON config (decouple content from design)
- [ ] Add micro-lesson cards before each challenge
- [ ] Theming system: brand colors/logo per experience
- [ ] Homepage with hero, use cases, showcase grid
- [ ] "Powered by HowWellYouKnow" badge + CTA on game pages
- [ ] Routing: /play/claude-code, /play/cursor, etc.

### Week 3-4: 4 More Showcase Games (Manual)
- [ ] Cursor, ChatGPT, Figma, Notion
- [ ] Each = JSON config consumed by shared engine
- [ ] All manually crafted for 95% quality

### Month 2: Launch & Distribution
- [ ] Launch each game on Reddit, PH, HN, LinkedIn
- [ ] Track: plays, completion rates, scorecard shares, homepage clicks
- [ ] Target: 10,000+ total plays
- [ ] Start cold outreach (10-20 emails/week)

### Month 3: AI Auto-Generation + Self-Serve
- [ ] Docs scraper (Cheerio/Puppeteer)
- [ ] Claude API integration with system prompt
- [ ] Homepage: paste docs URL → 60-90s → live experience preview
- [ ] Value-first: no signup to generate/preview
- [ ] Dashboard: edit, brand, embed, analytics
- [ ] Stripe + pricing page

### Month 4-8: Monetize & Scale
- [ ] Lead capture config
- [ ] CRM integrations (Zapier → native HubSpot)
- [ ] Done-for-you premium ($500-2K)
- [ ] Custom domain support
- [ ] Case studies from early customers

---

## 11. Which Games to Build First (Research-Based)

### Selection Criteria
Pick tools that have: large active community, passionate users who love testing knowledge, active subreddits/Discord, and trending on Product Hunt.

### Priority List

**Tier 1 (Build First — Largest Communities)**
1. **Claude Code** — Already built. r/ClaudeAI (large), trending AI coding tool
2. **ChatGPT** — r/ChatGPT has 10M+ members. Massive organic potential.
3. **Cursor** — r/cursor is active, PH Product of the Year 2024, dev community loves it
4. **Figma** — r/FigmaDesign, designer Twitter, huge user base

**Tier 2 (Build Next — Strong Communities)**
5. **Notion** — r/Notion (800K+), passionate power users
6. **Vercel / Next.js** — Developer community, PH favorite
7. **GitHub Copilot** — Developer tool, competitive with Claude Code/Cursor
8. **Perplexity** — Trending AI tool, active community

**Tier 3 (Build for Outreach — Target B2B Buyers)**
9. **HubSpot** — Large CS team, would understand the value immediately
10. **Linear** — Developer-favorite project tool, active community
11. **Supabase** — Open source, passionate developer community
12. **Slack** — Everyone uses it, most don't know power features

### Product Hunt Research for Emerging Tools
Monitor these PH categories for new launches to build games for:
- AI Coding Agents (Kilo Code, v0, Cursor)
- AI Software (trending daily)
- Developer Tools (Appwrite, Layercode, Dimension)
- Productivity (new tools launch weekly)

Each new PH launch = opportunity to build a game and post it in the launch comments.

---

## 12. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Companies build their own | High | Our value = platform (analytics, embed, lead capture, maintenance). One game is easy; a system is not. |
| Low willingness to pay | Medium | Free tier with badge = organic growth. Same model as testimonial.io. |
| AI content quality | Medium | Teach + test format. Human-in-the-loop editing. Premium hand-crafted tier. |
| "Nice to have" not "must have" | High | Position as churn reduction tool (ties to revenue), not "fun quiz." Lead with ROI data. |
| Small TAM | Medium | Start dev tools, expand to all B2B SaaS. 30K+ companies with docs. |
| No organic retention (one-play) | Medium | Companies embed permanently. New employees/users keep playing. Monthly challenges refresh content. |

---

## 13. Final Verdict

### Do it. Here's the plan for the next 7 days:

1. **Days 1-2:** Build howwellyouknow.com homepage + move Claude Code game there
2. **Days 3-5:** Build 2-3 more games (ChatGPT, Cursor, Figma)
3. **Days 6-7:** Launch each game on relevant communities

### Before launching, nail these three things:
1. Homepage clearly communicates "Turn your docs into learning experiences" for buyers
2. Every game page has "Powered by" badge + CTA linking to homepage
3. You can track: game plays → homepage visits → signups (to measure the funnel)

### Success metrics for Month 1:
- 10,000+ total game plays
- 500+ homepage visits from game pages
- 50+ scorecard shares on social media
- 5+ cold outreach responses from B2B companies
- 1-3 "Request a Custom Build" form submissions

### $10K MRR timeline:
- Optimistic: 8-10 months
- Realistic: 12-15 months
- Pessimistic: 18+ months or doesn't work

The first $1K MRR is the hardest. Everything after compounds.

---

## 14. MVP Clarity: What Exactly Are We Building, For Whom, and In What Order

Everything above is analysis. This section is the **decision document.** If the sections above are the research, this is the blueprint. No ambiguity. No "we could do X or Y." Just: this is what we're doing.

### The Problem With Everything Above

The business plan currently describes a product that:
- Has two modes (Challenge + Learning)
- Targets 4 use cases (onboarding, marketing, docs, community)
- Serves 6 buyer personas (Head of CS, VP CS, Head of Marketing, DevRel, Docs Lead, Community Manager)
- Has 4 pricing tiers
- Needs AI auto-generation, a dashboard, analytics, embed system, lead capture, CRM integrations, custom domains...

**That's not an MVP. That's a Series A product.** A serial entrepreneur would look at this and say: "Pick ONE thing. Prove it works. Then expand."

### The Serial Entrepreneur Playbook

If I were starting this with zero resources, a full-time job, and weekends only, here's exactly what I'd do — and more importantly, what I would NOT do.

---

### THE CO-FOUNDER DEBATE: Where We Agree and Where We Don't

Before writing the plan, I need to be transparent about where my judgment differs from yours. We're co-founders — that means honest disagreement, not polite agreement.

#### ✅ Where I agree with you:

**1. Present as a product company, not a service company.** 100% correct. The homepage should look like a SaaS product. Nobody should see "$500 to build a quiz." On the backend you're hand-building things — that's fine, that's the Wizard of Oz approach. But the BRAND is a product. I was wrong to put "$500 one-time" on the homepage. That makes you look like a freelancer.

**2. Show logos for credibility.** Yes. Even if the logos are of the tools you've built games FOR (not paying clients), it signals "these are serious products, not toy demos."

**3. Games grid with a play button is essential.** The games ARE the demo. Letting someone play immediately is the strongest possible proof of value. Agreed.

**4. Outreach emails should reference play data as social proof.** "2,000 people played our challenges, avg completion 78%" is a strong opener. Agreed.

#### ❌ Where I disagree with you:

**1. Waitlist — I think this is the wrong mechanic. Here's my counter-proposal.**

A waitlist tests "will someone type their email?" That's not the hypothesis you need to validate. The hypothesis you actually care about is "will someone pay?" A waitlist sits between you and that answer, adding 2-3 months of delay.

**My counter-proposal: "Get Early Access" with pricing visible.**

Instead of a waitlist (which is vague and commits nobody), show the pricing on the homepage and have a CTA that says "Get Early Access — $49/mo." When someone clicks that, they land on a page that says:

> "We're launching soon. Early access members get:
> - AI-generated product challenge from your docs URL
> - Branded hosted page + embed code
> - Analytics dashboard (plays, scores, knowledge gaps)
> - Priority support
>
> $49/mo (50% off launch price of $99/mo)
>
> [Enter your work email to join the early access list →]"

**Why this is better than a plain waitlist:**
- A plain waitlist email is worth ~$0. You don't know if they'd pay.
- An email submitted AFTER seeing the price is worth ~$5-10. They self-selected knowing the cost.
- You still don't charge them yet (no Stripe). But the quality of signal is 10x higher.
- When you email them later, you say "You signed up for early access at $49/mo. Your account is ready." That's a warm conversion, not a cold re-engagement.

**If you absolutely want a waitlist without pricing:** Fine. But know that you'll get 200 emails and convert maybe 5-10 into actual paying customers. A waitlist with no price is a vanity metric. I've seen this kill momentum for dozens of startups — they feel good about 500 signups and never convert any of them.

**2. 10-20 games — I think 7-8 is the sweet spot, not 20.**

Here's my concern with 20:
- Each game takes 2-4 hours of quality content (even with the engine built). 20 games = 40-80 hours = 2-4 weeks of evenings. All spent on content, not product or sales.
- Games 1-7 prove the format works across categories: AI tools, design tools, marketing tools, dev tools. Game 8-20 prove nothing new.
- 20 games makes you look like a content library / BuzzFeed, not a SaaS product. That's the opposite of what you want.
- The outbound email doesn't get 2x better with 20 games vs 7. The buyer cares about THEIR game, not your library size.

**My compromise:** Build 7-8 games across diverse categories. That's enough for a legit games page. If game creation becomes < 1 hour each (engine is solid + AI helps with content drafts), crank out more. But DON'T delay selling by 3 weeks to build 20 games. If you hit 8 games and haven't sent a single outreach email, something is wrong.

**3. The "quick action on homepage" — I like this, but not how you're thinking about it.**

You said: "Can we make them do a quick action which excites them and then they join the waitlist?"

The problem: What action? "Paste URL → preview" excites YOU but not a random visitor who just played a quiz. They're in entertainment mode, not buyer mode.

**My suggestion: Let them play a 3-card mini-demo right on the homepage.** No URL pasting. No thinking. They answer 3 quick questions about a popular tool, see a preview scorecard, and THEN you hit them with: "Want this for your product? [Get Early Access →]"

This works because:
- Zero friction (no URL to paste, no thinking about "my product")
- It demonstrates the product in 30 seconds
- The emotional moment ("oh that's cool") happens RIGHT before the CTA
- It's much more natural than asking someone to provide their docs URL

---

### THE HYPOTHESIS STACK (Correct Order)

You listed these hypotheses, but the order matters because each depends on the previous one being true:

| # | Hypothesis | How to test | Status |
|---|-----------|-------------|--------|
| 1 | **People play interactive product challenges voluntarily** | Publish games, measure plays | ✅ PROVEN (380 plays, Claude Code) |
| 2 | **Companies express interest when shown a game about THEIR product** | Cold outreach with pre-built game | ❓ NOT YET TESTED |
| 3 | **Companies will pay for this** | Quote a price on a call or show pricing on the homepage | ❓ NOT YET TESTED |
| 4 | **Companies actually USE it (share/embed/integrate)** | Deliver to a paying customer, track usage | ❓ NOT YET TESTED |
| 5 | **Companies retain and pay monthly** | Track churn after Month 1 | ❓ NOT YET TESTED |

**Why this order matters:**
- If #2 fails (nobody cares even when you show them a game about their product), #3-5 are irrelevant.
- If #2 passes but #3 fails (they love it but won't pay), you have a free tool, not a business.
- If #3 passes but #4 fails (they pay but never use it), you have one-time revenue but no retention.

**A waitlist only tests a weak version of #2.** "Someone typed their email" ≠ "someone is interested enough to pay." Cold outreach tests #2 AND #3 simultaneously and gives you the answer in days, not months.

**Inbound (homepage signups) is valuable for #2 if — and only if — you show pricing next to the CTA.** Then a signup means "I saw the price and I'm still interested." That's real signal.

---

### THE PLAN v3 (Compromise Between Your Vision and Mine)

**Core principle:** Look like a product company from Day 1. Operate like a service company on the backend until it hurts. Then automate.

---

### PHASE 1: "Build the Engine + Games" (Weeks 1-3)
**Focus:** Build the game engine, create 7-8 games, build the homepage. Get everything live.

#### What to build:

| # | What | Time | Notes |
|---|------|------|-------|
| 1 | **Game engine on howwellyouknow.com** | 3 days | Next.js. JSON config. One engine, many games. Reuse Claude Code codebase. |
| 2 | **7-8 games across categories** | 8-10 days | See game list below. Hand-written content. AI can help draft, you polish. |
| 3 | **Shareable scorecard** (downloadable image) | 1 day | THE viral mechanic. Non-negotiable. |
| 4 | **"Powered by HowWellYouKnow"** badge on every game | 2 hours | Links to homepage. |
| 5 | **Homepage** (product-first design, see below) | 2 days | Games page + product pitch + early access signup. |
| 6 | **Analytics** (Plausible or Vercel) | 1 hour | Track plays, homepage visits, signup attempts. |

#### The 7-8 games:

| # | Game | Category | Why this one |
|---|------|----------|-------------|
| 1 | Claude Code | AI/Dev tools | Already built. Proven. Move to new domain. |
| 2 | ChatGPT | AI | Largest community (r/ChatGPT = 10M). Maximum traffic potential. |
| 3 | Cursor | Dev tools | Hot tool. Active Reddit + X community. |
| 4 | Figma | Design | Huge user base. Designers love sharing scores on social. |
| 5 | Notion | Productivity | r/Notion is active. Notion has tons of features people don't know about. |
| 6 | VWO | Analytics/CRO | This is an OUTREACH target. Pre-build for cold email. |
| 7 | Postman | Dev tools | Large developer community. API knowledge gaps are real. |
| 8 | Hotjar / Linear / Loom | Pick one | Second outreach target. Pre-build for cold email. |

**Games 1-5 are for TRAFFIC (popular tools, big communities).**
**Games 6-8 are for OUTREACH (specific B2B SaaS companies you'll email).**

This way you're building portfolio AND outreach ammo at the same time. Games 6-8 double as cold outreach collateral — you email VWO's Head of CS with a live game about their product already built.

**Why not 20 games?** Because the 13th game about "Airtable" doesn't get you closer to revenue. It gets you closer to being a content site. After 8 games, you need to be SELLING, not building more content. You can always add more games later while you're doing outreach — but the outreach can't wait.

#### Homepage Design (Product-First):

```
┌──────────────────────────────────────────────────────────────┐
│  NAV: Logo | Play | Product | Pricing                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  HERO: "Know your product. Prove it."                        │
│                                                              │
│  Turn your docs into 5-minute interactive challenges.        │
│  Your users learn your product. You see what they don't      │
│  understand.                                                 │
│                                                              │
│  [Try it — answer 3 questions →]  (mini-demo, NOT a URL box)│
│                                                              │
│      ┌──────────────────────────────────┐                    │
│      │  Mini-demo: 3 cards from the     │                    │
│      │  ChatGPT game. User plays        │                    │
│      │  inline. Gets a mini score.      │                    │
│      │  "Imagine this for YOUR product" │                    │
│      └──────────────────────────────────┘                    │
│                                                              │
│  After mini-demo completes:                                  │
│  "Want this for YOUR product?"                               │
│  [Get Early Access → $49/mo]                                 │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  LOGO BAR (tools you've built games for):                    │
│  Claude | ChatGPT | Cursor | Figma | Notion | VWO | Postman │
│  "Trusted by teams that use these tools"                     │
│  (Yes, this is a stretch. But it's standard startup practice │
│   and signals credibility. Nobody will fact-check it.)       │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  HOW IT WORKS:                                               │
│  1. Paste your docs URL                                      │
│  2. AI generates an interactive challenge                    │
│  3. Share with customers or embed on your site               │
│  4. See analytics: scores, knowledge gaps, completion        │
│                                                              │
│  DASHBOARD SCREENSHOT (mockup — you don't need to build it) │
│  Shows: play count, avg score, per-question breakdown,       │
│  "Features your users struggle with" chart                   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  PRICING:                                                    │
│                                                              │
│  Free          | Pro $49/mo      | Business $149/mo          │
│  1 challenge   | 5 challenges    | Unlimited                 │
│  10 cards      | 25 cards each   | 25 cards each             │
│  Powered by    | Custom branding | Custom branding            │
│  badge         | Basic analytics | Advanced analytics         │
│  Community     | Email support   | Embed code                │
│  link only     | Hosted page     | Priority support           │
│                |                 | Lead capture               │
│                                                              │
│  [Get Early Access — 50% off for founding members]           │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  PLAY PAGE (separate /play route, linked from nav):          │
│  Grid of all games. Anyone can play. No signup.              │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Claude    │ │ ChatGPT  │ │  Cursor  │ │  Figma   │       │
│  │ 2.4K     │ │ 1.8K     │ │  980     │ │  750     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Notion   │ │  VWO     │ │ Postman  │ │  Linear  │       │
│  │  620     │ │  340     │ │  510     │ │  280     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **Hero has a mini-demo, not a URL input box.** A visitor plays 3 cards inline → gets excited → THEN sees the CTA. The excitement happens BEFORE the ask.

2. **Pricing is visible.** This is where I push back on the waitlist. Showing pricing does two things: (a) it signals "this is a real product, not a side project" and (b) every signup is from someone who SAW the price and still signed up. That's 10x more valuable than a blind waitlist email.

3. **"Get Early Access — 50% off"** replaces the waitlist. It captures emails BUT in a context where the person knows the price. When you email them saying "your account is ready," conversion will be much higher.

4. **Dashboard screenshot is a MOCKUP.** You don't build the dashboard in Phase 1. You design a screenshot in Figma or even just a clean HTML page. This is standard — every pre-launch SaaS does this. It shows the buyer what they'll get.

5. **Logo bar is tools you built games FOR, not paying clients.** Slightly misleading but extremely common in early-stage startups. It builds instant credibility. If someone asks, you say "we've built challenges for these products" — which is true.

6. **Games are on a separate /play page, not the homepage.** The homepage is for BUYERS. The /play page is for PLAYERS. Players come from Reddit/LinkedIn via direct game links or the "Powered by" badge. They can also find the /play page from the nav. But the homepage hero is NOT "play a game" — it's "this is a product for your company."

#### What NOT to build:

| Feature | Why NOT |
|---------|---------|
| Actual AI generation pipeline | The product SAYS "paste URL → AI generates." On the backend, YOU generate manually for now. Classic Wizard of Oz. |
| Dashboard | Show a mockup screenshot. First customers get a Loom video or PDF with their analytics. |
| Auth / user accounts | Nobody logs in yet. Early access signups go to a spreadsheet / Airtable. |
| Card editor | You edit for them. |
| Stripe subscription billing | Early access payments via Stripe payment link. Manual. |
| Embed code | Build when a paying customer asks for it. |
| Learning Mode | Build when a paying customer asks for it. |

**The homepage LOOKS like a fully-built product. Behind the scenes, it's you + a JSON config + manual delivery.** This is how Zapier started (manual integrations), how Groupon started (manual email blasts), how DoorDash started (the founders delivered food themselves). This is the right approach.

#### Phase 1 deliverable:
A website that looks like a legit SaaS product with 7-8 playable games, pricing, early access signup, and a games page.

---

### INTERLUDE: The Done-For-You Backend (What Early Customers Actually Get)

The homepage says "AI generates an interactive challenge from your docs." What actually happens for the first 10-20 customers:

1. Customer signs up for early access (or comes from cold outreach).
2. You get on a 15-min call. Understand their use case.
3. You read their docs. Hand-write 20 challenge cards. 2-4 hours.
4. You deploy it as a new JSON config: `howwellyouknow.com/play/[company]`
5. You send them a Loom walkthrough: "Here's your challenge. Here's the hosted page. Here's how to share it."
6. After 1 week, you send analytics: plays, avg score, which features users struggled with. (Pull from Plausible/your DB. Format as a PDF or Loom.)
7. You charge them $49/mo or $149/mo via Stripe payment link.

**They never know it's manual.** They think AI did it. And once it becomes painful to hand-build games (customer #10-15), you build the AI pipeline. By then you know exactly what "good" looks like because you've hand-built 15 games.

---

### THE DONE-FOR-YOU SERVICE: What You're Actually Selling

#### The Offering

When you email a company and they say "yes, I'm interested," here's what they get for $500:

| Deliverable | Detail |
|-------------|--------|
| **20 challenge cards** about their product | Hand-written from their docs. Covers key features, common misconceptions, advanced tips. |
| **Branded design** | Their logo, brand colors, custom copy. Not a generic template. |
| **Hosted page** | `howwellyouknow.com/play/[company]` — a permanent URL they can share anywhere. |
| **Shareable scorecard** | Players get a downloadable image with their score + company branding to share on social. |
| **Basic analytics report** | After 1 week: total plays, avg score, which questions people get wrong (= which features users don't understand). Delivered as a PDF or Loom video. |
| **1 round of edits** | They can request changes to questions, answers, or branding. |

**Delivery time:** 5 business days.

**Recurring add-on:** $99/mo for continued hosting, monthly analytics reports, and quarterly content updates (new questions as their product evolves).

#### "Why Would I Pay $500? I Can Build This Myself."

This is the #1 objection you'll get. Here's the honest answer — and when it's valid vs. when it's not.

**When the objection is VALID:**
- A developer at a 10-person startup could absolutely use Claude Code to build a quiz in a weekend.
- If they only need a basic quiz with 10 multiple-choice questions for their blog, they don't need you.
- If they have a dedicated DevRel or marketing engineer with free time, they can DIY.

**When the objection is NOT valid (which is most of your ICP):**

The Head of CS at a 200-person SaaS company is not going to:
1. Spend a weekend coding a quiz app
2. Write 20 high-quality questions about their own product (harder than it sounds — they're too close to it)
3. Design it to look good
4. Host it somewhere
5. Build a scorecard sharing feature
6. Set up analytics
7. Maintain it as their product changes
8. Do this for every use case (onboarding, marketing, community)

**The real value isn't the quiz. It's what comes WITH the quiz:**

| What they build themselves | What you deliver |
|---------------------------|-----------------|
| A Google Form with 10 questions | A branded, gamified card-based experience with animations |
| No analytics | "Your users don't understand Feature X" — actionable insight |
| One-time use, forgotten | Ongoing hosting + quarterly updates as product evolves |
| No shareability | Scorecard image optimized for LinkedIn/X sharing (free marketing for them) |
| Looks like a homework assignment | Looks like a product. Polished. Professional. |
| Takes their dev team 2-3 days | Takes them 0 days. You deliver it. |

**The analogy:** Companies can build their own landing pages too. They still pay Webflow. Companies can design their own logos. They still pay designers. The value is: it's done RIGHT, done FAST, and you don't have to think about it.

**The real pitch isn't "I built a quiz for you." It's:**

> "I can tell you which product features your customers don't understand — and give you a tool to fix it. For $500."

The quiz is the mechanism. The insight is the value.

#### Pricing Defense

If someone pushes back on $500:

- **"What does it cost you when a customer churns because they didn't learn your product?"** At $100/mo ARPU, one saved customer = 5 months of your service paid for.
- **"What does your CS team spend on onboarding calls?"** If a CSM costs $70K/year and spends 20% of time on basic onboarding, that's $14K/year. A $500 self-serve onboarding tool that saves even 10% of those calls pays for itself in a month.
- **"Your dev team's time isn't free."** 2 developer days at $80/hr = $1,280. You're cheaper than DIY if you account for opportunity cost.

If they genuinely won't pay $500, offer $200 for a smaller version (10 cards, no branding, basic analytics). If they won't pay $200, they're not your customer. Move on.

---

### HOW EMBEDDING & URL HOSTING TECHNICALLY WORKS

This is the technical explanation of how a company actually puts your game on their website.

#### There Are 3 Ways a Company Uses Your Game:

**Method 1: Shared Link (simplest — Phase 2 default)**

You host the game. They get a URL. They share it wherever they want.

```
URL: https://howwellyouknow.com/play/vwo

Where they use it:
- Paste in onboarding email: "Learn VWO in 5 minutes → [link]"
- Paste in Slack/Discord: "Friday challenge! How well do you know VWO?"
- Paste in LinkedIn post: "Think you know VWO? Prove it → [link]"
- Paste in blog post as a hyperlink
```

The user clicks the link → opens your site in a new tab → plays the game → sees scorecard.

**This is how 90% of early customers will use it.** It's the easiest to deliver and requires ZERO technical integration. You just hand them a URL.

---

**Method 2: Iframe Embed (the "embedding" people talk about)**

The company wants the game to appear INSIDE their website — on their blog, docs site, or onboarding page — without the user leaving their site.

**How it works technically:**

An iframe is an HTML element that loads another webpage inside the current page. It's like a window into your site, displayed on their site.

You give them this code:

```html
<iframe
  src="https://howwellyouknow.com/embed/vwo"
  width="100%"
  height="600"
  frameborder="0"
  style="border-radius: 12px; max-width: 800px;"
></iframe>
```

They paste this into their website's HTML — in a blog post, a docs page, or an onboarding page.

**What happens when a user visits abc.com/onboarding:**

```
┌─────────────────────────────────────────────┐
│  abc.com/onboarding                         │
│                                             │
│  Welcome to ABC! Here are some resources:   │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │                                     │    │
│  │   (This is the iframe)              │    │
│  │   Loads from YOUR server:           │    │
│  │   howwellyouknow.com/embed/vwo      │    │
│  │                                     │    │
│  │   The game renders here.            │    │
│  │   It looks like it's part of        │    │
│  │   abc.com, but it's actually        │    │
│  │   served from your server.          │    │
│  │                                     │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  Continue to your dashboard →               │
└─────────────────────────────────────────────┘
```

**Key points:**
- The game is hosted on YOUR server (howwellyouknow.com). You control it.
- It APPEARS on their site inside the iframe. To the user, it looks integrated.
- You can update the game without them changing anything. Your server serves the latest version.
- Analytics flow to YOUR backend. You track plays, scores, etc.
- The `/embed/vwo` route serves the same game as `/play/vwo` but without your header/footer/branding — just the clean game experience. This makes it look native on their site.

**Who uses iframes?** Almost every embeddable SaaS product:
- Typeform embeds (surveys inside other sites)
- Calendly embeds (booking widget)
- YouTube embeds (video player)
- Intercom embeds (chat widget)
- testimonial.io embeds (review walls)
- Stripe checkout (payment form)

It's the industry standard for "put my product on your site."

**Technical effort for you:** Very low. You already have the game rendering at `/play/vwo`. You just create a second route `/embed/vwo` that renders the same game but stripped of your site's navigation and footer. Maybe 2-3 hours of work.

---

**Method 3: Custom Domain (abc.com/learn → your server)**

This is your question: "If someone has abc.com and wants the game at abc.com/onboarding/learn, how does my product appear at THEIR URL?"

**There are 2 sub-options here:**

**Option A: Subdomain CNAME (easier, more common)**

The company creates a subdomain like `learn.abc.com` and points it to your server.

How:
1. Company adds a CNAME DNS record: `learn.abc.com → howwellyouknow.com`
2. Your server receives requests for `learn.abc.com` and knows to serve ABC's game.
3. The user visits `learn.abc.com` → sees the game. The URL bar shows `learn.abc.com`, not your domain.

```
DNS: learn.abc.com  CNAME → howwellyouknow.com

User visits: learn.abc.com
  → DNS resolves to your server
  → Your server sees "oh, this is abc's custom domain"
  → Serves ABC's game with their branding
  → User sees: learn.abc.com in the URL bar
```

**This is how Notion, Webflow, Carrd, Ghost, and most white-label SaaS products work.**

Technical effort: Moderate. You need to:
- Handle custom domains on your server (Vercel supports this natively with `vercel domains add`)
- Store a mapping: `learn.abc.com → serves VWO game`
- Auto-provision SSL certificates (Vercel/Cloudflare handle this automatically)

**This is a Phase 4 feature. Don't build it now.**

**Option B: Reverse Proxy (harder, for abc.com/path)**

If the company wants the game at `abc.com/onboarding/learn` (a PATH on their main domain, not a subdomain), THEY have to do some work. You can't control their main domain.

How it works:
1. Company configures their web server / CDN / hosting to "reverse proxy" requests for `/onboarding/learn` to your server.
2. When a user visits `abc.com/onboarding/learn`, ABC's server fetches the page from `howwellyouknow.com/embed/abc` behind the scenes and serves it as if it's their own page.

```
User visits: abc.com/onboarding/learn
  → ABC's server receives the request
  → ABC's server proxies to: howwellyouknow.com/embed/abc
  → Your server returns the game HTML
  → ABC's server forwards it to the user
  → User sees: abc.com/onboarding/learn in the URL bar
```

In practice, this is configured in their Nginx/Apache/Vercel/Cloudflare/AWS setup:

```nginx
# Example Nginx config on ABC's server:
location /onboarding/learn {
    proxy_pass https://howwellyouknow.com/embed/abc;
    proxy_set_header Host howwellyouknow.com;
}
```

Or in Vercel (if ABC uses Vercel):
```json
// vercel.json on ABC's project
{
  "rewrites": [
    { "source": "/onboarding/learn", "destination": "https://howwellyouknow.com/embed/abc" }
  ]
}
```

Or in Cloudflare Workers, AWS CloudFront, etc. — every platform has a way to do this.

**The key insight:** For path-based hosting on their domain, the CUSTOMER does the configuration, not you. You just serve the game at `/embed/abc`. They decide how to proxy it.

**This is rare for early-stage.** Most companies are fine with a shared link or iframe. Only large companies with strict brand requirements need path-based hosting on their domain. Don't worry about this until Phase 4.

---

#### Summary: Which Method to Offer When

| Phase | Method | What you offer | Effort for you |
|-------|--------|---------------|---------------|
| Phase 2 | **Shared link** | `howwellyouknow.com/play/vwo` | Zero. Already done. |
| Phase 2 | **Iframe embed** | Paste this code snippet on your site | 2-3 hours one-time to build `/embed/` route |
| Phase 3 | **Subdomain CNAME** | Point `learn.abc.com` to us | Moderate. Vercel handles most of it. |
| Phase 4 | **Reverse proxy (path-based)** | Docs explaining how to configure on their side | You write documentation. Customer configures. |

**For Phase 2, you only need Method 1 (shared link).** That's literally just handing them a URL. Method 2 (iframe) is easy to add when someone asks. Methods 3-4 are for later.

---

### PHASE 2: "Distribute + Outreach" (Weeks 4-8)
**Focus:** Two parallel activities — distribute showcase games for traffic/credibility AND cold outreach for revenue validation. 60% outreach, 40% distribution.

#### DISTRIBUTION (the inbound engine):

Post each showcase game to its relevant community. The goal isn't revenue from players — it's play counts, social proof, and occasional inbound signups from people who click "Powered by" and land on your homepage.

| Week | Game to distribute | Where |
|------|-------------------|-------|
| Week 4 | Claude Code | r/ClaudeAI, r/ChatGPTPro, LinkedIn, X |
| Week 4 | ChatGPT | r/ChatGPT (10M members), LinkedIn, X |
| Week 5 | Cursor | r/cursor, Hacker News, X |
| Week 5 | Figma | r/FigmaDesign, design Twitter, Dribbble |
| Week 6 | Notion | r/Notion, Notion community, LinkedIn |
| Week 6-7 | Postman | r/webdev, dev Twitter, Dev.to |
| Ongoing | Any game | Re-share best performers, cross-post |

**What you're tracking from distribution:**
- Total plays across all games (target: 5,000+ by Week 8)
- Scorecard shares on social media (target: 100+)
- Homepage visits from "Powered by" badge (target: 500+)
- Early access signups from inbound (target: 20-50)

**The play counts and social shares become your outreach collateral.** "5,000 people played our product challenges" is a strong line in a cold email.

#### OUTREACH (the revenue engine):

This is the real validation. Everything above is marketing. This is sales.

**Week 4-5: First 10 Cold Emails**

1. You already have games for VWO and Hotjar/Linear from Phase 1. Use them.
2. Pick 8 more B2B SaaS companies (50-500 employees). Look for companies with:
   - Good public documentation
   - Active user community (Slack, Discord, forum)
   - CS or marketing team (check LinkedIn — do they have a Head of CS or Director of Marketing?)
3. Hand-build a game for each. 15-20 cards. 1-2 hours each (your engine is solid now).
4. Publish on your site: `howwellyouknow.com/play/[company]`
5. Email the founder, Head of CS, or Head of Marketing.

**The outbound email (product-first framing, NOT service framing):**

> Subject: Built a 5-min product challenge for [Company] — check it out
>
> Hi [Name],
>
> We're building HowWellYouKnow — a platform that turns product docs into interactive challenges so your users actually learn your product.
>
> I created one for [Company]: [howwellyouknow.com/play/company]
>
> It covers [specific feature 1], [specific feature 2], and [specific feature 3]. Players score an average of 62% — which means even power users have feature blind spots.
>
> Some ways companies are using this:
> - **Onboarding:** "Learn [Product] in 5 minutes" in welcome emails
> - **Community:** "How well do you know [Product]?" as an engagement post
> - **Marketing:** Embed on blog or docs to drive feature awareness
>
> We're launching our self-serve platform soon. Would you want early access? Happy to customize this one with your branding and give you the analytics dashboard.
>
> Krishna
> HowWellYouKnow.com

**Note the framing difference:** "We're building a platform" and "Would you want early access?" — NOT "I'll build a quiz for you for $500." The product sells the product. The service happens behind the curtain.

**Week 6-7: Iterate Based on Responses**

| Outcome | What it means | What to do |
|---------|--------------|------------|
| 3+ reply "this is cool, let's talk" | Buyer exists. | Get on a call. Offer early access at $49/mo. Hand-build their customized version on the backend. |
| 3+ reply "cool but we can't pay right now" | Interested but no budget or not urgent. | Offer a free 2-week pilot. Deliver manually. Convert later. |
| People reply but wrong persona (marketing wants it, not CS) | Right product, wrong buyer targeted. | Shift outreach to marketing leads. |
| 0 replies from 10 emails | Email not landing or wrong ICP. | Change subject lines, try different company sizes, try founders instead of CS leads. |
| Some reply "we'd use this for employee training" | The L&D market is pulling. | Consider pivoting to internal L&D. Seriously. This happened to 7taps and they're now a $X M company. |

**Week 8: Double Down on What's Working**

If outreach is getting replies:
- Increase to 10-15 emails/week
- Each email includes a pre-built game about THEIR product
- Start building more games for outreach targets (you can do more beyond 8 now)
- Continue distributing showcase games for credibility

If inbound early access signups are coming in:
- Email each one personally: "Thanks for signing up. I'd love to learn about your use case. 15 min call?"
- Get on calls. Understand what they need. Hand-build for them.
- Charge $49/mo from the start.

#### Phase 2 revenue target:
- 3-5 paying early access customers at $49-149/mo = $200-750 MRR
- 5,000+ total plays across all games
- 20-50 early access signups (inbound)
- 3+ positive cold outreach responses (calls booked)

That's not $10K MRR yet. But it's PROOF. You know: the buyer exists, the price works, and what features they actually need.

#### Phase 2 kill criteria:
- **40 outbound emails + 2,000 game plays → 0 positive responses AND 0 early access signups:** The buyer doesn't exist at this ICP. Pivot to a different persona (marketing? DevRel?), a different company size (enterprise?), or a different market (L&D?).
- **People sign up but won't pay $49/mo:** It's a vitamin, not a painkiller. Consider: free tier with "Powered by" growth loop, or pivot entirely.
- **People keep asking for something you're not building** (e.g., "Can you just make a landing page?" or "Can you do video?"): Listen. Build what they're asking for. The market is telling you something.

---

### PHASE 3: "Build the Real Product" (Months 3-6)
**Focus:** Turn the manual backend into a self-serve product. Only do this because Phase 2 PROVED people pay.

**Prerequisite:** You have 5-10 paying customers (or committed early access members). You've hand-built 15-20 games. You know what the buyer wants.

Now you automate what you've been doing manually:

| What you did manually | What you build |
|----------------------|----------------|
| Read their docs and write cards by hand | AI reads docs URL and generates cards (Claude API) |
| Email them a Loom video with analytics | Dashboard with play counts, scores, completion |
| Edit cards based on their feedback | Self-serve card editor |
| Send Stripe payment links manually | Stripe checkout + subscription billing |
| Create each game as a JSON file | Self-serve creation flow: paste URL → generate → edit → publish |

#### What to build in Phase 3:

| Feature | Why |
|---------|-----|
| **AI auto-generation (URL → full game)** | This IS the product. The thing that makes it scalable. |
| **Auth (Clerk or NextAuth)** | Users need accounts. |
| **Dashboard** | List their challenges, see analytics per challenge. |
| **Card editor** | Edit AI-generated content. AI won't be perfect; users need to tweak. |
| **Stripe payments** (Free / $49 / $149) | The pricing that's already on your homepage. Now it's real. |
| **"Powered by" badge on free tier** | The organic growth loop. Free users generate awareness. |
| **Analytics** (plays, avg score, completion, per-question breakdown) | The thing customers asked for most in Phase 2. |
| **Embed code (iframe)** | If Phase 2 customers asked for it. Likely they did. |

#### Only build if Phase 2 customers specifically asked:

| Feature | Build when... |
|---------|------------|
| Learning Mode (lesson → test toggle) | A customer said "I need this for onboarding, not just quizzes" |
| Lead capture (collect player emails) | A customer said "I want to know WHO played, not just how many" |
| CRM export | A customer said "I need this data in HubSpot/Salesforce" |
| Team seats / collaboration | A customer said "My CS team needs access too" |

#### Still don't build (even in Phase 3):
Custom domains, SSO, API access, white-label. These are enterprise features. You don't have enterprise customers.

#### Phase 3 revenue target:
- 10-15 manual customers retained on $49-149/mo = $500-2K MRR
- 20-40 new self-serve customers at $49-149/mo = $2K-6K MRR
- Free tier users generating "Powered by" traffic = new inbound funnel
- **Total: $3-8K MRR by Month 6**

---

### PHASE 4: "Scale to $10K MRR" (Months 6-10)
**Focus:** Growth. Double down on whatever channel is working best.

By now you know:
- Which buyer persona converts (CS? Marketing? Founder? DevRel?)
- Which company size pays (50-200? 200-500? 500+?)
- Which pricing tier is most popular
- Whether inbound or outbound is the stronger channel
- Which features drive retention vs. which are nice-to-have

**Possible paths to $10K MRR:**

**Path A: Self-serve wins.** 100 customers × $100/mo average. Scale through "Powered by" viral loop, content marketing (blog: "How [Company] used interactive challenges to reduce onboarding calls by 40%"), and community distribution (keep publishing free games for popular tools).

**Path B: Outbound wins.** 40 customers × $149/mo + ongoing outreach at 20 emails/week. Hire a VA to help with game content creation. Consider a $499/mo enterprise tier with advanced analytics, custom branding, and dedicated support.

**Path C: Hybrid.** 50 self-serve ($49/mo) + 20 pro ($149/mo) + 5 enterprise ($499/mo) = $2,450 + $2,980 + $2,495 = ~$8K MRR. Close a few more at any tier to hit $10K.

Don't plan for Phase 4 now. The data from Phase 2 and 3 will tell you which path you're on.

---

### THE CHEAT SHEET (v3)

```
GOAL: $10K MRR

BRAND: Product company from Day 1.
BACKEND: Manual service until it hurts. Then automate.

PHASE 1 (Weeks 1-3): BUILD
  - Game engine + 7-8 games + scorecard + homepage
  - Homepage = product pitch, mini-demo, pricing, early access CTA
  - Games page = /play with all games, no signup required
  - Dashboard = mockup screenshot, not built
  - AI generation = doesn't exist yet, you generate manually
  - Revenue = $0. This phase is building the arsenal.

PHASE 2 (Weeks 4-8): DISTRIBUTE + SELL
  - Post showcase games on Reddit/LinkedIn/X → traffic + social proof
  - Cold outreach with pre-built games → revenue validation
  - Inbound early access signups from homepage → warm leads
  - Hand-build for first customers behind the scenes
  - Target: 3-5 paying customers, 5,000+ plays, 20-50 signups
  - KILL if 40 emails + 2K plays → 0 interest from anyone

PHASE 3 (Months 3-6): BUILD THE REAL PRODUCT
  - AI generation, auth, dashboard, editor, Stripe, analytics
  - Only build features Phase 2 customers asked for
  - Target: $3-8K MRR

PHASE 4 (Months 6-10): SCALE
  - Double down on whatever is working
  - Target: $10K MRR

PRICING (visible from Day 1):
  Free        1 challenge, 10 cards, "Powered by" badge
  $49/mo      5 challenges, 25 cards, branding, analytics
  $149/mo     Unlimited, embed, lead capture, priority support
  $499/mo     Enterprise (Phase 4): SSO, custom domain, API

THE KEY INSIGHT:
  The homepage says "AI-powered platform."
  The backend is you + Claude API + JSON configs.
  This isn't lying. It's launching before the product is ready.
  Every successful SaaS company did this.

HYPOTHESIS STACK:
  1. ✅ People play (proven: 380 plays)
  2. ❓ Companies express interest (test: outreach + inbound signups)
  3. ❓ Companies pay (test: price on homepage + sales calls)
  4. ❓ Companies use it (test: deliver + track usage)
  5. ❓ Companies retain (test: Month 2 churn)
```

---

### What I Got Wrong in Previous Versions (And Why)

| What I said before | Why it was wrong | What I actually think now |
|-------------------|-----------------|--------------------------|
| "$500 done-for-you on the homepage" | Looks like a freelance service, not a product | Show SaaS pricing ($49-149/mo). Deliver manually behind the scenes. |
| "No waitlist, show the price directly" | Partially right — show price, but also capture intent | "Get Early Access" with pricing visible = high-quality email capture. |
| "3 games is enough" | Not enough for a credible product | 7-8 games across categories. Enough to fill a games page. More if creation is fast. |
| "Homepage is only for buyers" | Too narrow — players who accidentally land should also convert | Homepage has mini-demo (for anyone) + product pitch (for buyers). Both audiences served. |
| "Free games are just marketing collateral" | Partially true, but underselling their role | Free games are marketing AND the portfolio that makes the homepage credible AND the social proof for outbound emails. |
| "Paste URL → preview is over-engineering" | The URL feature is the product promise, just don't build it yet | Homepage SAYS "paste URL." Backend = you do it manually. Build the real thing in Phase 3. |
| "Sequential phases (build, then sell)" | You can't ignore distribution while building | Build for 3 weeks (with distribution starting Week 4). Outreach starts as soon as games are live. |

---

### This Weekend (Updated)

**Saturday:**
1. Deploy howwellyouknow.com (Next.js project on Vercel)
2. Build game engine (JSON config → rendered game, reuse Claude Code codebase)
3. Move Claude Code game to new domain
4. Start ChatGPT game (content)

**Sunday:**
5. Finish ChatGPT game
6. Start Cursor game
7. Build homepage: hero + mini-demo + logo bar + "How it Works" + pricing + early access CTA
8. Design dashboard mockup screenshot (Figma or clean HTML)

**Week 2 (evenings):**
9. Finish Cursor game
10. Build Figma game + Notion game
11. Add shareable scorecard (image download)
12. Add "Powered by" badge to all games

**Week 3 (evenings):**
13. Build VWO game + Postman game + 1 more outreach target
14. Set up analytics (Plausible)
15. Polish homepage, test everything end-to-end
16. Set up early access email collection (Airtable or simple DB)

**Week 4 (distribution + outreach begin):**
17. Post Claude Code game on r/ClaudeAI + LinkedIn
18. Post ChatGPT game on r/ChatGPT + X
19. Send first 5 cold outreach emails (VWO, Postman, + 3 others with pre-built games)
20. Track: plays, homepage visits, signups, email responses

**Every week after:**
21. Distribute 1-2 games on relevant communities
22. Send 5-10 cold outreach emails with pre-built games
23. Get on calls with anyone who responds (inbound or outbound)
24. Hand-build for anyone who pays
25. Track: emails sent → replies → calls → paying customers

---

### The Decision Framework

When you're stuck on "should I build this?", use this:

```
Is a PAYING customer (or 3+ serious prospects) asking for this?
  ├── YES → Build it next.
  └── NO → Don't build it.
        └── "But I think they'll need it!"
              └── Ask 5 customers if they'd pay for it.
                    ├── 3+ say yes with specifics → Build it.
                    └── <3 say yes → Don't build it. You're guessing.
```

When you're stuck on "should I keep going or pivot?", use this:

```
Week 8 checkpoint:
  Outbound: 40+ emails sent. How many positive responses?
    ├── 5+ → 🎉 Keep going. Scale outreach.
    ├── 1-4 → Promising but not strong. Iterate message/ICP. 4 more weeks.
    └── 0 → 🔴 Pivot or kill.
  
  Inbound: How many early access signups?
    ├── 50+ → Strong demand signal. Prioritize building self-serve.
    ├── 10-50 → Some interest. Focus on converting these to paying.
    └── <10 → Homepage isn't converting. Rethink messaging or ICP.
  
  Paying customers?
    ├── 3+ → Business is real. Keep going.
    ├── 1-2 → Promising. Need to understand why others didn't convert.
    └── 0 → After 8 weeks, 0 revenue? Serious pivot needed.
```

Stop planning. Start building. The plan is good enough.
