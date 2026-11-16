# Rebuild YouTube with AI
<img width="389" height="337" alt="image" src="https://github.com/user-attachments/assets/2fc30936-96bc-4750-929e-2eb2d56bc477" />

## What you'll learn

### Build a working YouTube clone
* Ship a functional video site in four weeks
* Go from empty GitHub repo to deployed product
* Direct the build while AI writes the code

### Understand how YouTube actually works
* Map core surfaces from home feed to watch page
* Scope an MVP the way a PM and tech lead would
* Learn how real YouTube is built, and how to approximate it in React

### Master the AI coding workflow
* Break a complex app into small verifiable tasks
* Plan, execute, and review with Cursor agents
* Recover from bad diffs, dead ends, and derailed sessions

### Turn AI mockups into pages
* Generate UI mockups with AI image models
* Move designs from concept into working code
* Rebuild YouTube's core pages

### Develop a full backend
* Model data in Postgres with a CRUD admin
* Add sign-in with Better Auth and a working upload flow
* Seed realistic fake users and videos with generative AI

### Add search and related videos
* Generate multimodal embeddings for all your content
* Implement search and related videos
* Learn what real recommender systems do differently

### Ship to production on Vercel
* Deploy your project to Vercel
* Track watch time in an admin dashboard
* Test features with AI-driven Playwright checks

## YouTube Architecture and AI Coding Foundations
### YouTube architecture
* Core surfaces (home, watch, channel, search, shorts, subscriptions)
* Watch page deep dive like player, metadata, and related sidebar
* Project decomposition
### The AI-assisted foundations workflow
* Cursor fundamentals (Ask / Plan / Execute)
* Task decomposition for agents (small, well-specified, verifiable)
* Planning inside Cursor vs external LLMs like Claude or Codex
* Ensemble opinions and verification
* Failure recovery (rollback vs fix forward, every win into git)
* Common dead ends
### Framework choice
* A framework for choosing the framework (4 questions)
* Why React + Next.js
* InnerTube and Polymer (why real YouTube isn't React)
* Project setup (GitHub repo, Cursor, Next.js scaffold)
* Mock pipeline and design
* Implementation loop (run → diff → iterate) for home, watch, channel
### Assignments
* Add dark mode (saved in a cookie)
* Build a scrollable Shorts feed
* YouTube Architecture and AI Coding Foundations diagram
<img width="502" height="546" alt="image" src="https://github.com/user-attachments/assets/89281f72-8194-486a-ae9d-207c0e9b46d9" />

## Database, Fake Data, and Authentication
### From static to dynamic
* Data definitions and schema design
* Why Postgres
* ORM landscape (Prisma, Drizzle, Kysely, TypeORM)
* Supabase setup (static pages → dynamic from DB)
* CRUD and admin
### Data and GenAI
* Content pipeline (LLM plan → image → video → storage + DB)
* Model access like OpenRouter
* Cost ladder
* Consistent keyframes for base image and editing
* Generation scripts
### Auth, upload, and real storage
* Authentication (email + password)
* Extending auth (Gmail, Apple, magic links)
* Upload page
* Cloudflare serving (image CDN, Stream for video)
### Assignments
* Add a mobile web (mweb) mode
* Database, Fake Data, and Authentication diagram
<img width="442" height="197" alt="image" src="https://github.com/user-attachments/assets/3fed47ee-0383-4167-97eb-192c67f0cd9f" />
<img width="689" height="670" alt="image" src="https://github.com/user-attachments/assets/330aeda9-fa89-43b6-ad81-f039169ce0a4" />

## Search, Related Videos, and Embeddings
### How discovery works
* Search modalities (text ↔ video, video ↔ video, user ↔ video)
* Embedding types (text, image, video, multimodal)
* Shared embedding space (reusable for search)
* Model options (CLIP, Qwen3-VL-Embedding vs Gemini, Cohere)
* Skipping video embeddings
### Building on pgvector
* Why Postgres for vectors
* pgvector setup
* Query-time embeddings
* Search implementation
* Related videos (nearest neighbors, queue, mix order)
* Limitations: accuracy, domain mismatch, no user data
* Real recommender systems
### Assignments
* Create embeddings and an “endless” feed of Shorts picking a random out of N best recommendations
* Implement liking videos and use them as a signal for recommendations, i.e. averaging/clustering a bunch of likes and picking the closest video
* Search, Related Videos, and Embeddings diagram
<img width="469" height="746" alt="image" src="https://github.com/user-attachments/assets/98b248a3-d244-4c91-9402-edd0b67bf550" />

## Deploy, Polish, and Real-World Workflows
### Shipping to production
* Vercel deployment (GitHub connect, env vars, push to deploy)
* Why Vercel
* Alternatives
* Performance
### Team workflows
* Linear + MCP
* Beyond Linear (Slack tasks, cloud agents, Devin)
* Watch-time tracking
* Admin dashboards
* Subscriptions feature
* AI-driven testing (Playwright + Midscene)
### Advanced YouTube topics
* What's missing
* InnerTube deep cut
* Video processing graph
* Better generated videos
### Assignments
* Add channel creation flow (in YT user and channel are two different things) and implement personal and channels pages
* Add video upload system which sends videos to Cloudflare (compliments AI uploaded videos)
* Deploy, Polish, and Real-World Workflows diagram
<img width="528" height="732" alt="image" src="https://github.com/user-attachments/assets/9877ed74-3ec8-4777-a0c2-4985417afd7b" />

## Build Your Capstone
### Extend MewTube
* YouTube Shorts (new aspect ratio, generate + serve)
* Generated comments
* Better videos (complex scenarios, multiple beats)
### Go beyond YouTube
* Instagram / TikTok web clone
* Content farm
* Your own product (same approach, new domain)
* Build Your Capstone diagram
<img width="418" height="429" alt="image" src="https://github.com/user-attachments/assets/94dab12a-1e40-4782-80c5-719002f72f18" />

* https://live.bytebytego.com/courses/youtube-ai?promotionCode=INSIDER20
