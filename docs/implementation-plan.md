# Maple Dental voice agent: implementation plan

Companion to [project-definition.md](maple-dental-clinic-agent/docs/project-definition.md), which defines the clinic, agent behavior, tools, metrics and stack.

Each step ends with a "Done when" checklist. A step is done only when every item is met.

---

## Phase 0: Setup

### Step 0.1: Repo and accounts
- Create a GitHub repo `maple-dental-clinic-agent` with folders `agent/`, `backend/`, `config/`, `docs/` and a `README.md`.
- Run `uv init` for a Python 3.11+ project, `uv add openai python-dotenv`, and create a `.env` file (git-ignored) for API keys.
- Get an OpenAI API key.

**Done when:**
- [ ] Repo is on GitHub with the folder structure and a `.gitignore` that covers `.env`
- [ ] A 5-line script sends "hello" to the OpenAI API and prints the reply

---

## Phase 1: Clinic backend

The fake practice management system the agent's tools call.

### Step 1.1: Data model and seed data
- In `backend/`, define SQLite tables: `locations`, `providers`, `patients`, `appointment_types`, `slots`, `appointments`.
- Write `seed.py` that creates the clinic from the project definition: 3 locations, 6 dentists and 5 hygienists, the two appointment types (cleaning with a hygienist, 60 min; checkup with a dentist, 30 min), about 50 patients, and 4 weeks of open slots within each location's hours.
- Include some messy patient data: duplicate names, similar dates of birth, and French names with accents.

**Done when:**
- [ ] Running `uv run python backend/seed.py` creates `clinic.db` from scratch every time
- [ ] You can open the database and see seeded rows in every table
- [ ] Saturday slots exist only at Plateau

### Step 1.2: Read endpoints
- Build a FastAPI app with `GET /locations/{id}`, `GET /slots?location=&type=&from=&to=`, and `POST /patients/verify` (name, date of birth, phone).
- Verification matches names ignoring case and accents, and compares phone numbers by digits only.

**Done when:**
- [ ] All three endpoints return correct data in FastAPI's `/docs` page
- [ ] Verify returns a patient ID for a correct match and a clear "not found" for a wrong one
- [ ] "helene tremblay" matches a patient stored as "Hélène Tremblay", and "(514) 555-0142" matches "514-555-0142"

### Step 1.3: Write endpoints and tests
- Add `POST /appointments` (book), `PATCH /appointments/{id}` (reschedule) and `DELETE /appointments/{id}` (cancel). Booking marks the slot as taken, and cancelling frees it.
- Reject bookings in the past or more than 4 weeks ahead.
- Write pytest tests for each endpoint, including double-booking a slot.

**Done when:**
- [ ] Booking the same slot twice returns an error
- [ ] Booking in the past or beyond 4 weeks returns an error
- [ ] `uv run pytest backend/` passes with at least 6 tests

---

## Phase 2: Text agent

The agent's logic, built and tested by typing in the terminal before adding voice.

### Step 2.1: Chat loop
- In `agent/`, write a terminal chat loop: you type, the model replies, and conversation history is kept.
- Write the first system prompt in `config/maple_dental.yaml`: clinic name, persona, languages, greeting.

**Done when:**
- [ ] You can hold a 5-turn conversation in the terminal
- [ ] The agent greets in French first ("Bonjour, hi") and replies in whichever language you use, switching when you switch

### Step 2.2: First tools
- Define `get_clinic_info` and `check_availability` as OpenAI tools that call the backend over HTTP.
- Put the clinic facts (addresses, parking, accepted insurers) in `config/maple_dental.yaml`.
- Handle the tool-calling loop: the model requests a tool, your code calls it, and the result goes back to the model.
- Tools return errors as data (for example `{"ok": false, "error": "slot already taken"}`), not exceptions.

**Done when:**
- [ ] Asking "Are you open Saturday in Verdun?" triggers `get_clinic_info`, says no, and mentions Plateau
- [ ] Asking for a cleaning next week returns real slots from the database
- [ ] A tool error is explained to you in plain language instead of crashing the loop
- [ ] Every tool call and result is printed in the terminal for debugging

### Step 2.3: Booking with verification
- Add `verify_patient` and `book_appointment`. Enforce verification in code: `book_appointment` refuses if the patient isn't verified in this conversation.
- When nothing is open for the requested date, time or location, the agent offers the closest alternatives.

**Done when:**
- [ ] A full booking works end to end, and the new appointment appears in the database
- [ ] Trying to book without verifying fails, even if the LLM tries
- [ ] Asking for a fully booked day gets alternative dates or another open location

### Step 2.4: Reschedule and cancel
- Add `reschedule_appointment` and `cancel_appointment`, with the same verification check.
- If the caller changes a detail partway through, the agent uses the latest request.

**Done when:**
- [ ] You can reschedule and cancel by conversation, and the database reflects both
- [ ] Changing the date partway through books the new date, not the original
- [ ] After failed verification, no appointment changes

### Step 2.5: Escalation
- Add `transfer_to_human(queue, summary)` with queues `urgent`, `billing` and `front_desk`. In text mode it prints the transfer and summary and ends the conversation.
- Put the transfer rules from the project definition in config: emergency language to `urgent` immediately; billing and claims to `billing`; other clinical questions, failed verification twice, a request for a person, a caller not on file, and no progress after three turns to `front_desk`.

**Done when:**
- [ ] "I have swelling and a lot of pain" transfers to `urgent` immediately, with no booking attempt
- [ ] A caller not on file is transferred to `front_desk`
- [ ] The summary is written in the caller's language
- [ ] All clinic-specific text (prompt, rules, clinic info) lives in `config/`, not in the Python code

---

## Phase 3: Voice

The same agent logic, now with speech. Tested in the browser, no phone yet.

### Step 3.1: LiveKit hello world
- Create a LiveKit Cloud project. Run LiveKit's Python voice agent quickstart unchanged.

**Done when:**
- [ ] You talk to the quickstart agent in LiveKit's browser playground and it answers out loud

### Step 3.2: Plug in the Maple Dental agent
- Replace the quickstart's prompt and tools with yours. Reuse the same tool functions and config from phase 2, not copies.
- Use Deepgram's multilingual model for speech to text and Cartesia for text to speech.

**Done when:**
- [ ] You book an appointment by voice in English, and it shows up in the database
- [ ] You book one in French
- [ ] The phase 2 text loop still works, so both modes share one codebase

### Step 3.3: Voice behavior
- Use LiveKit's built-in turn detection and interruption handling, and tune their settings.
- Test switching from English to French mid-call.

**Done when:**
- [ ] You can interrupt the agent mid-sentence and it stops and listens
- [ ] A mid-call language switch works in 3 out of 3 tries

### Step 3.4: Logging and latency
- Log each call to Langfuse: transcript, tool calls, and timing for each turn (speech end, transcript ready, LLM first token, first audio out).
- Find the slowest stage and reduce it, preferring streaming over blocking calls.

**Done when:**
- [ ] Every call produces a trace in Langfuse you can open and read
- [ ] You can state your p50 and p95 response latency from 10 test calls, against the targets of 1 and 2 seconds

---

## Phase 4: Phone

### Step 4.1: Twilio number to LiveKit
- Buy a Twilio number (a 514 number if available). Set up Elastic SIP Trunking pointed at your LiveKit SIP address, and add a LiveKit inbound trunk and dispatch rule for the agent.

**Done when:**
- [ ] You call the number from your cell phone and the Maple Dental agent answers

### Step 4.2: Human transfer
- Make `transfer_to_human` do a real transfer (SIP REFER or LiveKit's transfer feature) to your own cell phone as the front desk.

**Done when:**
- [ ] Saying "I want to talk to a person" rings your cell phone, and you're connected to the caller

### Step 4.3: Deploy
- Deploy the backend and the agent to Fly.io.

**Done when:**
- [ ] The number works with your laptop turned off
- [ ] A friend calls it and books an appointment without your help
