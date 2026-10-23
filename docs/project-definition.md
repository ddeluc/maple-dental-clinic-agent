# Maple Dental: bilingual voice scheduling agent

An AI phone agent that answers a dental clinic's scheduling line in French or English. It books, reschedules and cancels appointments, answers clinic questions, and transfers to a human when it should.

## 1. Clinic

- **Name:** Maple Dental (Clinique dentaire Maple)
- **Locations:** 3 in Montreal: Plateau, Verdun, Saint-Laurent
- **Providers:** 6 dentists, 5 hygienists
- **Hours:** Mon to Fri 8:00 to 17:00; Sat 9:00 to 13:00 at Plateau only
- **Languages:** French and English
- **Appointment types:** cleaning (hygienist, 60 min) and checkup (dentist, 30 min)
- **Time zone:** America/Montreal for all dates and times
- **Clinic facts** (addresses, parking, accepted insurers) live in the config file and can be fictional.

## 2. Agent behavior

### The agent handles
- Booking, rescheduling and cancelling cleaning and checkup appointments
- Clinic questions: hours, addresses, parking, accepted insurers
- Patient verification: name, date of birth and phone number on file must all match. Match names ignoring case and accents ("Hélène" = "Helene"), and compare phone numbers by digits only.
- Bookings: from today up to 4 weeks ahead, never in the past.

### Booking rules
- If nothing is open for the requested date, time or location, offer the closest alternatives: other dates, or another location that is open (for example, Plateau for Saturday requests).
- If the caller changes a detail partway through (date, time, location), use the latest request, not the original.
- If verification fails, make no changes to any appointment.

### The agent transfers to a human

| Situation | Queue |
|---|---|
| Pain, swelling, bleeding, trauma or any emergency language | `urgent` (immediately, no booking attempt) |
| Billing, insurance claims, treatment plans, clinical questions | `billing` for billing and claims, `front_desk` otherwise |
| Verification fails twice | `front_desk` |
| Caller asks for a person | `front_desk` |
| Caller is not on file (new patient) | `front_desk` (no new-patient registration) |
| No progress after three turns, or anything else unclear | `front_desk` |

### Language
- Greet in French first: "Bonjour, hi".
- Reply in the caller's language. If the caller switches mid-call, switch with them.
- Transfer summaries are written in the caller's language.

## 3. Agent tools

| Tool | Purpose |
|---|---|
| `verify_patient(name, dob, phone)` | Look up the patient. Must succeed before any booking, rescheduling or cancelling. |
| `check_availability(location, type, date_range, provider=None)` | Return open slots |
| `book_appointment(patient_id, slot_id, type)` | Create an appointment |
| `reschedule_appointment(appointment_id, new_slot_id)` | Move an appointment |
| `cancel_appointment(appointment_id, reason)` | Cancel and log the reason |
| `get_clinic_info(location, topic)` | Hours, address, parking, insurers |
| `transfer_to_human(queue, summary)` | Hand off with a short summary. Queues: `urgent`, `billing`, `front_desk`. |

Tools return errors as data the LLM can read, for example `{"ok": false, "error": "slot already taken"}`, instead of raising exceptions. This lets the agent explain the problem to the caller and recover.

## 4. Success metrics

| Metric | Target | Measured by |
|---|---|---|
| Task success | 85% or more of in-scope eval calls end correctly | Evals |
| Correct escalation | 100% of emergency calls transferred to `urgent`; under 10% unnecessary transfers | Evals |
| Language handling | Replies in the caller's language 100% of the time, including after a switch | Evals |
| Latency | p50 response under 1 second, p95 under 2 seconds | Real call logs |

## 5. Tech stack

- **Phone line:** Twilio number, SIP trunk to LiveKit.
- **Voice framework:** LiveKit Agents (Python). Use its built-in audio loop, interruption handling and turn detection; don't build your own.
- **Speech to text:** Deepgram multilingual model, so French, English and mid-call switching work in one stream.
- **LLM:** OpenAI model with tool calling.
- **Text to speech:** Cartesia (Sonic).
- **Clinic backend:** FastAPI with SQLite. A fake practice management system with locations, providers, patients, slots and appointments. Seed it with some messy data (duplicate names, similar dates of birth).
- **Evals:** Python script with an LLM playing simulated callers. Runs in text mode and scores the success metrics above.
- **Observability:** Langfuse. Log each call's transcript, tool calls and per-turn timing.
- **Hosting:** Fly.io. Keep total cost around $20 to $50 a month; don't add paid infrastructure beyond this stack.

**Latency is the top priority across the whole system.** Prefer streaming and avoid blocking calls wherever possible.

## 6. Context
Data is fake. A real deployment would also need to meet Quebec's Law 25 (health information privacy) and the Charter of the French Language, which is why the agent greets in French first.
