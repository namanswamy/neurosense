# 🧠 NeuroSense — Autism Sensory Health Dashboard

A full-stack AI/ML-powered platform for autism spectrum screening based on sensory abnormalities research.

## Features

### Authentication & User Management
- **Signup/Login** with bcrypt-hashed passwords and Express sessions
- **SQLite database** persists all user data, health profiles, questionnaires, and ML results
- Each new user gets their own isolated profile and assessment history

### Onboarding Flow
1. **Personal Details** — Name, DOB, gender, phone
2. **Health Profile** — Height, weight, blood group, conditions, medications, allergies, family history (autism/ADHD/sensory), sleep, exercise, diet

### Comprehensive Questionnaire (11 Sections, ~40 Questions)
- Sensory: Visual, Auditory, Tactile, Olfactory, Vestibular, Proprioceptive, Interoceptive
- Behavioral: Social Communication, Repetitive Behaviors, Emotional Regulation, Executive Function
- Each question scored 0-10 with descriptive anchors

### AI/ML Neural Network Engine
- **4-layer Multi-Layer Perceptron** (15→20→12→8→1)
- 15 input features extracted from questionnaire + health profile
- ReLU activation in hidden layers, Sigmoid output
- He initialization with seeded deterministic weights
- Produces: Autism Score (0-100), Level classification, Pattern type, Risk category
- Feature importance analysis and confidence scoring

### Dashboard
- **Overview** — Metric cards, sensory bars, domain scores, top recommendations
- **AI Results** — Score visualization, level classification, neural network details
- **Sensory Profile** — Per-channel breakdown with descriptions
- **Remedial Plan** — Categorized recommendations (Environment, Therapy, Daily Living, Lifestyle, Medical)
- **Detailed Report** — Complete analysis with NN activation details

## Setup & Running

```bash
cd neurosense
npm install
node server.js
```

Then open **http://localhost:3000** in your browser.

## Tech Stack
- **Backend:** Node.js, Express 5, sql.js (SQLite), bcryptjs, express-session
- **Frontend:** Vanilla HTML/CSS/JS (single-page app, no framework dependencies)
- **Database:** SQLite (file-based, persists as neurosense.db)
- **ML Engine:** Custom neural network in pure JavaScript (server-side)

## Database Schema
- `users` — Auth + profile data
- `health_profiles` — Medical/health information
- `questionnaire_responses` — All questionnaire answers
- `ml_results` — AI analysis outputs + recommendations
- `sensory_entries` — Ongoing tracking entries

## Disclaimer
This is a research/educational tool. Results are NOT clinical diagnoses. Always consult qualified healthcare professionals.
