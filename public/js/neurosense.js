// ═══════════════════════════════════════════
// neurosense.js — Shared Data Bridge
// Include this in every page with:
// <script src="../js/neurosense.js"></script>
// ═══════════════════════════════════════════

const NeuroSense = {

  // ── AUTH ──
  getUser() {
    try { return JSON.parse(localStorage.getItem('ns_user')); } catch { return null; }
  },
  setUser(u) {
    localStorage.setItem('ns_user', JSON.stringify(u));
  },

  // ── ML RESULTS ──
  getResults() {
    try { return JSON.parse(localStorage.getItem('ns_ml_results')); } catch { return null; }
  },
  setResults(ml) {
    localStorage.setItem('ns_ml_results', JSON.stringify(ml));
    // Also append to history for trajectory charts
    const history = this.getHistory();
    history.push({ date: new Date().toISOString(), ...ml });
    localStorage.setItem('ns_ml_history', JSON.stringify(history));
  },

  // ── ASSESSMENT HISTORY (for analytics trajectory) ──
  getHistory() {
    try { return JSON.parse(localStorage.getItem('ns_ml_history')) || []; } catch { return []; }
  },

  // ── DAILY LOGS ──
  getLogs(type = null) {
    try {
      const all = JSON.parse(localStorage.getItem('ns_daily_logs')) || [];
      return type ? all.filter(l => l.type === type) : all;
    } catch { return []; }
  },
  saveLog(entry) {
    const logs = this.getLogs();
    logs.unshift({ ...entry, id: Date.now(), date: new Date().toISOString() });
    localStorage.setItem('ns_daily_logs', JSON.stringify(logs));
  },

  // ── STREAK ──
  getStreak() {
    return parseInt(localStorage.getItem('ns_streak') || '0');
  },
  updateStreak() {
    const last = localStorage.getItem('ns_last_checkin');
    const today = new Date().toDateString();
    if (last === today) return this.getStreak(); // already checked in today
    const yesterday = new Date(Date.now() - 86400000).toDateString();
    const streak = last === yesterday ? this.getStreak() + 1 : 1;
    localStorage.setItem('ns_streak', streak);
    localStorage.setItem('ns_last_checkin', today);
    return streak;
  },

  // ── TASKS (action plan) ──
  getTasks() {
    try { return JSON.parse(localStorage.getItem('ns_tasks')) || {}; } catch { return {}; }
  },
  setTask(id, done) {
    const tasks = this.getTasks();
    tasks[id] = { done, date: new Date().toISOString() };
    localStorage.setItem('ns_tasks', JSON.stringify(tasks));
  },

  // ── POINTS ──
  getPoints() {
    return parseInt(localStorage.getItem('ns_points') || '0');
  },
  addPoints(pts) {
    const total = this.getPoints() + pts;
    localStorage.setItem('ns_points', total);
    return total;
  },

  // ── GUARD: call on every page load ──
  requireAuth(redirectTo = '../index.html') {
    if (!this.getUser()) {
      window.location.href = redirectTo;
      return false;
    }
    return true;
  },

  // ── SHARED SIDEBAR: inject dynamic user info ──
  initSidebar() {
    const user = this.getUser();
    const streak = this.getStreak();
    const nameEl = document.getElementById('sb-user-name');
    const roleEl = document.getElementById('sb-user-role');
    const avEl   = document.getElementById('sb-user-av');
    const streakEl = document.getElementById('sb-streak-count');
    if (nameEl && user) nameEl.textContent = user.name || 'User';
    if (roleEl && user) roleEl.textContent = user.role || 'Parent';
    if (avEl   && user) avEl.textContent = (user.name || 'U').slice(0,2).toUpperCase();
    if (streakEl) streakEl.textContent = streak + '-day streak — keep it up!';
    // Mark active nav item
    const page = window.location.pathname.split('/').pop();
    document.querySelectorAll('.nav-item[data-page]').forEach(el => {
      el.classList.toggle('active', el.dataset.page === page);
    });
  }
};

window.NeuroSense = NeuroSense;