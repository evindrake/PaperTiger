# PaperTiger

A defined-risk, paper-first automated trading bot in Python.

**This is a learning project with a hard risk ceiling, not a profit
maximizer.** The point is to practice building a *safe autonomous system* --
one with clear structural guarantees, layered software safeguards, and
honest reporting about whether its strategy actually has an edge. It is
not financial advice, and the bundled strategy is not expected to beat the
market.

## Disclaimer

This project was written collaboratively by its human author and Claude
(Anthropic's AI). No warranty of any kind is given, express or implied.

**Use this software entirely at your own risk.** Nothing here is
financial, investment, tax, or legal advice. Trading involves real risk of
loss even in a defined-risk, cash-account setup, and the author(s) are not
responsible or liable for any money gained or lost, directly or
indirectly, through the use, misuse, or malfunction of this code -- paper
or live. If you ever configure this to trade real money, that is a
decision you make yourself, with money you can afford to lose, on an
account dedicated to that purpose alone. Read the code before you trust it
with anything, paper or real.

## Philosophy

Two separate safety layers exist here, and it matters which is which:

1. **STRUCTURAL (load-bearing):** a **CASH account, long-only**. This is
   what makes it *impossible to owe money* -- max loss is your account
   balance, full stop. No code in this repository can override it; it's a
   property of the Alpaca account itself. `preflight.py` checks for it and
   treats a margin-enabled account as a hard failure.
2. **SOFTWARE (capital preservation only):** the kill switch, circuit
   breakers, and pre-trade checks in `safety.py`. These protect capital
   *within* whatever balance you have. They are defense in depth, not the
   reason you can't lose more than you funded -- and they fail closed: on
   any doubt, the system does nothing new.

Note on day trades: the classic "3 day-trades per 5 days under $25k"
Pattern Day Trader rule is a **margin-account** rule and doesn't directly
apply to the cash account this project requires. A cash account has its
own version of the same problem instead -- buying again with unsettled
(pre-T+1) proceeds from a same-day sale is a "good-faith violation," and
repeated violations get the account restricted to settled-cash-only
trading. `engine.py` refuses to submit an order that would complete a
same-day round trip on the same symbol, regardless of loop frequency or
account type -- see `Engine._is_same_day_round_trip`.

A strategy has to clear **two bars** to be taken seriously here:

- Beat buy-and-hold (see `backtest.py`'s benchmark).
- Survive walk-forward, out-of-sample testing (see `walkforward.py`).

Three signals are bundled -- SMA crossover (trend-following), RSI reversion
(mean-reversion), and an experimental scikit-learn classifier (see
`ml_signal.py`) -- and **none of them have been shown to reliably beat
plain buy-and-hold** in backtests against real historical data. They exist
to exercise the plumbing (signal -> order -> risk check -> fill) and to
compare different ideas honestly, not because any of them is expected to
make money. Don't mistake "the bot runs" for "the strategy works."

Because of that, this project also supports a **core-satellite split**
(see `core.py`): permanently set aside a fixed fraction of your seed
capital (`CORE_ALLOCATION_PCT`, default 50%) into an equal-weight
buy-and-hold position across the whitelist, bought once and never sold by
the bot. That guarantees at least part of your capital captures the
market's long-run drift regardless of whether the tactical signal ever
finds a real edge. Set `CORE_ALLOCATION_PCT=0` to disable it entirely.

## Risk profiles & the dynamic tactical universe

Two more knobs, both live-adjustable from the dashboard's **Config** tab
without restarting anything:

- **Risk profile** (Conservative / Normal / Aggressive): a named preset
  over exactly 7 sizing/circuit-breaker fields (trade size, per-position
  cap, concentration cap, cash buffer, daily loss limit, max drawdown,
  and `max_open_positions`), plus optional manual overrides on top. Picked
  from the dashboard, written to `risk_profile.json`, and re-read fresh by
  the engine every tick -- see `safety.RiskProfileStore`. This can **never**
  touch the symbol whitelist, the account type, or the same-day round-trip
  check: those have no configurable backing at all, so there is no lever
  on this page that could reach them, even in principle. A missing or
  corrupted `risk_profile.json` changes nothing -- it fails closed to
  whatever `.env` already says, never to a preset's hardcoded numbers.
- **Dynamic tactical/satellite universe**: `SYMBOLS` in `.env` remains the
  static **core** whitelist (drives `core.py`'s bootstrap sizing and never
  changes automatically). Separately, `candidate_universe.json` is a small,
  static, user-editable pool of well-known liquid US stocks; a weekly job
  (`scripts/refresh_tactical_universe.py`) confirms which candidates are
  currently tradable, ranks the survivors by recent dollar volume, and
  writes the top `TACTICAL_UNIVERSE_SIZE` (default 25) to
  `tactical_universe.json`. The engine reads that file fresh every tick and
  trades it **in addition to** (never instead of) the core symbols.
  `MAX_OPEN_POSITIONS` (also profile-tunable) caps how many *distinct*
  tactical symbols can be open at once, so a wider universe can't turn into
  a pile of tiny buys.

## Setup checklist

New to this? "Paper trading" means Alpaca gives you a fake account funded
with fake money that behaves like a real one -- real market prices, real
order rules, zero real money at risk. Everything below defaults to paper
mode, and going live is a separate, deliberate, multi-step opt-in (step 4)
-- not something that can happen by accident.

1. **Create a free Alpaca paper trading account** at
   [alpaca.markets](https://alpaca.markets). Generate a paper API key/secret
   from the dashboard's Paper Trading tab (no real bank/card info needed
   for this -- paper accounts are free and don't touch real money).
2. Copy `.env.example` to `.env` and fill in `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY` with the values from step 1. Leave `ALPACA_PAPER=true`.
3. Create a virtual environment (an isolated Python install just for this
   project, so it doesn't collide with anything else on your machine) and
   install dependencies:
   ```
   python -m venv .venv
   .venv/Scripts/activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```
4. If you ever intend to go live (real money), later: open a **dedicated
   CASH account** kept entirely separate from any retirement or other
   brokerage account you hold. Never point this bot at an account you
   depend on for anything else. Going live also requires setting
   `ALPACA_PAPER=false` **and** `I_UNDERSTAND_THIS_IS_REAL_MONEY=yes` in
   `.env` -- see `config.py`'s `guard_live()`. This is a hard stop, not a
   suggestion.
5. **Revisit `SYMBOLS` before going live.** `BND` and `GLD` were added to
   the original `SPY,QQQ,VTI,IVV` on 2026-08-24 for paper-mode
   diversification learning (the original four are ~0.85-0.99 correlated
   with each other and didn't diversify anything) -- this list has not been
   deliberated as a live-money allocation and should be re-examined,
   not just carried over, when that day comes. The same goes for
   `candidate_universe.json` if you've enabled the dynamic tactical
   universe (see below) -- it's a hand-picked starter list, not a
   deliberated live-money allocation either.

## Typical session runbook

Run these **in order**. Do not skip ahead to `run.py` just because
`backtest.py` looked good once.

```
# 0. (offline testing only) generate synthetic price data
python generate_synthetic_data.py

# 1. Preflight -- verifies keys work, account is CASH (not margin),
#    symbols are tradable/fractionable, etc. NEVER places an order.
python preflight.py

# 2. Backtest -- offline, against data/*.csv or Alpaca historical bars.
#    Read the LIMITATIONS block at the top of backtest.py.
python backtest.py --source csv

# 3. Walk-forward -- out-of-sample validation. Read the verdict, not just
#    the headline numbers.
python walkforward.py

# 2b/3b. (optional) train and validate the experimental ML signal the same way:
python train_ml_signal.py --source csv
SIGNAL_KIND=ml_classifier python walkforward.py

# 4. Only if 2 and 3 actually hold up (beats buy-and-hold, survives OOS,
#    stable parameters): start the watchdog, the dashboard, and the engine.
#    Three separate processes/terminals:
python watchdog.py
python dashboard.py          # http://127.0.0.1:8787 (Live/Kill Switch/Events/Backtest/Walk-Forward/Status/Config/About tabs)
python run.py                 # the live (paper, by default) trading loop
```

**To stop anything, at any time:** use the Kill Switch buttons at the top of
the dashboard (`http://127.0.0.1:8787`) -- HALT stops new entries, FLATTEN
also liquidates everything back to cash, and CLEAR resumes trading. Under
the hood these just create/remove a file named `HALT` in this directory
(an empty `touch HALT` / `New-Item HALT` also works by hand -- it stops new
entries; put the single word `FLATTEN` inside it to also liquidate
everything back to cash). The engine checks for this file every tick, and
the watchdog will create it automatically if the engine's heartbeat goes
stale.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Frozen config loaded from `.env`. `guard_live()` gate. |
| `signals.py` | The single source of truth for buy/sell/hold logic (SMA crossover, RSI reversion, ML classifier). Both live and backtest import from here. |
| `ml_signal.py` | Feature engineering + model loading for the experimental `ml_classifier` signal. The one deliberate exception to `signals.py`'s "no I/O" rule. |
| `train_ml_signal.py` | Trains the scikit-learn model `ml_signal.py` loads, with a chronological (never shuffled) train/test split. |
| `core.py` | Core-satellite bootstrap: buys and permanently holds a fixed fraction of capital, equal-weight, never sold. |
| `broker.py` | The *only* module that talks to Alpaca. Normalizes SDK objects into plain dataclasses. |
| `strategy.py` | Thin, pure adapter: signal -> dollar-sized `OrderIntent`, aware of the core-satellite carve-out. Cannot place orders itself. |
| `safety.py` | Kill switch, live-reloadable risk profile presets/overrides (`RiskProfileStore`), circuit breakers, pre-trade validation (including an independent core-carve-out guard). |
| `engine.py` | The live loop: fold in risk profile + tactical universe -> kill check -> broker truth -> reconcile -> breakers -> core bootstrap -> propose -> round-trip/position-cap filter -> validate -> submit -> snapshot. |
| `watchdog.py` | Separate stdlib-only process. Its only power: creating the kill file if the engine's heartbeat goes stale. |
| `backtest.py` | Offline simulator. Same caps as live, fills at next bar's open, buy-and-hold benchmark, core-satellite support. |
| `walkforward.py` | Train/test parameter sweep + out-of-sample verdict, signal-agnostic (grid picked from `SIGNAL_KIND`). |
| `sweep_signal_params.py` | Quick single-pass comparison across many parameter combos -- exploration only, NOT a substitute for walk-forward. |
| `dashboard.py` | Stdlib HTTP status page, tabbed (Live/Kill Switch/Events/Backtest/Walk-Forward/Status/Config/About), with beginner-friendly explanations and a kill-switch control (HALT/FLATTEN also send a notification and kick off a background self-test). The Config tab additionally lets you pick a risk profile and set manual per-field overrides (see `safety.RiskProfileStore`) -- still cannot place a trade or touch the symbol whitelist. |
| `preflight.py` | Pre-run checks. Never places an order. |
| `run.py` | Entrypoint: wires config -> broker -> strategy -> engine. |
| `scripts/refresh_tactical_universe.py` | Weekly job: ranks `candidate_universe.json` by liquidity and writes the top symbols to `tactical_universe.json` -- the dynamic satellite pool the engine trades in addition to the static core `SYMBOLS`. |
| `generate_synthetic_data.py` | Writes fake OHLCV CSVs into `data/` for offline testing. |
| `selftest.py` | Runs the full unit test suite programmatically, writes `selftest_results.json` for the dashboard's Status tab. A health check for unattended deployments, not a dev-testing replacement. |
| `notify.py` | Optional, free-by-construction email/SMS notifications (plain SMTP + carrier email-to-SMS gateways) for HALT/FLATTEN/watchdog events. Always writes a local record for the desktop notifier (`scripts/tray_notifier.ps1` on Windows, `scripts/notifier.sh` on Linux/macOS) too. Never raises. |
| `trade_log.py` | Durable, append-only JSONL record (`trade_history.jsonl`) of every order submitted (tactical and core-satellite bootstrap) -- independent of the rolling 200-entry event log and the broker's own history. |
| `equity_history.py` | Self-pruning JSONL log (`equity_history.jsonl`, one snapshot per engine tick, last 35 days kept) of equity/cash/per-symbol position value -- feeds the Live tab's "Last 30 Days" performance chart. |

## Running unattended

All three platforms install the same shape of thing: `watchdog.py`,
`dashboard.py`, and `run.py` as auto-starting, auto-restarting background
services; a daily task that reruns `backtest.py` + `walkforward.py` +
`train_ml_signal.py` against fresh data (the part that keeps searching for
a better configuration -- `run.py` itself only ever executes whatever
signal is currently set in `.env`); a **weekly** task that reruns
`scripts/refresh_tactical_universe.py` to re-rank the dynamic tactical
universe by liquidity (see "Risk profiles & the dynamic tactical universe"
above); a periodic self-test (`selftest.py`, at startup/login and every 4
hours) that catches environment drift in the unattended deployment itself,
separate from `preflight.py` (which checks the account, not the code); and
a desktop notification popup for HALT/FLATTEN/watchdog events, which -- on
every platform -- has to run in your interactive login session rather than
as a background service, since none of them allow a headless service to
draw desktop UI.

This is safe to run unattended in paper mode on any platform:
`config.py`'s `guard_live()` refuses to trade real money unless both
`ALPACA_PAPER=false` and `I_UNDERSTAND_THIS_IS_REAL_MONEY=yes` are
explicitly set in `.env` -- an auto-restarting service can't flip itself
from paper to live, only a human editing `.env` can, and that check runs
again the instant they do.

**Windows** (the most tested path -- this is what the project was
originally built and run on): `scripts/install_services.ps1` uses NSSM
(`winget install NSSM.NSSM`) for the three services and Task Scheduler for
`PaperTiger-DailyResearch` / `PaperTiger-TacticalUniverseRefresh` /
`PaperTiger-SelfTest` / `PaperTiger-TrayNotifier` (the tray-balloon
notifier). Run once from an **elevated (Administrator)** PowerShell
prompt:
```
.\scripts\install_services.ps1
```
Useful commands afterward:
```
Get-Service PaperTiger-*                          # status of all three services
Get-ScheduledTask -TaskName PaperTiger-DailyResearch, PaperTiger-TacticalUniverseRefresh, PaperTiger-SelfTest, PaperTiger-TrayNotifier
nssm stop PaperTiger-Engine                        # stop just the trading loop
.\scripts\uninstall_services.ps1                   # remove everything (elevated)
```
Logs land in `logs\PaperTiger-<Name>.out.log` / `.err.log` (rotated at
5MB; `daily_retrain.ps1` also deletes rotated copies older than 30 days),
and the daily research run logs to `logs\daily_retrain.log`.

**Linux** and **macOS**: `scripts/linux/install_services.sh` (systemd
`--user` services + timers, including `papertiger-universerefresh.timer`)
and `scripts/macos/install_services.sh` (launchd LaunchAgents, including
`com.papertiger.universerefresh`) provide the same setup. Neither needs
root/sudo -- everything installs under your own user account. **These two
were written carefully but developed/tested on Windows, not verified
against a real Linux or Mac machine** -- read the script before running
it, and please open an issue if something doesn't match what's described
there.
```
# Linux
./scripts/linux/install_services.sh
systemctl --user status 'papertiger-*'
./scripts/linux/uninstall_services.sh

# macOS
./scripts/macos/install_services.sh
launchctl list | grep papertiger
./scripts/macos/uninstall_services.sh
```
On Linux, if this machine won't stay logged in as you (e.g. a headless
server), run `loginctl enable-linger "$USER"` once so the services keep
running after logout. On both, logs land in `logs/`, same as Windows. The
desktop notifier uses `notify-send` on Linux (install `libnotify-bin`/
`libnotify` if it's missing) and the built-in `osascript` on macOS.

On every platform: sleep/shutdown still stops everything, same as any
other unattended process -- these survive crashes and reboots into a
running OS, not the OS being off entirely.

## Notifications (optional, free)

`notify.py` fires on engine HALT (consecutive errors, unrecognized order),
FLATTEN (circuit breaker tripped), watchdog-triggered HALT (stale
heartbeat), and any HALT/FLATTEN/CLEAR triggered manually from the
dashboard's Kill Switch tab -- the events you'd otherwise only discover by
checking the dashboard. A dashboard-triggered HALT/FLATTEN also kicks off
the full unit test suite in the background (results land on the Status
tab a few seconds later), and the Status tab has a "Send Test
Notification" button to verify your setup anytime, independent of the
kill switch. Two channels, both free:

- **Email/SMS** via your own email provider's SMTP server -- no paid
  notification service, no API key. "Text message" delivery uses each
  carrier's free email-to-SMS gateway (e.g. `5551234567@vtext.com` for
  Verizon) as just another recipient. Configure `NOTIFY_SMTP_*` /
  `NOTIFY_TO` in `.env` -- see `.env.example` for a Gmail app-password
  walkthrough and the common carrier gateway addresses. Leave
  `NOTIFY_SMTP_HOST` blank to disable this channel entirely.
- **Desktop popup** -- a Windows system-tray balloon via
  `scripts/tray_notifier.ps1` (registered by `install_services.ps1` as the
  `PaperTiger-TrayNotifier` task), or on Linux/macOS a desktop notification
  via `scripts/notifier.sh` (registered by the install scripts in
  `scripts/linux/` / `scripts/macos/`) -- works even with zero email setup,
  since `notify.py` always writes a local `notifications.json` regardless
  of SMTP configuration. The Windows tray icon (`scripts/papertiger.ico`,
  same orange/black-stripe mark as the dashboard header) is generated by
  `scripts/generate_tray_icon.ps1` via .NET's `System.Drawing` -- rerun it
  any time you want to tweak the design; `tray_notifier.ps1` falls back to
  a generic default icon if the file is ever missing.

A durable, independent trade history also gets written to
`trade_history.jsonl` (one JSON object per line) for every order this bot
submits, tactical or core-satellite bootstrap -- unlike the dashboard's
rolling 200-entry event log, this file is never trimmed or reset on
restart.

## Future: remote access from your phone (not yet implemented)

If you later want to check the dashboard from your phone over cellular
data, here's the tradeoff space, cheapest and simplest first. None of this
is implemented -- it's here so the decision is easy to revisit.

1. **Tailscale (recommended).** A free personal-use WireGuard-based mesh
   VPN. Install it on the PC and on your phone (both free apps), sign in
   with the same account on each, and the dashboard becomes reachable at a
   private Tailscale address/hostname (MagicDNS) -- no port forwarding, no
   public exposure at all, no certificate to buy (Tailscale encrypts
   everything itself). This is purpose-built for exactly this "reach my
   home machine from my phone, just me" use case, and is the option to
   default to unless you specifically want a plain public URL.
2. **Cloudflare Tunnel + Cloudflare Access.** Free tier, no port forwarding
   either (an outbound-only `cloudflared` daemon on the PC creates the
   tunnel), and Cloudflare issues/manages the HTTPS certificate for you
   automatically -- no DigiCert purchase needed. Cloudflare Access sits in
   front and requires a login (e.g. via a one-time email code) before
   anyone reaches the dashboard. Needs a domain name you control pointed at
   Cloudflare (~$10-15/year) but gives you a normal browser URL with no VPN
   client required on the phone. A reasonable alternative to Tailscale if
   you want that.
3. **Traditional port-forward + certificate (not recommended).** Buying a
   cert from DigiCert is unnecessary even here -- Let's Encrypt issues
   free, trusted certs -- but this whole approach means opening a port on
   your home router directly to the internet behind a reverse proxy
   (nginx/Caddy), which is both the most setup work and the most exposed
   attack surface of the three. Only worth it if you specifically need
   something neither of the above provides.

Either of the first two keeps the dashboard exactly as it is today (bound
to `127.0.0.1`, cannot place a trade) -- remote access would layer on top,
not change what the dashboard itself can do.

## Running the tests

```
python -m unittest discover -s tests -v
```

CI (`.github/workflows/tests.yml`) runs the same suite on every push/PR
against Python 3.11 and 3.12.

This repo also ships a local pre-commit hook (`.githooks/pre-commit`) that
scans staged files for anything that looks like a real `.env` file or a
hardcoded credential, as a backstop behind `.gitignore`. Enable it once per
clone with:

```
git config core.hooksPath .githooks
```

## Honest limitations (see also the top of `backtest.py`)

- Daily bars only -- no intraday modeling.
- Slippage/commission are flat, configurable assumptions, not a market-impact model.
- No T+1 settlement modeling in the backtest simulator (unlike the live engine,
  which structurally refuses same-day round trips -- see `Engine._is_same_day_round_trip`).
- The core whitelist is small and static -- survivorship bias is real even for
  "boring" ETFs. The optional dynamic tactical universe doesn't fully escape
  this either: it only ever selects from `candidate_universe.json`, a small,
  hand-picked starter list, not a real index membership feed -- it changes
  *which* liquid large-caps get considered week to week, not the underlying
  selection bias of "someone picked this list by hand."
- A backtest or even a walk-forward pass is evidence, not proof, of a forward edge.
- All three bundled signals (SMA crossover, RSI reversion, ML classifier) are
  placeholders/experiments. It is not investment advice, and no part of this
  project should be read as a recommendation to trade any particular security.
- The ML signal's walk-forward test sweeps decision thresholds only -- the
  underlying model is trained once, ahead of time, not retrained per fold.
  See `ml_signal.py` and `backtest.py`'s LIMITATIONS block for more.
- A core-satellite allocation tested through `walkforward.py` gets
  re-established at the start of every fold, not bought once for the whole
  span -- see `backtest.py`'s LIMITATIONS block.

## License

MIT -- see `LICENSE`. This is a learning project shared as-is; the
Disclaimer section above still applies regardless of license terms.
