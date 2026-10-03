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

Because of that, part of the account is always a plain **buy-and-hold
core** (see `core.py`): a fixed pool (`CORE_POOL_USD`, default $500) bought
once, equal-weight across the core `SYMBOLS`, and never sold by the bot.
That guarantees at least part of your capital captures the market's
long-run drift regardless of whether any signal ever finds a real edge --
and it's the baseline every signal is measured against.

## Comparing strategies side by side (strategy sleeves)

Instead of running one signal and wondering how the others would have
done, the engine can run all three at once in the same (paper) account,
each as its own **sleeve** with its own pool of money and its own symbols
(see `sleeves.py`):

| Sleeve | Trades | Pool |
|---|---|---|
| Buy & hold (core) | the core `SYMBOLS` (SPY, QQQ, VTI, IVV, BND, GLD), bought once, never sold | `CORE_POOL_USD` |
| SMA crossover | ~10 liquid stocks of its own | `SLEEVE_POOL_USD` |
| RSI reversion | ~10 different, similar stocks | `SLEEVE_POOL_USD` |
| ML classifier | ~10 more | `SLEEVE_POOL_USD` |

Which signals run is `STRATEGY_SLEEVES` in `.env` (empty = just
`SIGNAL_KIND`, one sleeve). How it stays safe inside one account:

- **No symbol belongs to two sleeves**, and no signal sleeve ever touches
  a core symbol. Every position at the broker therefore belongs to exactly
  one sleeve, by its ticker -- no splitting shares between strategies, no
  two strategies sending opposite orders for the same stock, and the
  existing per-symbol safety rules (same-day round trip, duplicate order,
  position cap) keep working unchanged. If `sleeves.json` ever lists a
  symbol twice, it's dropped from every sleeve (fail closed).
- **Each sleeve has its own cash.** The account only has one cash balance,
  so each sleeve's cash is rebuilt every tick from the broker's own order
  history: its pool, minus what its filled buys cost, minus what its
  still-open buys could cost, plus what its sells brought in. Every sleeve
  order carries the sleeve's name in its order id
  (`pt-sma-AAPL-buy-2026-10-05`), which is how it's attributed. The
  engine then hands each sleeve a view of the account scoped to that
  money, so the same pre-trade checks as always apply per sleeve -- a
  sleeve can never spend another's pool, however much the account holds.
- **Circuit breakers measure the strategies' money** (all sleeves
  together), not the whole account. A paper account starts with ~$100k,
  so a 3% daily-loss limit on the account would be $3,000 -- more than the
  strategies even have -- and could never trip.
- **Running more than one signal with real money is refused** unless you
  set `ALLOW_MULTI_STRATEGY_LIVE=yes` on top of the usual live-money flags
  (and the dashboard then shows a red warning banner on every tab). It's a
  paper-trading experiment: with real money it would split a small account
  into even smaller pools.

**Starting a comparison** (stop the engine first):

```
python scripts/refresh_tactical_universe.py      # re-rank the candidate stocks by liquidity
python scripts/start_sleeve_experiment.py        # dry run: prints what it would do, changes nothing
python scripts/start_sleeve_experiment.py --execute   # market must be open
```

With `--execute` it (1) sells every position that isn't core, so all three
signals start from cash on the same day; (2) tops each core symbol up to
its share of `CORE_POOL_USD`; (3) deals the ranked `tactical_universe.json`
out to the signal sleeves sector by sector (using the `sectors` map in
`candidate_universe.json`), so each gets a similar mix -- some big tech, a
bank, a health-care or consumer name -- and no sleeve always gets the most
liquid name; and (4) records every symbol's starting price and writes
`sleeves.json`. Each sleeve's symbols then **stay fixed** for the whole
experiment; the weekly universe refresh doesn't change them (re-run with
`--restart` to start over). It refuses to run against a live account,
while the engine is still running, or with orders still open.

**Reading the results** -- the dashboard's **Compare** tab shows each
sleeve's value, return, max drawdown, trades, round trips and win rate,
plus one chart of every sleeve's % return since the start. The column
that matters most is **vs own buy & hold**: how the signal did compared
with simply buying its own symbols on day one and holding them. The
sleeves trade different stocks, so their raw returns mostly show which
stocks happened to rise; each sleeve against its own buy-and-hold takes
that luck out. Until every signal has around 30 closed round trips, the
tab says "too early to tell" -- and even a few months is one market mood,
so treat the result as evidence, not proof.

The ML sleeve needs a model trained on its own symbols:
`python train_ml_signal.py --source alpaca --start <date> --end <date> --sleeve ml`
(the nightly retrain does this automatically once `sleeves.json` exists,
and the engine picks up a retrained model without a restart). Until the
model file exists the ML sleeve just sits out, with one warning a day --
the other sleeves keep trading. `backtest.py` and `walkforward.py` accept
`--sleeve <sma|rsi|ml>` too, to test one sleeve's signal on its own
symbols.

## Risk profiles

Live-adjustable from the dashboard's **Config** tab without restarting
anything: a named preset (Conservative / Normal / Aggressive) over exactly
7 sizing/circuit-breaker fields, plus optional manual overrides on top.
Picked from the dashboard, written to `risk_profile.json`, and re-read
fresh by the engine every tick -- see `safety.RiskProfileStore`.

Sizes are a **share of each strategy's own pool**, so a profile means the
same thing whatever the pool is (shown here for a $500 pool):

| | Conservative | Normal | Aggressive |
|---|---|---|---|
| Trade size | 10% ($50) | 13% ($65) | 14% ($70) |
| Max open positions (per strategy) | 5 | 6 | 7 |
| **Most of the pool invested at once** | **~50%** | **~78%** | **~98%** |
| Per-position cap | 15% ($75) | 18% ($90) | 20% ($100) |
| Concentration cap (of what the strategy is worth now) | 25% | 35% | 50% |
| Cash buffer | 5% ($25) | 3% ($15) | 2% ($10) |
| Daily loss limit (all strategies together) | 2% | 3% | 5% |
| Max drawdown limit (all strategies together) | 10% | 15% | 25% |

The main difference is how much of each pool can be invested at once.
That matters for the strategy comparison: each signal is measured against
buy-and-hold of its own stocks, which is always fully invested, so a
strategy that can only invest half its pool trails that in a rising
market just from holding cash. **Aggressive is the fairest setting for the
comparison.** The same values apply to every signal sleeve, so sizing stays
the same across strategies and only the signal differs. This can **never** touch which symbols a sleeve trades, the
account type, or the same-day round-trip check: those have no configurable
backing on that page at all, so there is no lever there that could reach
them, even in principle. A missing or corrupted `risk_profile.json`
changes nothing -- it fails closed to whatever `.env` already says, never
to a preset's hardcoded numbers.

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
   `candidate_universe.json` (the stocks the signal sleeves are dealt
   from, see above) -- it's a hand-picked starter list, not a deliberated
   live-money allocation either.

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

# 3c. Give each strategy sleeve its symbols and a clean start (see
#     "Comparing strategies side by side" above) -- without sleeves.json the
#     signal sleeves have no symbols and only the buy-and-hold core trades:
python scripts/refresh_tactical_universe.py
python scripts/start_sleeve_experiment.py            # dry run first
python scripts/start_sleeve_experiment.py --execute  # market hours

# 4. Only if 2 and 3 actually hold up (beats buy-and-hold, survives OOS,
#    stable parameters): start the watchdog, the dashboard, and the engine.
#    Three separate processes/terminals:
python watchdog.py
python dashboard.py          # http://127.0.0.1:8787 (Live/Compare/Kill Switch/Events/Backtest/Walk-Forward/Status/Config/About tabs)
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

A HALT is otherwise always manual -- with one narrow, self-verifying
exception. If the engine HALTs because `MAX_CONSECUTIVE_ERRORS` (default 3)
ticks in a row all failed with the broker/API itself erroring out -- Alpaca
returning a 5xx, a timeout, or a local network/DNS blip (e.g. Wi-Fi or VPN
dropping briefly) -- never a logic bug, an unrecognized order, or a
circuit-breaker loss/drawdown trip, all of which still require you -- it
tags the HALT file accordingly and probes the broker once per tick
from then on. The first failed probe logs a single line (not one every
loop); the moment a probe actually succeeds, it clears the HALT itself,
logs once, sends a notification, and resumes trading in that same tick --
no waiting for the next scheduled loop, no manual `rm HALT`/`Remove-Item
HALT` needed. Anything else -- a bug, a reconcile mismatch, a circuit
breaker, or a HALT you triggered yourself from the dashboard or by hand --
stays exactly as manual as ever.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Frozen config loaded from `.env`. `guard_live()` gate. |
| `signals.py` | The single source of truth for buy/sell/hold logic (SMA crossover, RSI reversion, ML classifier). Both live and backtest import from here. |
| `ml_signal.py` | Feature engineering + model loading for the experimental `ml_classifier` signal. The one deliberate exception to `signals.py`'s "no I/O" rule. |
| `train_ml_signal.py` | Trains the scikit-learn model `ml_signal.py` loads, with a chronological (never shuffled) train/test split. |
| `core.py` | The buy-and-hold core sleeve: buys `CORE_POOL_USD` once, equal-weight across the core `SYMBOLS`, never sold. |
| `sleeves.py` | Strategy sleeves: which symbols and pool each strategy owns (`sleeves.json`, fail-closed loader that drops any symbol listed twice), each sleeve's cash/value rebuilt from the broker's order history, its own-symbols buy-and-hold benchmark, and the sector-balanced symbol dealing. No broker access. |
| `sleeve_history.py` | One line per day (`sleeve_history.jsonl`, never pruned) of every sleeve's value -- feeds the Compare tab's chart. |
| `broker.py` | The *only* module that talks to Alpaca. Normalizes SDK objects into plain dataclasses; every call site raises `BrokerError` uniformly, whether the failure was Alpaca's API itself or the underlying network (DNS, timeout, connection refused). |
| `strategy.py` | Thin, pure adapter: signal -> dollar-sized `OrderIntent`, tagged with its sleeve in the order id. Cannot place orders itself. |
| `safety.py` | Kill switch (manual by design, with one self-verifying auto-clear exception for broker-connectivity HALTs -- see above), live-reloadable risk profile presets/overrides (`RiskProfileStore`), circuit breakers, pre-trade validation (including an independent core-carve-out guard). |
| `engine.py` | The live loop: fold in risk profile + sleeves -> kill check (auto-probes/resumes a broker-connectivity HALT, otherwise obeys it) -> broker truth -> reconcile -> rebuild each sleeve's cash -> breakers (on the strategies' money) -> core bootstrap -> for each signal sleeve, on its own symbols and cash: propose -> round-trip/position-cap filter -> validate -> submit -> snapshot. |
| `watchdog.py` | Separate stdlib-only process. Its only power: creating the kill file if the engine's heartbeat goes stale. |
| `backtest.py` | Offline simulator. Same caps as live, fills at next bar's open, buy-and-hold benchmark, core-satellite support. `--sleeve <id>` tests one sleeve's signal on its own symbols. |
| `walkforward.py` | Train/test parameter sweep + out-of-sample verdict, signal-agnostic (grid picked from `SIGNAL_KIND`, or the sleeve's signal with `--sleeve <id>`). |
| `sweep_signal_params.py` | Quick single-pass comparison across many parameter combos -- exploration only, NOT a substitute for walk-forward. |
| `dashboard.py` | Stdlib HTTP status page, tabbed (Live/Compare/Kill Switch/Events/Backtest/Walk-Forward/Status/Config/About), with beginner-friendly explanations and a kill-switch control (HALT/FLATTEN also send a notification and kick off a background self-test). The Compare tab shows the strategy sleeves side by side. The Config tab additionally lets you pick a risk profile and set manual per-field overrides (see `safety.RiskProfileStore`) -- still cannot place a trade or change which symbols any strategy trades. Optionally reachable from your phone over Tailscale (`--tailscale`, off by default) -- see "Remote access from your phone" below. |
| `preflight.py` | Pre-run checks. Never places an order. |
| `run.py` | Entrypoint: wires config -> broker -> strategy -> engine. |
| `scripts/refresh_tactical_universe.py` | Weekly job: ranks `candidate_universe.json` by liquidity and writes the top symbols to `tactical_universe.json` -- the pool `start_sleeve_experiment.py` deals out to the signal sleeves. Doesn't change a running experiment's symbols. |
| `scripts/start_sleeve_experiment.py` | One-time start of a strategy comparison: sells everything that isn't core, tops core up to its pool, deals symbols to the sleeves by sector, records start prices, writes `sleeves.json`. Dry run by default; paper only. |
| `generate_synthetic_data.py` | Writes fake OHLCV CSVs into `data/` for offline testing. |
| `selftest.py` | Runs the full unit test suite programmatically, writes `selftest_results.json` for the dashboard's Status tab. A health check for unattended deployments, not a dev-testing replacement. |
| `notify.py` | Optional, free-by-construction email/SMS notifications (plain SMTP + carrier email-to-SMS gateways) for HALT/FLATTEN/watchdog events. Always writes a local record for the desktop notifier (`scripts/tray_notifier.ps1` on Windows, `scripts/notifier.sh` on Linux/macOS) too. Never raises. |
| `trade_log.py` | Durable, append-only JSONL record (`trade_history.jsonl`) of every order submitted (each sleeve's, and the core bootstrap) -- independent of the rolling 200-entry event log and the broker's own history. |
| `equity_history.py` | Self-pruning JSONL log (`equity_history.jsonl`, one snapshot per engine tick, last 35 days kept) of equity/cash/per-symbol position value -- feeds the Live tab's "Last 30 Days" performance chart. |

## Running unattended

All three platforms install the same shape of thing: `watchdog.py`,
`dashboard.py`, and `run.py` as auto-starting, auto-restarting background
services; a daily task that reruns `backtest.py` + `walkforward.py` +
`train_ml_signal.py` against fresh data (the part that keeps searching for
a better configuration -- `run.py` itself only ever executes whatever
signals are currently set in `.env`; once a strategy comparison is running,
the ML model is retrained on the ML sleeve's own symbols); a **weekly**
task that reruns `scripts/refresh_tactical_universe.py` to re-rank the
candidate stocks by liquidity (see "Comparing strategies side by side"
above -- it doesn't change a running comparison); a periodic self-test (`selftest.py`, at startup/login and every 4
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
heartbeat), auto-resume from a broker-connectivity HALT, and any
HALT/FLATTEN/CLEAR triggered manually from the dashboard's Kill Switch tab
-- the events you'd otherwise only discover by
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
submits, each sleeve's or the core bootstrap -- unlike the dashboard's
rolling 200-entry event log, this file is never trimmed or reset on
restart.

## Remote access from your phone (Tailscale)

**Implemented, opt-in, off by default.** `dashboard.py --tailscale` (or the
`ENABLE_TAILSCALE`-equivalent -- see `run()`'s `--tailscale` flag) binds a
*second* listener on this machine's current Tailscale IPv4 address,
auto-detected fresh at every startup via `tailscale ip -4` (so it survives
Tailscale ever reassigning the address). The primary `127.0.0.1` listener
is unaffected -- this is additive, not a replacement.

Setup (once):
1. Install Tailscale on the PC (`winget install Tailscale.Tailscale`) and
   sign in (`tailscale up` prints a login URL). Install the Tailscale app
   on your phone and sign in with the same account.
2. Pass `--tailscale` when starting `dashboard.py` (already wired into
   `scripts/install_services.ps1`'s `PaperTiger-Dashboard` registration via
   `Install-PtService`'s `-ExtraArgs`).
3. From your phone (on the tailnet, cellular data is fine -- no VPN client
   config needed beyond the Tailscale app itself), browse to
   `http://<this-PC's-tailscale-IP>:8787` (find it with `tailscale ip -4`
   on the PC, or `tailscale status`).

This is deliberately NOT a bind to `0.0.0.0`: the socket only ever listens
on `127.0.0.1` and this machine's specific Tailscale IP, so it's reachable
from other devices on your own tailnet and nowhere else -- not the rest of
your home LAN, not the public internet. If Tailscale isn't installed or
isn't logged in, `--tailscale` silently degrades to local-only rather than
failing to start (see `dashboard._detect_tailscale_ip()`).

**MagicDNS (recommended, free, on by default for most tailnets):** gives
each device a stable hostname (`<device-name>.<tailnet-id>.ts.net`) instead
of a bare IP, so bookmark that instead of the numeric address -- it
survives Tailscale ever reassigning the IP, with zero code changes here.
Check `tailscale status` / the admin console if you're not sure it's on.

**HTTPS: deliberately not enabled, and not recommended for this setup.**
Tailscale can issue a real Let's Encrypt cert for a device's `.ts.net`
name (`tailscale cert`, or `tailscale serve` to front the existing
plain-HTTP listener with zero code changes here) -- but enabling
certificate issuance for a tailnet publishes that device name in
Certificate Transparency logs: a public, append-only ledger that cannot be
edited or removed once an entry lands, by design, across every CT log
operator. That permanently, publicly links your tailnet's ID to whatever
device name you issue a cert for. The security upside is marginal here --
Tailscale's WireGuard transport already encrypts everything end-to-end
between your devices, so plain HTTP over the tailnet isn't the same
exposure as plain HTTP over the open internet; HTTPS on top would mainly
buy a browser padlock and defense against something else on the same PC
sniffing loopback traffic before it hits the tunnel. A permanent public
ledger entry isn't a good trade for that, so this project leaves HTTPS off.

Two alternatives, if you'd rather not use Tailscale (not implemented here,
listed for reference):

- **Cloudflare Tunnel + Cloudflare Access.** Free tier, no port forwarding
  (an outbound-only `cloudflared` daemon creates the tunnel), Cloudflare
  issues/manages the HTTPS certificate automatically. Needs a domain name
  you control pointed at Cloudflare (~$10-15/year), but gives a normal
  browser URL with no VPN client required on the phone.
- **Traditional port-forward + certificate (not recommended).** Let's
  Encrypt issues free certs, but this means opening a port on your home
  router directly to the internet behind a reverse proxy (nginx/Caddy) --
  both the most setup work and the most exposed attack surface of the
  three. Only worth it if you specifically need something neither of the
  above provides.

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
- The core list is small and static -- survivorship bias is real even for
  "boring" ETFs. The signal sleeves' stocks don't fully escape this either:
  they only ever come from `candidate_universe.json`, a small, hand-picked
  starter list, not a real index membership feed.
- The strategy comparison runs each signal on DIFFERENT stocks, so luck in
  which stocks a sleeve was dealt is mixed into its result. Comparing each
  sleeve with buy-and-hold of its own stocks (the Compare tab's key column)
  removes most of that, not all of it. And a couple of months of paper
  trading is a few dozen trades per sleeve -- one market mood, not a verdict.
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
