# Hardening playbook (fair, deterministic)

Goal
- Reduce trivial "thread the parameter" solutions without adding undocumented requirements.

Requirements
- Keep tests deterministic and offline.
- Avoid timing checks and exact error strings unless required.
- Use public API entrypoints.

Fair anti-shortcut traps (examples)
- Coercion trap: pass an object that raises if int()/__index__ is called. Non-applicable code paths must ignore it.
- Multi-entrypoint trap: require identical behavior across instance methods and classmethods where both are public.
- Alias trap: enforce that "week" and "weeks" behave identically across all touched entrypoints.
- Timezone trap: enforce tz string and tzinfo object equivalence, including fixed-offset timezones.

Invariants (examples)
- Contiguity under different bounds.
- exact=True truncation rules.
- Non-week frames ignore week_start and never validate it.
- week_start validation triggers only for week/week(s).

Keep it fair
- Do not assert internal helpers, private attributes, or line numbers.
- Do not require a specific implementation strategy.
