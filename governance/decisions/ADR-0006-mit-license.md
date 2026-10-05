# ADR-0006: MIT license and original code only

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

The mission is that anyone can run frontier-scale MoE models on hardware they
already own. That favours the most permissive, best understood license, so
that Hearth can be embedded, packaged and redistributed without friction.
Prior art exists in the same space (for example Colibri, Apache-2.0). Copying
from such projects would bring license obligations and attribution duties into
our code, and would blur what this project actually designed.

## Decision

* All code, documentation and tooling are released under the MIT License
  (LICENSE). Contributions come in under the same terms (inbound = outbound).
* No code is copied from other projects, whatever their license. Ideas and
  published techniques may be used and should be cited in docs or ADRs. The
  implementation is written from scratch.
* The engine has no third-party code at all (INV-DEP), so there is nothing to
  relicense or bundle.
* Model weights are never distributed by the project (INV-DATA). Each model
  keeps its own license, which users accept on their own.

## Consequences

* Downstream users can use Hearth almost without restriction.
* Unlike Apache-2.0, MIT contains no explicit patent grant. If that becomes a
  concern for contributors or adopters, a new ADR must supersede this one.
* Reviewers must reject code that looks lifted from elsewhere. Contributors,
  human or AI, are responsible for the originality of what they submit.
