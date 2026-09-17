# W&B MCP Server vX.Y.Z

**Release date:** Pending

## Availability

| Channel | State | Immutable artifact |
|---|---|---|
| Public source | Pending | Source revision; signed tag and GitHub Release when available |
| Container | Pending | Public repository and verified immutable digest |
| Dedicated/Self-Managed | Pending | Compatible chart version and installation notes |

Report each artifact's actual status. Do not infer a signed source release or
installation qualification from a published container.

## Summary

Describe the customer-visible outcome in two or three sentences.

## Highlights

- Describe the most important behavior change.
- State performance, correctness, security, and privacy outcomes precisely.
- Separate measured results from design expectations.

## Breaking changes and migration

State “None” or list each required caller/operator migration with a before and
after example.

## Compatibility

- Supported Python versions:
- Supported MCP protocol/SDK range:
- Supported W&B SDK/server range:

## Operator changes

List new settings, defaults, deployment constraints, and rollback effects.
Link to generated configuration and tool-profile output rather than copying
transient counts or policy values.

## Security and privacy

Describe the actual enforced boundary and the scans performed. Avoid absolute
claims such as “anonymous” or “no data collected” unless the implementation and
tests prove them for every mode.

## Known limitations

List bounded behavior, optional compatibility paths, and unavailable channels.

## Validation

Link public attestations and summarize completed compatibility, security, and
installation checks. State missing checks and known findings. Keep private
workflow links, rollout procedures, customer identifiers, and infrastructure
details out of public release notes.

## Immutable artifacts

- Signed source tag: Pending
- Source commit/tree: Pending
- GitHub Release: Pending (draft until channel evidence is complete)
- Wheel and source-distribution checksums: Pending
- Public qualification/provenance attestations: Pending
- Public container repository and digest: Pending
- Helm chart version: Pending

## Rollback

- Public source: fix forward from the signed tag.
- Dedicated/Self-Managed: restore the previous chart and image digest.
