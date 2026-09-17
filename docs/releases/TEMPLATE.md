# W&B MCP Server vX.Y.Z

**Release date:** Pending

## Availability

| Channel | State | Immutable artifact |
|---|---|---|
| Public source | Candidate | Pending signed tag and GitHub Release |
| W&B-hosted | Pending | Pending verified image digest |
| Dedicated/Self-Managed | Pending | Pending verified image digest and chart |
| Customer container | Pending | Pending verified public-registry digest |

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

Link the signed public attestation and summarize exact-profile, compatibility,
security, staging, and rollback evidence without customer or infrastructure
identifiers.

## Immutable artifacts

- Signed source tag: Pending
- Source commit/tree: Pending
- GitHub Release: Pending (draft until channel evidence is complete)
- Wheel and source-distribution checksums: Pending
- Public qualification/provenance attestations: Pending
- Hosted image digest: Pending
- Customer image digest: Pending
- Helm chart version: Pending

## Rollback

- Public source: fix forward from the signed tag.
- Managed: restore and verify the captured previous configuration and traffic.
- Dedicated/Self-Managed: restore the previous chart and image digest.
