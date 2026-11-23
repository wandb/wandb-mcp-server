import { z } from "zod";

export const PointSchema = z.object({
  x: z.number(),
  y: z.number(),
  y0: z.number().optional(),
});

export const SeriesEntrySchema = z.object({
  seriesId: z.string(),
  metadataId: z.string(),
  role: z.enum(["primary", "original", "band"]),
  kind: z.enum(["line", "area"]),
  metricName: z.string(),
  groupKey: z.string().optional(),
  historySeries: z.array(PointSchema),
  rawHistorySeries: z.array(PointSchema).optional(),
});

export const SeriesStyleSchema = z.object({
  color: z.string(),
  lineWidth: z.number(),
  mark: z.enum(["solid", "dashed", "dotted", "dotdash", "dotdotdash", "points"]),
  colorKey: z.string(),
});

export const SeriesMetadataSchema = z.object({
  displayName: z.string(),
  title: z.string(),
  runName: z.string().nullable().optional(),
  runState: z.string().nullable().optional(),
  legendHref: z.string(),
  entityName: z.string(),
  projectName: z.string(),
});

export const RunLifecycleSchema = z.object({
  displayName: z.string(),
  state: z.string(),
  active: z.boolean(),
  group: z.unknown().optional(),
  url: z.string(),
});

export const SmoothingTypeSchema = z.enum([
  "none",
  "exponential",
  "exponentialTimeWeighted",
  "gaussian",
  "average",
]);

export const GraphPayloadSchema = z.object({
  schema: z.literal("wandb.graph/v1"),
  series: z.array(SeriesEntrySchema),
  seriesStylesById: z.record(z.string(), SeriesStyleSchema),
  seriesMetadataById: z.record(z.string(), SeriesMetadataSchema),
  runsById: z.record(z.string(), RunLifecycleSchema).default({}),
  hasActiveRuns: z.boolean().default(false),
  axis: z.object({
    xKey: z.string(),
    xIsDatetime: z.boolean(),
    xScale: z.enum(["linear", "log"]),
    yScale: z.enum(["linear", "log"]),
    xTitle: z.string(),
    yTitle: z.string(),
  }),
  xDomain: z.tuple([z.number().nullable(), z.number().nullable()]),
  yDomain: z.tuple([z.number().nullable(), z.number().nullable()]),
  title: z.string(),
  legend: z.object({
    show: z.boolean(),
    position: z.enum(["north", "south", "east", "west"]),
  }),
  effectiveConfig: z
    .object({
      metrics: z.array(z.string()),
      xAxis: z.string(),
      smoothingWeight: z.number(),
      smoothingType: SmoothingTypeSchema,
      showOriginalAfterSmoothing: z.boolean().optional(),
    })
    .passthrough(),
  coverage: z.object({
    sampled: z.boolean(),
    truncated: z.boolean(),
    ignoredConfig: z.array(z.string()).default([]),
    perRunErrors: z
      .array(
        z.object({
          runId: z.string(),
          error: z.string(),
          url: z.string(),
        }),
      )
      .default([]),
    groupingAlignment: z.string().optional(),
    grouped: z.boolean(),
    autoDiscoveredRuns: z.boolean().default(false),
  }),
  url: z.string(),
  workspaceUrl: z.string().optional(),
  range_cap: z.number(),
  arguments: z.object({
    entity_name: z.string(),
    project_name: z.string(),
    run_ids: z.array(z.string()),
    config: z.record(z.string(), z.unknown()),
    samples: z.number(),
    min_step: z.number().nullable().optional(),
    max_step: z.number().nullable().optional(),
  }),
});

export const ToolErrorPayloadSchema = z.object({
  error: z.string(),
  message: z.string(),
});

export type Point = z.infer<typeof PointSchema>;
export type SeriesEntry = z.infer<typeof SeriesEntrySchema>;
export type SeriesStyle = z.infer<typeof SeriesStyleSchema>;
export type SeriesMetadata = z.infer<typeof SeriesMetadataSchema>;
export type RunLifecycle = z.infer<typeof RunLifecycleSchema>;
export type SmoothingType = z.infer<typeof SmoothingTypeSchema>;
export type GraphPayload = z.infer<typeof GraphPayloadSchema>;
