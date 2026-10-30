import { createQuery } from '@tanstack/svelte-query';
import type { PipelineModelView } from '$lib/api/lightly_studio_local';
import { listPipelineModelsOptions } from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';

/** The detection models in the models folder with their training history, newest first. */
export const usePipelineModels = () => createQuery(() => listPipelineModelsOptions());

/** "YOLOX-M" for a known size, otherwise the raw depth and width, or "size unknown". */
export const sizeLabel = (
    model: Pick<PipelineModelView, 'depth' | 'width'>,
    presets: Record<string, number[]>
): string => {
    if (model.depth === null || model.width === null) return 'size unknown';
    const match = Object.entries(presets).find(
        ([, [d, w]]) => d === model.depth && w === model.width
    );
    return match
        ? `YOLOX-${match[0].toUpperCase()}`
        : `depth ${model.depth} · width ${model.width}`;
};

/** "yolox_m.pth" from "weights:yolox_m.pth", or "random init" without a baseline. */
export const baselineLabel = (baseline: string | null | undefined): string =>
    baseline ? baseline.slice(baseline.indexOf(':') + 1) : 'random init';
