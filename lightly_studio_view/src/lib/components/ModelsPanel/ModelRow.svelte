<script lang="ts">
    import type { PipelineModelView } from '$lib/api/lightly_studio_local';
    import { Button, Typography } from '$lib/components';
    import { Badge } from '$lib/components/ui/badge';
    import { Checkbox } from '$lib/components/ui/checkbox';
    import { Rocket } from '@lucide/svelte';
    import { formatDate } from '$lib/utils';
    import { formatBytes } from '$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte';
    import {
        baselineLabel,
        sizeLabel
    } from '$lib/hooks/usePipelineModels/usePipelineModels.svelte';

    interface Props {
        model: PipelineModelView;
        presets: Record<string, number[]>;
        selected: boolean;
        canTrain: boolean;
        onSelect: (selected: boolean) => void;
        onTrainFrom: () => void;
    }
    let { model, presets, selected, canTrain, onSelect, onTrainFrom }: Props = $props();

    const card = $derived(model.card);
</script>

<li class="rounded-md border border-border bg-background p-3" data-testid="model-row">
    <div class="flex items-start gap-3">
        <Checkbox
            checked={selected}
            onCheckedChange={(v: boolean) => onSelect(v)}
            aria-label={`Select ${model.name} for comparison`}
            class="mt-1"
        />
        <div class="min-w-0 flex-1">
            <div class="flex flex-wrap items-center gap-2">
                <Typography variant="subtitle2" className="truncate">{model.name}</Typography>
                <Badge variant="secondary">{sizeLabel(model, presets)}</Badge>
                {#if !model.has_checkpoint}
                    <Badge
                        variant="outline"
                        title="Link its best_ckpt.pth next to the ONNX to train from it"
                    >
                        no checkpoint
                    </Badge>
                {/if}
            </div>
            <Typography variant="caption" className="block text-muted-foreground">
                {model.classes.join(', ')} · {formatBytes(model.size_bytes)} · {formatDate(
                    new Date(model.modified_at)
                )}
            </Typography>
            {#if card}
                <Typography variant="caption" className="block text-muted-foreground">
                    Trained on {card.dataset.name} ({card.dataset.images} images, {card.dataset
                        .boxes} boxes{card.dataset.annotation_source
                        ? `, from "${card.dataset.annotation_source}"`
                        : ''}) · from {baselineLabel(card.baseline)} · {card.trained_epochs} epochs
                    {#if card.best_ap50_95 !== null}
                        · best AP {(card.best_ap50_95 * 100).toFixed(1)}
                    {/if}
                </Typography>
            {:else}
                <Typography variant="caption" className="block text-muted-foreground">
                    No training history: trained outside the pipeline.
                </Typography>
            {/if}
        </div>
        <Button
            icon={Rocket}
            ariaLabel={`Train from ${model.name}`}
            disabledReason={!model.has_checkpoint
                ? 'No checkpoint next to the ONNX file.'
                : !canTrain
                  ? 'Prepare a dataset first.'
                  : undefined}
            buttonProps={{
                size: 'icon',
                title: 'Train a new model from this one',
                disabled: !model.has_checkpoint || !canTrain,
                onclick: onTrainFrom
            }}
        />
    </div>
</li>
