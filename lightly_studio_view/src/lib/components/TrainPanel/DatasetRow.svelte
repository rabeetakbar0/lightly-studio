<script lang="ts">
    import type { PipelineDatasetView } from '$lib/api/lightly_studio_local';
    import { Button, Typography } from '$lib/components';
    import { Eye, Trash2 } from '@lucide/svelte';
    import { formatDate } from '$lib/utils';
    import { formatBytes } from '$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte';

    interface Props {
        dataset: PipelineDatasetView;
        onPreview: () => void;
        onDelete: () => void;
    }
    let { dataset, onPreview, onDelete }: Props = $props();

    const splits = $derived(
        Object.entries(dataset.splits)
            .map(([split, counts]) => `${split.replace('2017', '')} ${counts.images}`)
            .join(' · ')
    );
</script>

<li class="rounded-md border border-border bg-background p-3" data-testid="dataset-row">
    <div class="flex items-start justify-between gap-2">
        <div class="min-w-0">
            <Typography variant="subtitle2" className="truncate">{dataset.name}</Typography>
            <Typography variant="caption" className="text-muted-foreground">
                {dataset.images} images · {dataset.boxes} boxes · {splits} · {formatBytes(
                    dataset.size_bytes
                )}
            </Typography>
            <Typography variant="caption" className="block text-muted-foreground">
                {dataset.classes.join(', ') || 'no classes'}
                {#if dataset.sources.length > 0}
                    · merged from {dataset.sources.join(' + ')}
                {:else if dataset.annotation_source}
                    · from "{dataset.annotation_source}"
                {/if}
                · {formatDate(new Date(dataset.created_at))}
            </Typography>
        </div>
        <div class="flex shrink-0 gap-1">
            <Button
                icon={Eye}
                ariaLabel={`Preview ${dataset.name}`}
                buttonProps={{ size: 'icon', title: 'Preview in a new tab', onclick: onPreview }}
            />
            <Button
                icon={Trash2}
                ariaLabel={`Delete ${dataset.name}`}
                buttonProps={{ size: 'icon', title: 'Delete dataset', onclick: onDelete }}
            />
        </div>
    </div>
</li>
