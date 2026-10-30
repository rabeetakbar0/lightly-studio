<script lang="ts">
    import type { PipelineModelView, TrainConfig } from '$lib/api/lightly_studio_local';
    import { Button, Spinner, Typography } from '$lib/components';
    import TrainModelDialog from '$lib/components/TrainPanel/TrainModelDialog.svelte';
    import { Columns3, X } from '@lucide/svelte';
    import { usePipelineDatasets } from '$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte';
    import { usePipelineModels } from '$lib/hooks/usePipelineModels/usePipelineModels.svelte';
    import {
        useTrainingActions,
        useTrainingOptions
    } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';
    import CompareModelsDialog from './CompareModelsDialog.svelte';
    import ModelRow from './ModelRow.svelte';

    interface Props {
        onClose: () => void;
    }
    let { onClose }: Props = $props();

    const query = usePipelineModels();
    const datasets = usePipelineDatasets();
    const options = useTrainingOptions();
    const { startMutation, start } = useTrainingActions();
    const models = $derived(query.data ?? []);
    const datasetNames = $derived((datasets.data ?? []).map((d) => d.name));
    const presets = $derived(options.data?.presets ?? {});

    let selected = $state<string[]>([]);
    let compareOpen = $state(false);
    let trainFrom = $state<PipelineModelView | null>(null);
    const compared = $derived(models.filter((m) => selected.includes(m.name)));
    // The new run starts from the model and repeats its settings when it has a card.
    const initialConfig = $derived.by((): Partial<TrainConfig> | undefined => {
        if (!trainFrom) return undefined;
        const size =
            trainFrom.depth != null && trainFrom.width != null
                ? { depth: trainFrom.depth, width: trainFrom.width }
                : {};
        return { ...(trainFrom.card?.config ?? {}), ...size, baseline: `model:${trainFrom.name}` };
    });

    const toggle = (name: string, on: boolean) => {
        selected = on ? [...selected, name] : selected.filter((n) => n !== name);
    };
    $effect(() => {
        if (!startMutation.isPending && startMutation.isSuccess) trainFrom = null;
    });
</script>

<div class="flex min-h-0 flex-1 flex-col rounded-[1vw] bg-card p-4" data-testid="models-panel">
    <div class="mb-3 flex items-center justify-between">
        <Typography variant="h5" component="h2">Models</Typography>
        <Button
            icon={X}
            ariaLabel="Close models panel"
            buttonProps={{ size: 'icon', onclick: onClose, class: 'h-8 w-8' }}
        />
    </div>
    <p class="mb-3 text-sm text-muted-foreground">
        Every model in the models folder, each available as a "Detect" operator. Models trained here
        record their dataset, settings and scores. Select two or more to compare them.
    </p>
    <div class="mb-3 flex gap-2">
        <Button
            variant="default"
            icon={Columns3}
            buttonProps={{
                size: 'sm',
                disabled: compared.length < 2,
                onclick: () => (compareOpen = true),
                'data-testid': 'compare-models'
            }}
        >
            Compare selected ({compared.length})
        </Button>
    </div>
    <div class="flex min-h-0 flex-1 flex-col overflow-y-auto dark:[color-scheme:dark]">
        {#if query.isLoading}
            <Spinner size="medium" align="center" />
        {:else if query.error}
            <Typography variant="body2" className="text-red-500"
                >Error loading models: {query.error.message}</Typography
            >
        {:else if models.length === 0}
            <Typography variant="body2" className="text-muted-foreground">
                No models yet. Drop an ONNX model with its class list into the models folder, or
                train one from the Train panel.
            </Typography>
        {:else}
            <ul class="flex flex-col gap-2" data-testid="models-list">
                {#each models as model (model.name)}
                    <ModelRow
                        {model}
                        {presets}
                        selected={selected.includes(model.name)}
                        canTrain={datasetNames.length > 0 && !!options.data}
                        onSelect={(on) => toggle(model.name, on)}
                        onTrainFrom={() => (trainFrom = model)}
                    />
                {/each}
            </ul>
        {/if}
    </div>
</div>

{#if compareOpen}
    <CompareModelsDialog models={compared} onClose={() => (compareOpen = false)} />
{/if}
{#if trainFrom && options.data}
    <TrainModelDialog
        options={options.data}
        {datasetNames}
        {initialConfig}
        isPending={startMutation.isPending}
        onConfirm={start}
        onCancel={() => (trainFrom = null)}
    />
{/if}
