<script lang="ts">
    import { Button, Spinner, Typography } from '$lib/components';
    import ConfirmDeleteDialog from '$lib/components/ConfirmDeleteDialog/ConfirmDeleteDialog.svelte';
    import { GitMerge, Rocket, X } from '@lucide/svelte';
    import {
        usePipelineDatasetActions,
        usePipelineDatasets
    } from '$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte';
    import {
        useTrainingActions,
        useTrainingOptions
    } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';
    import DatasetRow from './DatasetRow.svelte';
    import MergeDatasetsDialog from './MergeDatasetsDialog.svelte';
    import RunsSection from './RunsSection.svelte';
    import TrainModelDialog from './TrainModelDialog.svelte';

    interface Props {
        onClose: () => void;
    }
    let { onClose }: Props = $props();

    const query = usePipelineDatasets();
    const { mergeMutation, deleteMutation, preview, merge, remove } = usePipelineDatasetActions();
    const options = useTrainingOptions();
    const { startMutation, start } = useTrainingActions();
    const datasets = $derived(query.data ?? []);
    const datasetNames = $derived(datasets.map((d) => d.name));

    let mergeOpen = $state(false);
    let trainOpen = $state(false);
    let toDelete = $state<string | null>(null);

    $effect(() => {
        if (!mergeMutation.isPending && mergeMutation.isSuccess) mergeOpen = false;
    });
    $effect(() => {
        if (!startMutation.isPending && startMutation.isSuccess) trainOpen = false;
    });
    $effect(() => {
        if (!deleteMutation.isPending && deleteMutation.isSuccess) toDelete = null;
    });
</script>

<div class="flex min-h-0 flex-1 flex-col rounded-[1vw] bg-card p-4" data-testid="train-panel">
    <div class="mb-3 flex items-center justify-between">
        <Typography variant="h5" component="h2">Train</Typography>
        <Button
            icon={X}
            ariaLabel="Close train panel"
            buttonProps={{ size: 'icon', onclick: onClose, class: 'h-8 w-8' }}
        />
    </div>
    <p class="mb-3 text-sm text-muted-foreground">
        Datasets prepared with Export → "Prepare COCO dataset". Preview opens a dataset in Studio in
        a new tab; merge combines two; "Train model" starts a YOLOX run on one.
    </p>
    <div class="mb-3 flex gap-2">
        <Button
            variant="default"
            icon={GitMerge}
            buttonProps={{
                size: 'sm',
                disabled: datasets.length < 2,
                onclick: () => (mergeOpen = true),
                'data-testid': 'merge-datasets'
            }}
        >
            Merge datasets
        </Button>
        <Button
            variant="outline"
            icon={Rocket}
            disabledReason={datasets.length === 0 ? 'Prepare a dataset first.' : undefined}
            buttonProps={{
                size: 'sm',
                disabled: datasets.length === 0 || !options.data,
                onclick: () => (trainOpen = true),
                'data-testid': 'train-model'
            }}
        >
            Train model
        </Button>
    </div>
    <div class="flex min-h-0 flex-1 flex-col overflow-y-auto dark:[color-scheme:dark]">
        <Typography variant="subtitle2" component="h3" className="mb-2">Datasets</Typography>
        {#if query.isLoading}
            <Spinner size="small" align="center" />
        {:else if query.error}
            <Typography variant="body2" className="text-red-500"
                >Error loading datasets: {query.error.message}</Typography
            >
        {:else if datasets.length === 0}
            <Typography variant="body2" className="text-muted-foreground">
                No datasets yet. Use the Export menu and choose "Prepare COCO dataset".
            </Typography>
        {:else}
            <ul class="flex flex-col gap-2" data-testid="datasets-list">
                {#each datasets as dataset (dataset.name)}
                    <DatasetRow
                        {dataset}
                        onPreview={() => preview(dataset.name)}
                        onDelete={() => (toDelete = dataset.name)}
                    />
                {/each}
            </ul>
        {/if}
        <RunsSection />
    </div>
</div>

{#if mergeOpen}
    <MergeDatasetsDialog
        {datasetNames}
        isPending={mergeMutation.isPending}
        onConfirm={merge}
        onCancel={() => (mergeOpen = false)}
    />
{/if}
{#if trainOpen && options.data}
    <TrainModelDialog
        options={options.data}
        {datasetNames}
        isPending={startMutation.isPending}
        onConfirm={start}
        onCancel={() => (trainOpen = false)}
    />
{/if}
{#if toDelete}
    <ConfirmDeleteDialog
        title="Delete dataset?"
        description={`The folder of "${toDelete}" is removed. The images it was exported from stay in Studio.`}
        isPending={deleteMutation.isPending}
        onConfirm={() => toDelete && remove(toDelete)}
        onCancel={() => (toDelete = null)}
    />
{/if}
